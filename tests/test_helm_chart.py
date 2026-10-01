"""The chart deploys one GPU pod with persistent storage and in-cluster ports."""

import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CHART = ROOT / "Charts" / "lllm2"

# CI installs Helm, so the chart tests never skip there.
needs_helm = pytest.mark.skipif(
    shutil.which("helm") is None and os.environ.get("GITHUB_ACTIONS") != "true",
    reason="Helm is not installed",
)


def render(*options, chart=CHART):
    result = subprocess.run(
        ["helm", "template", "lllm2", str(chart), *options],
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        (item["kind"], item["metadata"]["name"]): item
        for item in yaml.safe_load_all(result.stdout)
        if item
    }


def pod(objects):
    return objects["Deployment", "lllm2"]["spec"]["template"]["spec"]


@needs_helm
def test_defaults_request_one_gpu_and_mount_persistent_models_and_state():
    objects = render()
    deployment = objects["Deployment", "lllm2"]
    assert deployment["spec"]["replicas"] == 1
    assert deployment["spec"]["strategy"] == {"type": "Recreate"}
    container = pod(objects)["containers"][0]
    assert container["resources"] == {"limits": {"nvidia.com/gpu": 1}}
    # A chart from a checkout runs the moving main image, so it pulls each start.
    assert container["image"] == "ghcr.io/gilesknap/lllm2:main"
    assert container["imagePullPolicy"] == "Always"
    mounts = {m["mountPath"]: m["name"] for m in container["volumeMounts"]}
    assert mounts == {"/models": "models", "/data": "data"}
    claims = {
        v["name"]: v["persistentVolumeClaim"]["claimName"]
        for v in pod(objects)["volumes"]
    }
    assert claims == {"models": "lllm2-models", "data": "lllm2-data"}
    for name in claims.values():
        claim = objects["PersistentVolumeClaim", name]
        assert claim["metadata"]["annotations"]["helm.sh/resource-policy"] == "keep"
        assert claim["spec"]["accessModes"] == ["ReadWriteOnce"]
    assert pod(objects)["securityContext"]["runAsNonRoot"] is True


@needs_helm
def test_engine_port_is_served_inside_the_cluster_and_the_panel_only_in_the_pod():
    objects = render()
    service = objects["Service", "lllm2"]
    assert service["spec"]["type"] == "ClusterIP"
    assert service["spec"]["ports"] == [
        {"name": "engine", "port": 1920, "targetPort": "engine"}
    ]
    container = pod(objects)["containers"][0]
    assert container["ports"] == [{"name": "engine", "containerPort": 1920}]
    assert {"name": "LLLM2_ENGINE_HOST", "value": "0.0.0.0"} in container["env"]
    assert container["args"] == ["panel", "--host", "127.0.0.1", "--port", "8082"]
    # Nothing reaches outside the cluster: no Ingress, NodePort or LoadBalancer.
    assert {kind for kind, _ in objects} == {
        "Deployment",
        "Service",
        "PersistentVolumeClaim",
    }


@needs_helm
def test_the_panel_can_join_the_service_on_request():
    objects = render("--set", "panel.host=0.0.0.0")
    container = pod(objects)["containers"][0]
    assert container["args"] == ["panel", "--host", "0.0.0.0", "--port", "8082"]
    assert {"name": "panel", "containerPort": 8082} in container["ports"]
    service = objects["Service", "lllm2"]["spec"]
    assert service["type"] == "ClusterIP"
    assert {"name": "panel", "port": 8082, "targetPort": "panel"} in service["ports"]


@needs_helm
def test_every_optional_feature_renders():
    objects = render("-f", str(CHART / "ci" / "all-features-values.yaml"))
    spec = pod(objects)
    assert spec["runtimeClassName"] == "nvidia"
    assert spec["nodeSelector"] == {"nvidia.com/gpu.product": "NVIDIA-L40S"}
    assert spec["tolerations"][0]["key"] == "nvidia.com/gpu"
    assert spec["imagePullSecrets"] == [{"name": "mirror-pull"}]
    assert spec["securityContext"]["runAsUser"] == 2000
    container = spec["containers"][0]
    assert container["image"] == "registry.example.com/mirror/lllm2:0.11.0"
    assert container["imagePullPolicy"] == "Always"
    assert container["resources"] == {
        "limits": {"memory": "64Gi", "nvidia.com/gpu": 1},
        "requests": {"cpu": "4", "memory": "32Gi"},
    }
    assert {
        "name": "HTTPS_PROXY",
        "value": "http://proxy.example.com:3128",
    } in container["env"]
    claims = {
        v["name"]: v["persistentVolumeClaim"]["claimName"] for v in spec["volumes"]
    }
    assert claims == {"models": "shared-models", "data": "lllm2-data"}
    assert ("PersistentVolumeClaim", "shared-models") not in objects
    data = objects["PersistentVolumeClaim", "lllm2-data"]["spec"]
    assert data["storageClassName"] == "fast-local"
    assert data["resources"]["requests"]["storage"] == "50Gi"


@needs_helm
def test_a_gpu_count_in_the_values_replaces_the_default():
    objects = render("--set", "resources.limits.nvidia\\.com/gpu=2")
    container = pod(objects)["containers"][0]
    assert container["resources"]["limits"] == {"nvidia.com/gpu": 2}


@needs_helm
@pytest.mark.parametrize(
    ("option", "message"),
    [
        ("ingress.enabled=true", "Additional property ingress is not allowed"),
        ("panel.host=10.0.0.1", "panel.host must be one of the following"),
    ],
)
def test_schema_rejects_values_the_chart_does_not_support(option, message):
    result = subprocess.run(
        ["helm", "template", "lllm2", str(CHART), "--set", option],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert message in result.stderr


@needs_helm
def test_a_release_package_runs_the_release_image(tmp_path):
    subprocess.run(
        ["helm", "package", "-d", str(tmp_path), "--version", "0.11.0"]
        + ["--app-version", "0.11.0", str(CHART)],
        capture_output=True,
        check=True,
    )
    package = tmp_path / "lllm2-0.11.0.tgz"
    with tarfile.open(package) as archive:
        names = set(archive.getnames())
    assert "lllm2/values.schema.json" in names
    # .helmignore keeps the schema generator's config and CI values out.
    assert not [n for n in names if n.startswith(("lllm2/ci/", "lllm2/.schema"))]
    container = pod(render(chart=package))["containers"][0]
    assert container["image"] == "ghcr.io/gilesknap/lllm2:0.11.0"
    assert container["imagePullPolicy"] == "IfNotPresent"


def test_the_published_schema_is_the_charts_values_schema():
    schema = ROOT / "schemas" / "lllm2.schema.json"
    assert schema.is_symlink()
    assert schema.resolve() == (CHART / "values.schema.json").resolve()
