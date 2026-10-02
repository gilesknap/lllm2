"""The chart deploys one GPU pod with persistent storage and in-cluster ports."""

import base64
import json
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


def refusal(*options):
    result = subprocess.run(
        ["helm", "template", "lllm2", str(CHART), *options],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    return result.stderr


def pod(objects):
    return objects["Deployment", "lllm2"]["spec"]["template"]["spec"]


def containers(objects):
    return {c["name"]: c for c in pod(objects)["containers"]}


def env(container):
    return {e["name"]: e.get("value", e.get("valueFrom")) for e in container["env"]}


def sets(values):
    return [arg for key, value in values.items() for arg in ("--set", f"{key}={value}")]


def without(values, key):
    return {k: v for k, v in values.items() if k != key}


def render_notes(tmp_path, *options):
    """Render NOTES.txt, which helm template leaves out, through a ConfigMap."""
    chart = tmp_path / "lllm2"
    shutil.copytree(CHART, chart)
    notes = chart / "templates" / "NOTES.txt"
    (chart / "templates" / "_notes_probe.tpl").write_text(
        '{{- define "probe.notes" -}}\n' + notes.read_text() + "\n{{- end -}}\n"
    )
    notes.unlink()
    (chart / "templates" / "notes-probe.yaml").write_text(
        "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: notes-probe\n"
        'data:\n  notes: {{ include "probe.notes" . | quote }}\n'
    )
    return render(*options, chart=chart)["ConfigMap", "notes-probe"]["data"]["notes"]


# The panel behind an Ingress that authenticates, and behind the sign-in
# sidecar, with the fewest values each mode needs.
EXTERNAL = {
    "panel.host": "0.0.0.0",
    "ingress.enabled": "true",
    "ingress.auth": "external",
    "ingress.hosts[0].host": "lllm2.example.com",
}
OIDC = {
    "ingress.enabled": "true",
    "ingress.auth": "oidc",
    "ingress.hosts[0].host": "lllm2.example.com",
    "ingress.oidc.issuerURL": "https://keycloak.example.com/realms/lab",
    "ingress.oidc.clientID": "lllm2",
    "ingress.oidc.existingSecret": "lllm2-oidc",
    "ingress.oidc.allowedGroups[0]": "/lllm2-users",
}


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
    assert "LLLM2_PANEL_ALLOWED_HOSTS" not in env(container)
    assert len(pod(objects)["containers"]) == 1
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
    ingress = objects["Ingress", "lllm2"]
    assert [rule["host"] for rule in ingress["spec"]["rules"]] == [
        "lllm2.example.com",
        "gpu1.lllm2.example.com",
    ]
    assert env(container)["LLLM2_PANEL_ALLOWED_HOSTS"] == (
        "lllm2.example.com,gpu1.lllm2.example.com"
    )
    policy = objects["NetworkPolicy", "lllm2"]["spec"]
    assert policy["ingress"][1]["ports"] == [{"port": 8082}]
    assert len(policy["ingress"][1]["from"]) == 2


@needs_helm
def test_a_gpu_count_in_the_values_replaces_the_default():
    objects = render("--set", "resources.limits.nvidia\\.com/gpu=2")
    container = pod(objects)["containers"][0]
    assert container["resources"]["limits"] == {"nvidia.com/gpu": 2}


@needs_helm
@pytest.mark.parametrize(
    ("option", "message"),
    [
        ("panel.host=10.0.0.1", "panel.host must be one of the following"),
        ("ingress.auth=basic", "ingress.auth must be one of the following"),
        ("ingress.hosts[0].host=*.example.com", "Does not match pattern"),
        ("ingress.hosts[0].host=LLLM2.example.com", "Does not match pattern"),
        # The panel serves only at /, so there is no path to set.
        ("ingress.hosts[0].paths[0].path=/lllm2", "Additional property paths"),
        # Annotations are strings, so a timeout must be quoted.
        (
            "ingress.annotations.nginx\\.ingress\\.kubernetes\\.io/proxy-read-timeout=200",
            "Expected: string, given: integer",
        ),
        ("ingress.oidc.provider=google", "must be one of the following"),
        ("ingress.oidc.issuerURL=https://kc.example.com/realms/lab/", "pattern"),
        ("ingress.oidc.issuerURL=kc.example.com/realms/lab", "pattern"),
        # Secrets stay in a Secret, never in values.
        ("ingress.oidc.clientSecret=x", "Additional property clientSecret"),
        ("networkPolicy.from[0].podSelectors.x=y", "Additional property podSelectors"),
    ],
)
def test_schema_rejects_values_the_chart_does_not_support(option, message):
    assert message in refusal("--set", option)


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


@needs_helm
def test_the_ingress_serves_the_panel_at_root_on_each_host():
    objects = render(
        *sets(EXTERNAL),
        *sets(
            {
                "ingress.className": "nginx",
                "ingress.hosts[1].host": "gpu1.lllm2.example.com",
                "ingress.tls[0].secretName": "lllm2-tls",
                "ingress.tls[0].hosts[0]": "lllm2.example.com",
            }
        ),
        "--set-string",
        "ingress.annotations.nginx\\.ingress\\.kubernetes\\.io/proxy-read-timeout=200",
    )
    ingress = objects["Ingress", "lllm2"]
    assert ingress["apiVersion"] == "networking.k8s.io/v1"
    assert ingress["metadata"]["annotations"] == {
        "nginx.ingress.kubernetes.io/proxy-read-timeout": "200"
    }
    spec = ingress["spec"]
    assert spec["ingressClassName"] == "nginx"
    assert spec["tls"] == [{"secretName": "lllm2-tls", "hosts": ["lllm2.example.com"]}]
    assert [rule["host"] for rule in spec["rules"]] == [
        "lllm2.example.com",
        "gpu1.lllm2.example.com",
    ]
    # One path per host, so the model API's port never rides the Ingress.
    for rule in spec["rules"]:
        assert rule["http"]["paths"] == [
            {
                "path": "/",
                "pathType": "Prefix",
                "backend": {"service": {"name": "lllm2", "port": {"name": "panel"}}},
            }
        ]
    container = containers(objects)["lllm2"]
    assert container["args"] == ["panel", "--host", "0.0.0.0", "--port", "8082"]
    assert env(container)["LLLM2_PANEL_ALLOWED_HOSTS"] == (
        "lllm2.example.com,gpu1.lllm2.example.com"
    )
    assert list(containers(objects)) == ["lllm2"]
    service = objects["Service", "lllm2"]["spec"]
    assert [port["name"] for port in service["ports"]] == ["engine", "panel"]
    assert "publishNotReadyAddresses" not in service


@needs_helm
def test_a_minimal_ingress_uses_the_default_class_and_no_tls():
    spec = render(*sets(EXTERNAL))["Ingress", "lllm2"]["spec"]
    assert "ingressClassName" not in spec
    assert "tls" not in spec


@needs_helm
def test_an_unauthenticated_ingress_renders_only_on_request():
    objects = render(*sets({**EXTERNAL, "ingress.auth": "none"}))
    assert ("Ingress", "lllm2") in objects


@needs_helm
def test_turning_the_ingress_off_drops_it_and_everything_it_needs():
    objects = render(*sets({**OIDC, "ingress.enabled": "false"}))
    assert {kind for kind, _ in objects} == {
        "Deployment",
        "Service",
        "PersistentVolumeClaim",
    }
    assert list(containers(objects)) == ["lllm2"]
    assert "LLLM2_PANEL_ALLOWED_HOSTS" not in env(containers(objects)["lllm2"])
    service = objects["Service", "lllm2"]["spec"]
    assert [port["name"] for port in service["ports"]] == ["engine"]


@needs_helm
def test_oidc_mode_signs_users_in_before_the_panel():
    objects = render(*sets(OIDC))
    assert list(containers(objects)) == ["lllm2", "oauth2-proxy"]
    panel = containers(objects)["lllm2"]
    # The panel stays on loopback, so only the sidecar reaches it.
    assert panel["args"] == ["panel", "--host", "127.0.0.1", "--port", "8082"]
    assert panel["ports"] == [{"name": "engine", "containerPort": 1920}]
    assert env(panel)["LLLM2_PANEL_ALLOWED_HOSTS"] == "lllm2.example.com"
    sidecar = containers(objects)["oauth2-proxy"]
    assert sidecar["image"] == "quay.io/oauth2-proxy/oauth2-proxy:v7.15.5"
    assert sidecar["imagePullPolicy"] == "IfNotPresent"
    assert sidecar["args"] == [
        "--provider=oidc",
        "--oidc-issuer-url=https://keycloak.example.com/realms/lab",
        "--client-id=lllm2",
        "--http-address=:4180",
        "--upstream=http://127.0.0.1:8082/",
        "--code-challenge-method=S256",
        "--skip-provider-button=true",
        "--upstream-timeout=300s",
        "--cookie-refresh=1m",
        "--api-route=^/api/",
        "--cookie-csrf-per-request=true",
        "--cookie-csrf-per-request-limit=5",
        "--silence-ping-logging=true",
        "--allowed-group=/lllm2-users",
        "--email-domain=*",
    ]
    # Secrets come from the Secret, never from arguments in the pod spec.
    assert env(sidecar) == {
        "OAUTH2_PROXY_CLIENT_SECRET": {
            "secretKeyRef": {"name": "lllm2-oidc", "key": "client-secret"}
        },
        "OAUTH2_PROXY_COOKIE_SECRET": {
            "secretKeyRef": {"name": "lllm2-oidc", "key": "cookie-secret"}
        },
    }
    assert sidecar["ports"] == [{"name": "auth", "containerPort": 4180}]
    assert sidecar["livenessProbe"] == {"httpGet": {"path": "/ping", "port": "auth"}}
    assert sidecar["readinessProbe"] == {"httpGet": {"path": "/ready", "port": "auth"}}
    assert sidecar["securityContext"] == {
        "allowPrivilegeEscalation": False,
        "readOnlyRootFilesystem": True,
        "capabilities": {"drop": ["ALL"]},
    }
    assert sidecar["resources"] == {
        "requests": {"cpu": "10m", "memory": "32Mi"},
        "limits": {"memory": "128Mi"},
    }
    assert "volumeMounts" not in sidecar
    assert [v["name"] for v in pod(objects)["volumes"]] == ["models", "data"]
    service = objects["Service", "lllm2"]["spec"]
    assert service["ports"] == [
        {"name": "engine", "port": 1920, "targetPort": "engine"},
        {"name": "auth", "port": 4180, "targetPort": "auth"},
    ]
    # The model API stays on the Service while the sidecar starts or fails.
    assert service["publishNotReadyAddresses"] is True
    for rule in objects["Ingress", "lllm2"]["spec"]["rules"]:
        (path,) = rule["http"]["paths"]
        assert path["backend"]["service"]["port"] == {"name": "auth"}


@needs_helm
def test_oidc_values_reach_the_sidecar():
    objects = render("-f", str(CHART / "ci" / "oidc-values.yaml"))
    sidecar = containers(objects)["oauth2-proxy"]
    args = sidecar["args"]
    assert "--provider=keycloak-oidc" in args
    assert [a for a in args if a.startswith("--allowed-")] == [
        "--allowed-group=/lllm2-users",
        "--allowed-group=/gpu-admins",
        "--allowed-role=lllm2:operator",
    ]
    assert [a for a in args if a.startswith("--email-domain")] == [
        "--email-domain=example.com"
    ]
    assert "--provider-ca-file=/etc/oauth2-proxy/ca/ca.crt" in args
    # extraArgs come last, so a repeated flag overrides the chart's.
    assert args[-1] == "--cookie-expire=12h"
    assert sidecar["image"] == "registry.example.com/mirror/oauth2-proxy:v7.15.5"
    assert sidecar["imagePullPolicy"] == "Always"
    assert list(env(sidecar)) == [
        "OAUTH2_PROXY_CLIENT_SECRET",
        "OAUTH2_PROXY_COOKIE_SECRET",
        "HTTPS_PROXY",
        "NO_PROXY",
    ]
    assert env(sidecar)["OAUTH2_PROXY_CLIENT_SECRET"] == {
        "secretKeyRef": {"name": "site-oidc", "key": "lllm2-client-secret"}
    }
    assert env(sidecar)["OAUTH2_PROXY_COOKIE_SECRET"] == {
        "secretKeyRef": {"name": "site-oidc", "key": "lllm2-cookie-secret"}
    }
    # The top-level env stays with the lllm2 container.
    assert "HTTPS_PROXY" not in env(containers(objects)["lllm2"])
    assert sidecar["volumeMounts"] == [
        {"name": "oidc-ca", "mountPath": "/etc/oauth2-proxy/ca", "readOnly": True}
    ]
    volumes = {v["name"]: v for v in pod(objects)["volumes"]}
    assert volumes["oidc-ca"] == {"name": "oidc-ca", "configMap": {"name": "site-ca"}}
    assert sidecar["resources"]["limits"] == {"cpu": "1", "memory": "256Mi"}
    policy = objects["NetworkPolicy", "lllm2"]["spec"]
    assert policy["ingress"][1]["ports"] == [{"port": 4180}]


@needs_helm
def test_sidecar_resources_may_be_numbers():
    objects = render(
        *sets(OIDC), "--set-json", "ingress.oidc.resources.requests.cpu=0.1"
    )
    sidecar = containers(objects)["oauth2-proxy"]
    assert sidecar["resources"]["requests"]["cpu"] == 0.1


@needs_helm
@pytest.mark.parametrize(
    ("options", "message"),
    [
        (sets({"ingress.enabled": "true"}), "ingress.auth is not set"),
        (
            sets(without(EXTERNAL, "panel.host")),
            "ingress.auth external needs panel.host: 0.0.0.0",
        ),
        (
            sets({**without(EXTERNAL, "panel.host"), "ingress.auth": "none"}),
            "ingress.auth none needs panel.host: 0.0.0.0",
        ),
        (
            sets(without(EXTERNAL, "ingress.hosts[0].host")),
            "ingress.hosts needs at least one host",
        ),
        (
            [
                *sets(without(EXTERNAL, "ingress.hosts[0].host")),
                "--set-json",
                "ingress.hosts=[{}]",
            ],
            "ingress.hosts needs at least one host",
        ),
        (
            sets({**EXTERNAL, "ingress.hosts[0].host": "10.0.0.5"}),
            "ingress.hosts has the IP address 10.0.0.5",
        ),
        (
            sets(without(OIDC, "ingress.oidc.issuerURL")),
            "set ingress.oidc.issuerURL",
        ),
        (sets(without(OIDC, "ingress.oidc.clientID")), "set ingress.oidc.clientID"),
        (
            sets(without(OIDC, "ingress.oidc.existingSecret")),
            "set ingress.oidc.existingSecret to a Secret in namespace default "
            "with the keys client-secret and cookie-secret",
        ),
        (
            sets(without(OIDC, "ingress.oidc.allowedGroups[0]")),
            "ingress.oidc needs allowedGroups, allowedRoles or emailDomains",
        ),
        (
            sets({**OIDC, "ingress.oidc.allowedRoles[0]": "lllm2-user"}),
            "allowedRoles needs ingress.oidc.provider: keycloak-oidc",
        ),
        (
            sets({**OIDC, "panel.host": "0.0.0.0"}),
            "ingress.auth oidc keeps the panel on 127.0.0.1",
        ),
        (
            sets(without(OIDC, "ingress.hosts[0].host")),
            "ingress.hosts needs at least one host",
        ),
        (
            sets({**EXTERNAL, "networkPolicy.enabled": "true"}),
            "networkPolicy.from is empty, so nothing could reach the panel on port 8082",
        ),
        (
            sets({**OIDC, "networkPolicy.enabled": "true"}),
            "networkPolicy.from is empty, so nothing could reach the panel on port 4180",
        ),
    ],
)
def test_the_chart_refuses_a_panel_left_open_or_unreachable(options, message):
    assert message in refusal(*options)


@needs_helm
def test_the_network_policy_admits_only_the_listed_peers_to_the_panel():
    peer = {"namespaceSelector": {"matchLabels": {"name": "ingress-nginx"}}}
    objects = render(
        *sets({**EXTERNAL, "networkPolicy.enabled": "true"}),
        "--set-json",
        f"networkPolicy.from=[{json.dumps(peer)}]",
    )
    policy = objects["NetworkPolicy", "lllm2"]["spec"]
    assert policy["podSelector"] == {
        "matchLabels": {
            "app.kubernetes.io/name": "lllm2",
            "app.kubernetes.io/instance": "lllm2",
        }
    }
    assert policy["policyTypes"] == ["Ingress"]
    assert policy["ingress"] == [
        # The model API stays open to the cluster, as without the policy.
        {"ports": [{"port": 1920}]},
        {"from": [peer], "ports": [{"port": 8082}]},
    ]


@needs_helm
def test_a_network_policy_without_a_panel_port_keeps_only_the_model_api():
    objects = render("--set", "networkPolicy.enabled=true")
    policy = objects["NetworkPolicy", "lllm2"]["spec"]
    assert policy["ingress"] == [{"ports": [{"port": 1920}]}]


@needs_helm
def test_notes_by_default_point_to_port_forward(tmp_path):
    notes = render_notes(tmp_path)
    assert "listens only inside the pod" in notes
    assert "kubectl port-forward -n default deploy/lllm2 8082:8082" in notes
    assert "Ingress" not in notes
    assert "networkPolicy" not in notes


@needs_helm
def test_notes_for_external_auth_ask_for_a_sign_in_check(tmp_path):
    options = sets(
        {
            **EXTERNAL,
            "ingress.tls[0].secretName": "lllm2-tls",
            "ingress.tls[0].hosts[0]": "lllm2.example.com",
        }
    )
    notes = render_notes(tmp_path, *options)
    assert "  https://lllm2.example.com\n" in notes
    assert "open the address in a private window" in notes
    assert "Any pod in the cluster can reach the panel" in notes
    assert "No TLS" not in notes
    assert "WARNING" not in notes
    notes = render_notes(
        tmp_path / "policy",
        *options,
        *sets(
            {
                "networkPolicy.enabled": "true",
                "networkPolicy.from[0].ipBlock.cidr": "10.0.0.0/24",
            }
        ),
    )
    assert "A NetworkPolicy lets only networkPolicy.from reach the panel" in notes
    assert "Any pod in the cluster" not in notes


@needs_helm
def test_notes_warn_about_an_ingress_without_auth(tmp_path):
    notes = render_notes(tmp_path, *sets({**EXTERNAL, "ingress.auth": "none"}))
    assert "WARNING: ingress.auth is none." in notes
    assert "  http://lllm2.example.com\n" in notes
    assert "No TLS on lllm2.example.com" in notes
    assert "passwords and session cookies cross the" in notes


@needs_helm
def test_notes_for_oidc_give_the_redirect_uris_and_the_sidecar_log(tmp_path):
    notes = render_notes(tmp_path, *sets(OIDC))
    assert "signs users in with https://keycloak.example.com/realms/lab" in notes
    assert "  https://lllm2.example.com/oauth2/callback" in notes
    assert "kubectl logs -n default deploy/lllm2 -c oauth2-proxy" in notes
    # Without TLS at the Ingress the Secure session cookie never comes back.
    assert "sign-in fails" in notes
    assert "WARNING" not in notes
    assert "Any pod in the cluster" not in notes


@needs_helm
def test_notes_say_when_ingress_settings_have_no_effect(tmp_path):
    notes = render_notes(tmp_path, *sets({**OIDC, "ingress.enabled": "false"}))
    assert "ingress.enabled is false" in notes
    assert "ingress.enabled is false" not in render_notes(tmp_path / "default")


# CI installs oauth2-proxy at the chart's pinned release, so this never skips there.
needs_oauth2_proxy = pytest.mark.skipif(
    shutil.which("oauth2-proxy") is None and os.environ.get("GITHUB_ACTIONS") != "true",
    reason="oauth2-proxy is not installed",
)


@needs_helm
@needs_oauth2_proxy
@pytest.mark.parametrize(
    "options",
    [sets(OIDC), ["-f", str(CHART / "ci" / "oidc-values.yaml")]],
    ids=["minimal", "ci-values"],
)
def test_the_pinned_oauth2_proxy_accepts_the_sidecar_arguments(options):
    binary = shutil.which("oauth2-proxy")
    assert binary, "oauth2-proxy is not installed"
    sidecar = containers(render(*options))["oauth2-proxy"]
    tag = sidecar["image"].rpartition(":")[2]
    version = subprocess.run(
        [binary, "--version"], capture_output=True, text=True, check=True
    )
    assert version.stdout.startswith(f"oauth2-proxy {tag} ")
    # The rendered arguments, with the CA file pointed at one that exists here.
    args = [
        "--provider-ca-file=/etc/ssl/certs/ca-certificates.crt"
        if arg.startswith("--provider-ca-file=")
        else arg
        for arg in sidecar["args"]
    ]
    cookie_secret = base64.urlsafe_b64encode(os.urandom(32)).decode()
    result = subprocess.run(
        [binary, "--config-test", *args],
        capture_output=True,
        text=True,
        env={
            "OAUTH2_PROXY_CLIENT_SECRET": "client-secret",
            "OAUTH2_PROXY_COOKIE_SECRET": cookie_secret,
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "configuration is valid" in result.stdout + result.stderr
