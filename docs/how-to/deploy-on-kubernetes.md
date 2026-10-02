# Deploy on Kubernetes

The `lllm2` Helm chart runs the panel and its model server in one pod on a GPU
node. The pod gets one whole GPU, and keeps models and state on persistent
volumes. The model API is a ClusterIP Service on port 1920. Nothing is exposed
outside the cluster unless you turn on the optional Ingress for the panel.

## Before you start

You need:

- A GPU node whose NVIDIA driver reports CUDA 12.9 or newer in `nvidia-smi`.
  The node needs no CUDA toolkit: the image bundles the CUDA runtime.
- The NVIDIA Container Toolkit and device plugin on that node, or the
  [GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html),
  so that pods can request `nvidia.com/gpu`.
- A default StorageClass, or claims you have made yourself.
- Helm 3.8 or newer, and kubectl.

The image `ghcr.io/gilesknap/lllm2` carries lllm2 and the CUDA 12.9.1 build of
its pinned llama.cpp engine. Each release's chart runs the image of the same
release, so the chart version picks the lllm2 version and the engine together.

## Install

Write a values file for your node. This one picks the NVIDIA runtime class, an
L40S node and its taint, and a larger models volume:

```yaml
runtimeClassName: nvidia
nodeSelector:
  nvidia.com/gpu.product: NVIDIA-L40S
tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
resources:
  # With no requests the pod is the first one evicted when the node runs short
  # of memory, which would end an experiment part way.
  requests:
    cpu: "4"
    memory: 16Gi
persistence:
  models:
    size: 500Gi
```

Install a release of the chart, for example `0.11.0`:

```bash
helm install lllm2 oci://ghcr.io/gilesknap/charts/lllm2 --version 0.11.0 \
  -n lllm2 --create-namespace -f values.yaml
kubectl -n lllm2 rollout status deploy/lllm2
```

Check that the pod sees its GPU and the engine:

```bash
kubectl -n lllm2 exec deploy/lllm2 -- nvidia-smi
kubectl -n lllm2 exec deploy/lllm2 -- lllm2 engines list
```

`engines list` shows the engine under `/opt/lllm2/engines` and says it matches
the running release. Run `helm show values oci://ghcr.io/gilesknap/charts/lllm2`
to see every value. Each release on GitHub also has `lllm2.schema.json`
attached, which editors can use to check a values file.

## Reach the panel

Forward the panel's port from your workstation:

```bash
kubectl -n lllm2 port-forward deploy/lllm2 8082:8082
```

Then open `http://127.0.0.1:8082`. Keep the local port at 8082, as the panel
accepts only its own port in the browser's address.

The panel has no login or TLS. Anyone who reaches it can download models, start
the GPU, and set the engine to any program in the pod and run it. So the chart
makes it listen only on the pod's loopback address. Only people with
`pods/portforward` rights in the namespace can reach it.

To reach the panel from a browser without port-forward, use the chart's
Ingress, which signs users in first (see
[Reach the panel through an Ingress](#reach-the-panel-through-an-ingress)).

To use a proxy of your own in the cluster instead, set `panel.host: 0.0.0.0`.
The chart then adds the panel to the Service on port 8082. The proxy must
authenticate every request and pass the browser's `Host` header on unchanged.
The panel refuses a `Host` other than its own address, so add the proxy's
public name to `LLLM2_PANEL_ALLOWED_HOSTS` through `env`. Anything else that
reaches port 8082 controls the workbench, so allow only the proxy with
`networkPolicy` (see
[Limit who reaches the panel in the cluster](#limit-who-reaches-the-panel-in-the-cluster)).

## Reach the panel through an Ingress

The panel has no login. Anyone who reaches it can download models, use the
GPU, and run any program in the pod. So the chart's Ingress is off by
default, and it refuses to render until `ingress.auth` says who signs users
in:

| `ingress.auth` | Who signs users in | `panel.host` |
| --- | --- | --- |
| `oidc` | An oauth2-proxy sidecar in the pod, with your OpenID Connect provider, such as Keycloak. This is the recommended mode. | `127.0.0.1`, the default |
| `external` | Your ingress controller, or a proxy in front of it | `0.0.0.0` |
| `none` | Nobody | `0.0.0.0` |

The panel serves only at `/`, as its pages and API use absolute paths. So
each entry in `ingress.hosts` is a DNS name of its own, with no path. An
Ingress rule cannot use an IP address: to reach the panel by address, use
port-forward. The chart passes the hosts to the panel in
`LLLM2_PANEL_ALLOWED_HOSTS`. If you also set that variable in `env`, yours
replaces the chart's list, so include the Ingress hosts in it.

### Sign in with Keycloak

With `ingress.auth: oidc`, the chart runs
[oauth2-proxy](https://oauth2-proxy.github.io/oauth2-proxy/) as a second
container in the pod. The Ingress sends every request to it, on Service port
4180. It signs the user in with Keycloak and checks their group, role or email
domain. Only then does it forward the request to the panel, which listens on
the pod's loopback address. The panel is not on the Service, so other pods
cannot skip the sign-in either. This works with any ingress controller, as the
controller needs no authentication settings of its own.

Any OpenID Connect provider works the same way, with `provider: oidc` and its
issuer. These steps set up Keycloak 19 or newer, using its admin console. The
names in angle brackets are yours.

1. **Find the issuer.** Open
   `https://<keycloak>/realms/<realm>/.well-known/openid-configuration` and
   copy its `issuer` value exactly into `ingress.oidc.issuerURL`, with no
   trailing slash. Keycloak before version 17 has `/auth` before `/realms`.
   The sidecar accepts only tokens from that exact issuer, and calls the
   token and key addresses the same document lists. So the pod must reach
   Keycloak at its public name. Check that from inside the cluster:

   ```bash
   kubectl -n lllm2 run oidc-check --rm -it --restart=Never \
     --image=curlimages/curl -- \
     -sS https://<keycloak>/realms/<realm>/.well-known/openid-configuration
   ```

   If Keycloak's certificate comes from a private CA, put the CA in a
   ConfigMap under the key `ca.crt` and name it in `ingress.oidc.caConfigMap`.
   If the pod needs an HTTP proxy to reach Keycloak, set `HTTPS_PROXY` and
   `NO_PROXY` in `ingress.oidc.env`.
2. **Create the client.** Go to **Clients**, then **Create client**.
   - Client type **OpenID Connect**, and a client ID such as `lllm2`, which
     goes into `ingress.oidc.clientID`.
   - Turn **Client authentication** on.
   - Keep **Standard flow** only, and clear **Direct access grants**.
   - Under **Valid redirect URIs**, add `https://<host>/oauth2/callback` for
     each host in `ingress.hosts`. `helm install` prints them.
   - Leave **Web origins** empty.
3. **Copy the client secret** from the client's **Credentials** tab.
4. **Add an audience mapper.** oauth2-proxy's Keycloak guide asks for it. Go
   to **Clients**, `lllm2`, **Client scopes**, `lllm2-dedicated`, then
   **Configure a new mapper**, and choose **Audience**. Set **Included Client
   Audience** to `lllm2` and turn **Add to access token** on.
5. **Choose who may sign in.** The chart refuses to render without at least
   one of these:
   - **Groups.** Under **Client scopes**, create a scope named `groups`, of
     type OpenID Connect. Give it a **Group Membership** mapper with token
     claim name `groups` and **Full group path** on, added to the ID token,
     access token and userinfo. Then add the scope to the `lllm2` client as a
     default scope. Name groups by their full path, such as `/lllm2-users`,
     in `ingress.oidc.allowedGroups`. The sidecar asks for the `groups`
     scope, and Keycloak may refuse a scope the client does not have
     (verify this on your Keycloak).
   - **Roles.** Set `provider: keycloak-oidc` and list realm roles by name,
     and client roles as `lllm2:<role>`, in `ingress.oidc.allowedRoles`.
   - **Email domains.** List them in `ingress.oidc.emailDomains`. `["*"]`
     admits every user of the realm, so use it only for a realm that holds
     just the people you trust.
6. **Check the users.** Each one needs an email address with **Email
   verified** on, or signing in fails.
7. **Create the Secret** in the release namespace:

   ```bash
   kubectl -n lllm2 create secret generic lllm2-oidc \
     --from-literal=client-secret='<client secret from step 3>' \
     --from-literal=cookie-secret="$(openssl rand -base64 32 | tr -- '+/' '-_')"
   ```

   Keep the `tr`. oauth2-proxy reads the cookie secret as URL-safe base64,
   and refuses a value with `+` or `/` in it. Without the `tr`, the command
   works only some of the time. The chart never creates this Secret and
   never sees its values.
8. **Write the values:**

   ```yaml
   ingress:
     enabled: true
     className: nginx                                # your controller's class
     auth: oidc
     annotations:
       cert-manager.io/cluster-issuer: letsencrypt   # your ClusterIssuer
       # ingress-nginx only: the panel waits up to three minutes for some
       # actions, and Keycloak sessions make large cookies.
       nginx.ingress.kubernetes.io/proxy-read-timeout: "200"
       nginx.ingress.kubernetes.io/proxy-buffer-size: 16k
     hosts:
       - host: lllm2.example.com
     tls:
       - secretName: lllm2-tls
         hosts: [lllm2.example.com]
     oidc:
       issuerURL: https://keycloak.example.com/realms/lab
       clientID: lllm2
       existingSecret: lllm2-oidc
       allowedGroups: [/lllm2-users]
   ```

   `panel.host` stays at its default, `127.0.0.1`. Other controllers need
   neither annotation, as long as their read timeout is above 180 seconds.
   Annotation values must be strings, so quote numbers.
9. **Install** the release with these values, as in [Install](#install),
   **then check:**
   - `kubectl -n lllm2 get pod` shows `2/2` ready.
   - A private browser window at `https://lllm2.example.com` goes to
     Keycloak, then to the panel.
   - A user outside the allowed groups gets a 403 page.
   - **Find models** completes, so long actions pass through.
   - The panel still works after ten minutes, so the session refreshes.
   - From another pod, `curl -sI http://lllm2.lllm2:4180/` answers 302 with
     a `Location` at Keycloak.

The sidecar checks each session with Keycloak every minute. A user disabled
or signed out in Keycloak loses the panel within a minute, and otherwise a
session lasts as long as the Keycloak session (SSO Session Max, 10 hours by
default; verify this on yours). When a session ends, the panel shows its
connection as unavailable and its actions fail. Reload the page to sign in
again. While Keycloak is down, nobody can sign in, and signed-in users lose
the panel within one token lifetime, 5 minutes by default, as their sessions
cannot refresh.

While the sidecar is not ready, `kubectl get pod` shows `1/2`. The Service
still sends the model API to the pod, so clients in the cluster keep working.

| Symptom | Cause and fix |
| --- | --- |
| The sidecar shows `CreateContainerConfigError` | The Secret is missing, or lacks a key. Create it as in step 7, or set `clientSecretKey` and `cookieSecretKey`. |
| The sidecar restarts, and its log says "failed to discover OIDC configuration" | The pod cannot reach the issuer. Check `issuerURL` and the network, as in step 1. |
| The sidecar log says "issuer did not match" | `issuerURL` differs from the `issuer` in the discovery document, for example by a trailing slash, the scheme, or an internal host name. |
| The sidecar log says "x509: certificate signed by unknown authority" | Keycloak uses a private CA. Set `caConfigMap`. |
| Keycloak says the redirect URI is invalid | Add `https://<host>/oauth2/callback` for every host to the client. |
| Keycloak reports `invalid_scope` | `allowedGroups` is set, but the client has no `groups` scope (step 5). |
| Error 500 after sign-in, and the sidecar log says "isn't verified" | The user's email address is not verified (step 6). |
| Error 403 after sign-in | The user is not in an allowed group or role, or a group lacks its leading `/`. Check the token under **Client scopes**, **Evaluate**. |
| Signing in returns to Keycloak again and again | The browser is not using https, so it drops the session cookie. Add TLS at the Ingress or in front of it. |
| ingress-nginx error 502, "upstream sent too big header" | Set `nginx.ingress.kubernetes.io/proxy-buffer-size: 16k`. |
| Error 403, "Use this workstation's panel address" | The controller rewrote the `Host` header, or an `env` entry replaced `LLLM2_PANEL_ALLOWED_HOSTS`. Pass `Host` on unchanged, and list every host. |

`ingress.oidc.extraArgs` passes more oauth2-proxy arguments, after the
chart's, so a repeated one overrides the chart's value. Never put secrets
there, as arguments show in the pod spec. Some sites can reach Keycloak only
at an internal name. Then keep `issuerURL` as Keycloak advertises it, and add
`--skip-oidc-discovery=true` with `--login-url` (the public authorization
address), `--redeem-url` and `--oidc-jwks-url` (internal addresses). This was
not tested with Keycloak.

The sidecar runs `quay.io/oauth2-proxy/oauth2-proxy:v7.15.5`. For an
air-gapped cluster, mirror it with all its platforms and set
`ingress.oidc.image.repository`. The pod's `imagePullSecrets` apply to both
containers.

### Authenticate at the ingress controller

With `ingress.auth: external`, the controller or a proxy in front of it signs
users in. The chart cannot check that it does. The Ingress reaches the panel
on the Service, so set `panel.host: 0.0.0.0`. This example uses basic auth in
ingress-nginx:

```bash
htpasswd -c auth alice        # from apache2-utils or httpd-tools
kubectl -n lllm2 create secret generic lllm2-basic-auth --from-file=auth
```

```yaml
panel:
  host: 0.0.0.0
ingress:
  enabled: true
  className: nginx
  auth: external
  annotations:
    nginx.ingress.kubernetes.io/auth-type: basic
    nginx.ingress.kubernetes.io/auth-secret: lllm2-basic-auth
    nginx.ingress.kubernetes.io/auth-realm: lllm2
    cert-manager.io/cluster-issuer: letsencrypt
    nginx.ingress.kubernetes.io/proxy-read-timeout: "200"
  hosts:
    - host: lllm2.example.com
  tls:
    - secretName: lllm2-tls
      hosts: [lllm2.example.com]
networkPolicy:
  enabled: true
  from:
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: ingress-nginx
```

After installing, open the address in a private window and check that it asks
for a password. The ingress-nginx project is archived upstream and no longer
fixed, so check your controller's own authentication. With Traefik, for
example, reference a `basicAuth` or `forwardAuth` Middleware in the
`traefik.ingress.kubernetes.io/router.middlewares` annotation. In every case,
set `ingress.auth: external`.

### Limit who reaches the panel in the cluster

With `external` or `none`, any pod in the cluster can reach the panel on
Service port 8082 without passing the Ingress. `networkPolicy` limits that
port to the peers you list, usually the ingress controller's pods. With
`oidc`, it limits the sidecar's port 4180 instead. The model API on port 1920
stays open to the cluster unless you list its clients in
`networkPolicy.engineFrom`. Port-forward is unaffected.

- A controller in its own namespace matches a `namespaceSelector`, as in the
  example above.
- A controller on the host network sends from the node's address, which no
  selector matches. List the nodes' addresses instead:

  ```yaml
  networkPolicy:
    enabled: true
    from:
      - ipBlock:
          cidr: 10.0.0.0/24     # your nodes' addresses
  ```

- A NetworkPolicy has no effect unless the cluster's network plugin enforces
  it. Calico and Cilium do, and flannel on its own does not.
- Policies add up: a port that any policy opens stays open. So limit the
  model API with `networkPolicy.engineFrom`, not with a policy of your own
  beside the chart's.

### No authentication

`ingress.auth: none` lets anyone who reaches the Ingress control the
workbench. Use it only on a network you trust completely, and set
`panel.host: 0.0.0.0`.

### What the Ingress does not cover

- The model API on port 1920 is not behind the Ingress or the sign-in
  sidecar. Reach it from inside the cluster, or with port-forward. Do not set
  `LLAMA_API_KEY` in `env`: lllm2 does not send it to its own engine, so
  models fail to start.
- Do not rewrite the `Host` header at the controller, or pass
  `--pass-host-header=false` to the sidecar. The panel would refuse the
  requests.
- The chart makes an Ingress, not a Gateway API route. To use an
  `HTTPRoute`, set `panel.host: 0.0.0.0`, point the route at Service port
  `panel` behind your own authentication, and add its host name to
  `LLLM2_PANEL_ALLOWED_HOSTS` through `env`.

## Point clients at the engine

Start a model in the panel first. Clients in the cluster use:

```text
http://lllm2.lllm2.svc.cluster.local:1920/v1
```

`helm install` prints this address for your release name and namespace. From
a workstation, forward the port:

```bash
kubectl -n lllm2 port-forward svc/lllm2 1920:1920
```

Clients then use the usual `http://127.0.0.1:1920/v1`, and `lllm2 claude` and
`lllm2 codex` work as they do with a local model (see
[Connect a client](connect-a-client.md)).

The model server has no API key, so any pod that reaches the Service can use
the GPU. In a shared cluster, allow only your clients: set
`networkPolicy.enabled` and list them in `networkPolicy.engineFrom` (see
[Limit who reaches the panel in the cluster](#limit-who-reaches-the-panel-in-the-cluster)).

## Models and storage

The chart creates two ReadWriteOnce claims:

| Claim | Mounted at | Holds | Default size |
| --- | --- | --- | --- |
| `lllm2-models` | `/models` | Downloaded GGUF files | 200Gi |
| `lllm2-data` | `/data` | `/data/state` (settings and results) and `/data/engines` (engines you add) | 10Gi |

Set `persistence.<name>.size` and `storageClassName` before the first install.
To use a volume you have prepared, such as an NFS share, set
`persistence.<name>.existingClaim`. The pod runs as user and group 1000, so the
volume must be writable by group 1000, or set your own `podSecurityContext`.

`helm uninstall` keeps both claims, as models are slow to download again and
the data claim holds your results. A new install with the same release name
and namespace uses them again. Delete them with `kubectl -n lllm2 delete pvc`
when you no longer need them.

**Find models** downloads from Hugging Face, so the pod needs HTTPS egress. Set
a proxy through `env` if your cluster needs one:

```yaml
env:
  - name: HTTPS_PROXY
    value: http://proxy.example.com:3128
```

Without egress, copy GGUF files into `/models` with `kubectl cp`. For an
air-gapped cluster, mirror the image and set `image.repository`.

If you set a memory limit, keep it above the model file size. llama.cpp maps
the model file into memory, and the page cache counts against the limit.

## Upgrade

Wait for running experiments to finish, then:

```bash
helm upgrade lllm2 oci://ghcr.io/gilesknap/charts/lllm2 --version 0.12.0 \
  -n lllm2 -f values.yaml
```

The old pod stops before the new one starts, as both need the GPU and the
panel's state lock. The engine moves to the new release's pin.

To run the newest build of `main` instead of a release, install the chart from
a checkout of `main`:

```bash
helm install lllm2 ./Charts/lllm2 -n lllm2 --create-namespace -f values.yaml
```

It runs the `main` image, which the pod pulls each time it starts, so
`kubectl -n lllm2 rollout restart deploy/lllm2` picks up a newer build.

## Limits

- One pod uses one whole GPU. Time-sliced or MPS-shared GPUs are not
  supported: lllm2 checks that no other process uses its GPU.
- The image carries only the CUDA 12.9.1 engine, which runs on every supported
  driver and GPU. On a driver that reports CUDA 13.3 or newer, you can add the
  CUDA 13.3.1 build with
  `kubectl -n lllm2 exec deploy/lllm2 -- lllm2 engines install cuda`. It goes
  to `/data/engines`, where lllm2 finds it first and uses it. It stays there
  through upgrades, so delete it before `helm upgrade`, or lllm2 may keep
  using the old engine.
- The Modal backend is not installed in the image.
- `lllm2 service` and the panel's upgrade notice do not apply. Upgrade with
  `helm upgrade`.
