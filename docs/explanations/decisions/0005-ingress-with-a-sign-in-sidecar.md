# Reach the panel through an optional Ingress with a sign-in sidecar

## Status

Accepted, 2 October 2026.

## Context

The Helm chart keeps the panel on the pod's loopback address, reached with
`kubectl port-forward` ([ADR 4](0004-kubernetes-image-and-chart.md)). Users
also want a browser address through an Ingress, and plan to sign users in with
Keycloak. The panel has no login: anyone who reaches it can download models,
use the GPU and run any program in the pod.

The panel also rejected every request an Ingress passes on. Its Host check,
which stops DNS rebinding, accepted only its own address, and its CSRF check
compared a POST's `Origin` with `http://<Host>`, which a TLS Ingress never
matches.

## Decision

**The panel accepts the names a proxy passes on.** `LLLM2_PANEL_ALLOWED_HOSTS`
lists extra host names, accepted on any port. The Origin check also accepts
`https://<Host>`: a cross-site page differs in host, and the session token
stays the CSRF defence.

**An optional Ingress that refuses to render without an auth choice.** The
chart has one `networking.k8s.io/v1` Ingress for the panel, off by default.
It serves only `/`, as the panel's pages use absolute paths, and passes its
hosts to the panel. `ingress.auth` has no usable default:

- `oidc`, the recommended mode, runs oauth2-proxy as a sidecar in the pod. It
  signs users in with any OpenID Connect provider, Keycloak being the
  documented one, and admits only the listed groups, roles or email domains.
  The panel stays on 127.0.0.1, and the Ingress reaches the sidecar on Service
  port 4180, so no other pod can skip the sign-in. Every site value is a chart
  value, and the client and cookie secrets live in a Secret the user creates,
  read through `secretKeyRef`.
- `external` means the controller, or a proxy in front of it, authenticates.
  The panel joins the Service on port 8082. The chart neither writes nor checks
  the controller's annotations, as each controller has its own.
- `none` is the deliberate opt-out, and NOTES warn about it.

Guards refuse what would leave the panel open or broken: no issuer, client,
Secret or sign-in restriction in `oidc` mode; roles with the plain `oidc`
provider, which ignores them; `panel.host: 0.0.0.0` with `oidc`, which would
let pods bypass the sidecar; `127.0.0.1` with `external` or `none`, which the
Ingress cannot reach; and missing hosts.

**Sidecar settings, each for a measured reason.** oauth2-proxy v7.15.5 is
pinned, as it fixes two critical authentication bypasses.

- `--upstream-timeout=300s`, as the default 30 s cuts off panel actions that
  take up to 180 s.
- `--cookie-refresh=1m`. Without it, a session outlives a revoked Keycloak
  session for the whole 168 h cookie lifetime.
- `--api-route=^/api/`, so the panel's polls get 401 when a session ends,
  rather than a redirect to Keycloak that `fetch()` cannot follow.
- `--cookie-csrf-per-request`, so two tabs can sign in at once.
- No `--reverse-proxy`. Without it, the callback is built from the `Host` the
  controller passes on, once for each host. With it, oauth2-proxy would trust
  `X-Forwarded-Host` from any pod.
- In `oidc` mode the Service publishes the pod's address before it is ready.
  A sidecar that cannot reach Keycloak would otherwise take the model API off
  the Service too.

**An optional NetworkPolicy.** With `external` or `none`, every pod can reach
port 8082 without passing the Ingress. `networkPolicy` limits the panel's port
(8082, or 4180 with `oidc`) to listed peers, such as the ingress controller.
`engineFrom` does the same for the model API, which stays open otherwise. A
user's own policy could not narrow a port the chart's policy opens.

**The model API is not exposed.** lllm2 sends no key to its own engine, so
`LLAMA_API_KEY` would stop models starting. Coding clients send a static key,
which oauth2-proxy's bearer-token mode refuses. A later change can give the
engine a key that lllm2 generates and sends, then add an optional engine
Ingress.

Alternatives not taken:

- Auth presets that write controller annotations tie the chart to one
  controller. ingress-nginx, whose annotations most guides use, is archived
  upstream.
- Detecting "well-known" auth annotations is wrong both ways.
- A password checked by the panel itself would work behind any controller and
  close the in-cluster bypass of `external`. It changes the panel's model, as
  it has no login, so it is left as a question rather than assumed. `oidc`
  already gives the safe path.

## Consequences

- The chart has a second image. Bumping oauth2-proxy means changing
  `ingress.oidc.image.tag` and the version and checksum in CI together. The
  chart test runs the pinned binary's `--config-test` on the rendered
  arguments, so a renamed flag fails CI.
- While Keycloak is down, nobody can sign in, and signed-in users lose the
  panel within one token lifetime.
- The sidecar logs each request's path. That includes the one-time code on
  `/oauth2/callback`, which is useless once the sidecar has redeemed it.
- An `env` entry named `LLLM2_PANEL_ALLOWED_HOSTS` replaces the chart's list.
- The design was tested against a mock OpenID provider with the real panel
  behind the pinned oauth2-proxy, not against Keycloak or a cluster. The docs
  carry a first-deployment checklist.
