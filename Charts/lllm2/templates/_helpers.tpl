{{- define "lllm2.fullname" -}}
{{- if contains .Chart.Name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "lllm2.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "lllm2.labels" -}}
{{ include "lllm2.selectorLabels" . }}
app.kubernetes.io/version: {{ include "lllm2.tag" . | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{- end -}}

{{/* The image tag: the release's own unless the values set one. */}}
{{- define "lllm2.tag" -}}
{{- .Values.image.tag | default .Chart.AppVersion -}}
{{- end -}}

{{/* main and latest move, so the kubelet pulls them on every start. */}}
{{- define "lllm2.pullPolicy" -}}
{{- $moving := has (include "lllm2.tag" .) (list "main" "latest") -}}
{{- .Values.image.pullPolicy | default (ternary "Always" "IfNotPresent" $moving) -}}
{{- end -}}

{{/* The panel joins the Service only when it listens on the pod address. */}}
{{- define "lllm2.panelExposed" -}}
{{- if eq .Values.panel.host "0.0.0.0" }}true{{ end -}}
{{- end -}}

{{- define "lllm2.claimName" -}}
{{- $claim := index .root.Values.persistence .name -}}
{{- $claim.existingClaim | default (printf "%s-%s" (include "lllm2.fullname" .root) .name) -}}
{{- end -}}

{{/* The Ingress host names, comma-separated, for the panel's Host check. */}}
{{- define "lllm2.ingressHosts" -}}
{{- $names := list -}}
{{- range .Values.ingress.hosts }}{{ $names = append $names .host }}{{ end -}}
{{- join "," $names -}}
{{- end -}}

{{/* An oauth2-proxy sidecar signs users in before the panel. */}}
{{- define "lllm2.oidc" -}}
{{- if and .Values.ingress.enabled (eq .Values.ingress.auth "oidc") }}true{{ end -}}
{{- end -}}

{{/* The Service port that reaches the panel, if any: the sidecar or the panel. */}}
{{- define "lllm2.panelPort" -}}
{{- if include "lllm2.oidc" . }}4180{{ else if include "lllm2.panelExposed" . }}8082{{ end -}}
{{- end -}}

{{/* Refuse an Ingress that would leave the panel open or unreachable. */}}
{{- define "lllm2.checkIngress" -}}
{{- $docs := "https://gilesknap.github.io/lllm2/how-to/deploy-on-kubernetes.html#reach-the-panel-through-an-ingress" -}}
{{- $ingress := .Values.ingress -}}
{{- if not $ingress.auth -}}
{{- fail (printf "ingress.auth is not set. The panel has no login, and anyone who reaches it can run any program in the pod. Set ingress.auth to oidc to sign users in with an OpenID Connect provider such as Keycloak, to external if the ingress controller, or a proxy in front of it, authenticates every request, or to none to let anyone who reaches the Ingress control the workbench. See %s" $docs) -}}
{{- end -}}
{{- if eq $ingress.auth "oidc" -}}
{{- $oidc := $ingress.oidc -}}
{{- if not $oidc.issuerURL -}}
{{- fail (printf "ingress.auth is oidc, so set ingress.oidc.issuerURL to the provider's issuer, such as https://keycloak.example.com/realms/<realm> for Keycloak. See %s" $docs) -}}
{{- end -}}
{{- if not $oidc.clientID -}}
{{- fail (printf "ingress.auth is oidc, so set ingress.oidc.clientID to the client registered for lllm2 at %s. See %s" $oidc.issuerURL $docs) -}}
{{- end -}}
{{- if not $oidc.existingSecret -}}
{{- fail (printf "ingress.auth is oidc, so set ingress.oidc.existingSecret to a Secret in namespace %s with the keys %s and %s. See %s for the kubectl command." .Release.Namespace $oidc.clientSecretKey $oidc.cookieSecretKey $docs) -}}
{{- end -}}
{{- if not (or $oidc.allowedGroups $oidc.allowedRoles $oidc.emailDomains) -}}
{{- fail "ingress.oidc needs allowedGroups, allowedRoles or emailDomains. Otherwise every user the issuer knows can sign in and run any program in the pod. To admit them all, set ingress.oidc.emailDomains to [\"*\"]." -}}
{{- end -}}
{{- if and $oidc.allowedRoles (ne $oidc.provider "keycloak-oidc") -}}
{{- fail "ingress.oidc.allowedRoles needs ingress.oidc.provider: keycloak-oidc. The oidc provider ignores roles, so it would admit users without them." -}}
{{- end -}}
{{- if include "lllm2.panelExposed" . -}}
{{- fail "ingress.auth oidc keeps the panel on 127.0.0.1, behind the sign-in sidecar. panel.host: 0.0.0.0 would let any pod in the cluster reach the panel without signing in, so set panel.host back to 127.0.0.1." -}}
{{- end -}}
{{- else if not (include "lllm2.panelExposed" .) -}}
{{- fail (printf "ingress.auth %s needs panel.host: 0.0.0.0. The Ingress reaches the panel through the Service, so the panel must listen on the pod address, where any pod in the cluster can also reach it without signing in. Limit that with networkPolicy, or use ingress.auth oidc, which keeps the panel inside the pod. See %s" $ingress.auth $docs) -}}
{{- end -}}
{{- $named := true -}}
{{- range $ingress.hosts }}{{ if not .host }}{{ $named = false }}{{ end }}{{ end -}}
{{- if not (and $ingress.hosts $named) -}}
{{- fail "ingress.hosts needs at least one host, such as lllm2.example.com. The panel answers only to the host names the chart passes it." -}}
{{- end -}}
{{- range $ingress.hosts -}}
{{- if regexMatch "^[0-9.]+$" .host -}}
{{- fail (printf "ingress.hosts has the IP address %s, but an Ingress rule needs a DNS name. To reach the panel by IP address, use kubectl port-forward. See %s" .host $docs) -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
oauth2-proxy's arguments. Secrets come from the environment, never from here,
because arguments show in the pod spec.
- --upstream-timeout is above the 180 s the panel waits for its slowest
  actions. oauth2-proxy's default of 30 s cuts them off with a 502.
- --cookie-refresh checks the session with the provider every minute, so a
  user disabled there loses the panel within a minute. Without it, a session
  lasts the whole cookie lifetime of 168 h.
- --api-route answers the panel's own calls with 401 once a session ends,
  instead of a redirect that fetch() cannot follow to the provider.
- --cookie-csrf-per-request lets two tabs, or a reload, sign in at once.
- extraArgs come last: for a repeated flag, the last one wins.
*/}}
{{- define "lllm2.oidcArgs" -}}
{{- with .Values.ingress.oidc -}}
- {{ printf "--provider=%s" .provider | quote }}
- {{ printf "--oidc-issuer-url=%s" .issuerURL | quote }}
- {{ printf "--client-id=%s" .clientID | quote }}
- --http-address=:4180
- --upstream=http://127.0.0.1:8082/
- --code-challenge-method=S256
- --skip-provider-button=true
- --upstream-timeout=300s
- --cookie-refresh=1m
- --api-route=^/api/
- --cookie-csrf-per-request=true
- --cookie-csrf-per-request-limit=5
- --silence-ping-logging=true
{{- range .allowedGroups }}
- {{ printf "--allowed-group=%s" . | quote }}
{{- end }}
{{- range .allowedRoles }}
- {{ printf "--allowed-role=%s" . | quote }}
{{- end }}
{{- range (.emailDomains | default (list "*")) }}
- {{ printf "--email-domain=%s" . | quote }}
{{- end }}
{{- if .caConfigMap }}
- --provider-ca-file=/etc/oauth2-proxy/ca/ca.crt
{{- end }}
{{- range .extraArgs }}
- {{ . | quote }}
{{- end }}
{{- end -}}
{{- end -}}
