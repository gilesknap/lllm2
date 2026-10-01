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
