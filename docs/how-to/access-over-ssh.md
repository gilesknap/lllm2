# Access over SSH

Forward the panel and API ports from your client machine:

```bash
ssh -N -L 8082:127.0.0.1:8082 -L 1920:127.0.0.1:1920 user@gpu-host
```

Open the usual localhost URL. `lllm2 panel --host 0.0.0.0` also exposes the
panel on a trusted LAN, with no login or TLS; anyone who can reach it can
control the workbench.
