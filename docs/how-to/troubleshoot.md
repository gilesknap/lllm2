# Troubleshoot

- **No engine:** check `lllm2 engines list` and the configured search roots;
  run `lllm2 engines install cuda` if needed.
- **No GPU:** check `nvidia-smi` and the selected engine's device probe.
- **Download certificate errors:** engine and model downloads automatically use
  RHEL's system CA bundle when present. Other systems use Python's default trust
  settings. Explicit `SSL_CERT_FILE` or `SSL_CERT_DIR` settings take precedence;
  check those settings if verification still fails. Site certificates should be
  installed in the system trust store.
- **Startup fails:** read the engine log, reduce context or GPU layers, and
  disable unsupported acceleration options. Automatic placement needs an
  engine with memory-fitting support.
- **Client exceeds context:** increase context or reduce slots, then restart
  the model and client. Available memory still limits the allocation.
