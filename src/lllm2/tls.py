"""Certificate trust for engine and model downloads."""

import os
import ssl
import sys
from pathlib import Path

SYSTEM_CA_BUNDLE = Path("/etc/pki/tls/certs/ca-bundle.crt")


def download_context() -> ssl.SSLContext:
    """Use the native Red Hat CA bundle unless trust was explicitly configured."""
    # Standalone Python can expect /etc/ssl/cert.pem, which RHEL8 may lack.
    # Prefer the distro's managed bundle, including site-installed certificates.
    # Leave explicit OpenSSL overrides alone, even when empty or invalid.
    if (
        sys.platform == "linux"
        and "SSL_CERT_FILE" not in os.environ
        and "SSL_CERT_DIR" not in os.environ
        and SYSTEM_CA_BUNDLE.is_file()
    ):
        return ssl.create_default_context(cafile=str(SYSTEM_CA_BUNDLE))
    return ssl.create_default_context()
