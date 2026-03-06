"""
Create and log in a Betfair API client. Uses certs + locale from config.
Single place for login so main and scripts share the same behaviour.
"""
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import config

logger = logging.getLogger(__name__)

try:
    import betfairlightweight
    from betfairlightweight.exceptions import LoginError
    HAS_BETFAIR = True
except ImportError:
    HAS_BETFAIR = False
    LoginError = Exception  # noqa


def inspect_betfair_auth() -> Dict[str, Any]:
    cert_dir = Path(config.BF_CERTS_PATH) if config.BF_CERTS_PATH else None
    cert_exists = False
    key_exists = False
    pem_exists = False
    if cert_dir and cert_dir.exists():
        cert_exists = any(cert_dir.glob("*.crt"))
        key_exists = any(cert_dir.glob("*.key"))
        pem_exists = any(cert_dir.glob("*.pem"))
    credentials_present = bool(config.BF_USERNAME and config.BF_PASSWORD and config.BF_APP_KEY)
    valid_cert_pair = bool((cert_exists and key_exists) or pem_exists)
    if not credentials_present:
        primary_failure_reason = "credentials_missing"
    elif cert_dir and cert_dir.exists() and not valid_cert_pair:
        primary_failure_reason = "cert_missing"
    else:
        primary_failure_reason = ""
    return {
        "credentials_present": credentials_present,
        "certs_path": str(cert_dir) if cert_dir else "",
        "cert_dir_exists": bool(cert_dir and cert_dir.exists()),
        "cert_file_exists": cert_exists,
        "key_file_exists": key_exists,
        "pem_file_exists": pem_exists,
        "valid_cert_pair": valid_cert_pair,
        "login_mode": "certificate" if valid_cert_pair else "interactive",
        "session_status": "blocked" if primary_failure_reason else "ready",
        "primary_failure_reason": primary_failure_reason,
    }


def create_and_login(
    username: Optional[str] = None,
    password: Optional[str] = None,
    app_key: Optional[str] = None,
    certs_path: Optional[str] = None,
    locale: Optional[str] = None,
) -> Any:
    """
    Create betfairlightweight.APIClient, log in (cert if certs exist else interactive), return client.
    Uses config defaults for any None. Raises if login fails.
    """
    if not HAS_BETFAIR:
        raise RuntimeError("betfairlightweight not installed; pip install betfairlightweight")

    username = username or config.BF_USERNAME
    password = password or config.BF_PASSWORD
    app_key = app_key or config.BF_APP_KEY
    certs_path = certs_path or config.BF_CERTS_PATH
    locale = locale or (config.BF_LOCALE or "spain")

    if not username or not password or not app_key:
        raise ValueError("BF_USERNAME, BF_PASSWORD, BF_APP_KEY must be set (e.g. in .env)")

    auth_info = inspect_betfair_auth()
    use_certs = bool(auth_info.get("valid_cert_pair"))
    client = betfairlightweight.APIClient(
        username,
        password,
        app_key=app_key,
        certs=certs_path if use_certs else None,
        locale=locale,
    )

    if use_certs:
        try:
            logger.info("Logging in via certificate (identitysso-cert.betfair.%s)", "es" if locale == "spain" else "com")
            client.login()
            return client
        except LoginError as e:
            if "CERT_AUTH_REQUIRED" in str(e) or "INVALID_USERNAME_OR_PASSWORD" in str(e):
                logger.warning("Cert login failed (%s); falling back to interactive login.", e)
                client.login_interactive()
                return client
            raise

    logger.info("No valid cert pair at %s; using interactive login.", certs_path)
    client.login_interactive()
    return client
