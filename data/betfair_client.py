"""
Create and log in a Betfair API client. Uses certs + locale from config.
Single place for login so main and scripts share the same behaviour.
"""
import logging
from pathlib import Path
from typing import Any, Optional

import config

logger = logging.getLogger(__name__)

try:
    import betfairlightweight
    from betfairlightweight.exceptions import LoginError
    HAS_BETFAIR = True
except ImportError:
    HAS_BETFAIR = False
    LoginError = Exception  # noqa


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

    use_certs = Path(certs_path).exists() if certs_path else False
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

    logger.info("No certs at %s; using interactive login.", certs_path)
    client.login_interactive()
    return client
