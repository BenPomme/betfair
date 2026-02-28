# Betfair API client certificate (non-interactive login)

The Betfair API supports **non-interactive (bot) login** using a self-signed client certificate. This avoids browser-based login and is required for headless/automated use.

**Betfair login docs (reference):**

| Method | Doc | When to use |
|--------|-----|-------------|
| **Non-interactive (cert)** | [Non-Interactive (bot) login](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687915/Non-Interactive+bot+login) | Bots / headless (recommended for this project) |
| **Interactive – API endpoint** | [Interactive Login - API Endpoint](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687772/Interactive+Login+-+API+Endpoint) | Username/password POST; requires `Content-Type: application/x-www-form-urlencoded` for special chars |
| **Interactive – desktop** | [Interactive Login - Desktop Application](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687926/Interactive+Login+-+Desktop+Application) | Embed SSO page, capture token from redirect (2FA, terms, etc.) |

---

## 1. Create the certificate

From the project root:

```bash
./scripts/create_betfair_cert.sh
```

This creates under `certs/`:

- **client-2048.crt** — public certificate (you will upload this to Betfair)
- **client-2048.key** — private key (keep secret, never share or commit)

The script uses OpenSSL and follows Betfair’s requirements (2048-bit RSA, clientAuth extended key usage).

**Alternative (Windows / GUI):** [Certificate Generation With XCA](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687673/Certificate+Generation+With+XCA) — use XCA to generate key, CSR, and self-signed cert, then export the `.crt` and upload it to Betfair as below.

---

## 2. Link the certificate to your Betfair account (Spain)

1. Log in at [betfair.es](https://www.betfair.es).
2. Open: **https://myaccount.betfair.es/accountdetails/mysecurity?showAPI=1**
3. Scroll to **“Automated Betting Program Access”** and click **Edit**.
4. Click **Browse** and select **client-2048.crt** from your `certs/` folder.
5. Click **Upload Certificate**.

Other jurisdictions use their own account URLs (e.g. [betfair.com](https://myaccount.betfair.com/accountdetails/mysecurity?showAPI=1), [betfair.com.au](https://myaccount.betfair.com.au/accountdetails/mysecurity?showAPI=1), [betfair.it](https://myaccount.betfair.it/accountdetails/mysecurity?showAPI=1)).

---

## 3. Configure the project

In `.env`:

```bash
BF_CERTS_PATH=./certs
```

If the path is correct, `scripts/test_real_data.py` and the main engine will use **cert login** (no browser). Spain uses the endpoint `https://identitysso-cert.betfair.es/api/certlogin`; the client uses it when `BF_LOCALE=spain`.

---

## Notes

- **Username and password**: Betfair expects them [URL-encoded](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687915/Non-Interactive+bot+login#Details of a Login Request) in the cert login request. If you see `CERT_AUTH_REQUIRED` or login failures with special characters in the password, ensure your client encodes them (our stack uses `betfairlightweight`, which sends form-encoded data).
- **Do not commit** `client-2048.key` or any `.key`/`.pem` files. The `certs/` folder is listed in `.gitignore` for `*.key`, `*.pem`; keep `*.crt` local too if you prefer.
