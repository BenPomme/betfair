#!/usr/bin/env bash
# Create a self-signed client certificate for Betfair API non-interactive (bot) login.
# Follows: https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687915/Non-Interactive+bot+login
#
# Usage: ./scripts/create_betfair_cert.sh [output_dir]
# Default output_dir: ./certs (from project root). Must run from project root or pass absolute path.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CERTS_DIR="${1:-$PROJECT_ROOT/certs}"
KEY_FILE="client-2048.key"
CSR_FILE="client-2048.csr"
CRT_FILE="client-2048.crt"
CONF_FILE="openssl-betfair.cnf"

cd "$PROJECT_ROOT"
mkdir -p "$CERTS_DIR"
cd "$CERTS_DIR"

# OpenSSL config for client cert (Betfair requires clientAuth, etc.)
# [req] and distinguished_name are needed for openssl req -new; [ssl_client] for x509 -extensions
cat > "$CONF_FILE" << 'EOF'
[ req ]
distinguished_name = dn
prompt = no

[ dn ]
C = ES
ST = Madrid
L = Madrid
O = BetfairAPI
OU = Automation
CN = Betfair API Client

[ ssl_client ]
basicConstraints = CA:FALSE
nsCertType = client
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = clientAuth
EOF

echo "Generating 2048-bit RSA key..."
openssl genrsa -out "$KEY_FILE" 2048

echo "Creating certificate signing request (CSR)..."
openssl req -new -config "$CONF_FILE" -key "$KEY_FILE" -out "$CSR_FILE"

echo "Self-signing certificate (valid 365 days)..."
openssl x509 -req -days 365 -in "$CSR_FILE" -signkey "$KEY_FILE" -out "$CRT_FILE" \
  -extfile "$CONF_FILE" -extensions ssl_client

rm -f "$CSR_FILE"
echo ""
echo "Done. Files in $CERTS_DIR:"
echo "  - $CRT_FILE  (upload this to Betfair)"
echo "  - $KEY_FILE  (keep private, do not share)"
echo ""
echo "Next steps:"
echo "  1. Upload $CRT_FILE to Betfair: https://myaccount.betfair.es/accountdetails/mysecurity?showAPI=1"
echo "     → Automated Betting Program Access → Edit → Browse → select $CRT_FILE → Upload Certificate"
echo "  2. Set in .env: BF_CERTS_PATH=$CERTS_DIR"
echo "  3. Run: python scripts/test_real_data.py  (cert login will be used automatically)"
echo ""
echo "Ref: https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687915/Non-Interactive+bot+login"
