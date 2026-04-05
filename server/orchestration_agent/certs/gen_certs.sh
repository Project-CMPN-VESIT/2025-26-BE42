#!/usr/bin/env bash
# gen_certs.sh — Regenerate all TLS certificates with correct SAN
# Run from: server/orchestration_agent/
# Usage:  bash certs/gen_certs.sh 192.168.1.7
#
# After running:
#   1. Copy certs/ca.pem → installer/runtime/keys/ca.pem
#   2. Rebuild Rust server: cargo build --release
#   3. Reinstall client (or just replace ~/.federated/keys/ca.pem)

set -euo pipefail

SERVER_IP="${1:-192.168.1.7}"
CERT_DIR="$(dirname "$0")"
DAYS_CA=3650
DAYS_SERVER=365
DAYS_CLIENT=365

echo "[GEN] Generating certificates for IP: ${SERVER_IP}"

# ── 1. CA ────────────────────────────────────────────────────────────────────
openssl genrsa -out "${CERT_DIR}/ca.key" 4096

openssl req -x509 -new -nodes \
    -key "${CERT_DIR}/ca.key" \
    -sha256 \
    -days ${DAYS_CA} \
    -subj "/CN=Federated-Root-CA" \
    -out "${CERT_DIR}/ca.pem"

echo "[OK] CA certificate generated"

# ── 2. Server key + CSR ───────────────────────────────────────────────────────
openssl genrsa -out "${CERT_DIR}/server.key" 2048

openssl req -new \
    -key "${CERT_DIR}/server.key" \
    -subj "/CN=${SERVER_IP}" \
    -out "${CERT_DIR}/server.csr"

# Server extension: SAN includes both IP and localhost
cat > "${CERT_DIR}/server.ext" <<EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage=digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth

subjectAltName = @alt_names

[alt_names]
IP.1 = ${SERVER_IP}
IP.2 = 127.0.0.1
DNS.1 = localhost
EOF

openssl x509 -req \
    -in "${CERT_DIR}/server.csr" \
    -CA "${CERT_DIR}/ca.pem" \
    -CAkey "${CERT_DIR}/ca.key" \
    -CAcreateserial \
    -out "${CERT_DIR}/server.pem" \
    -days ${DAYS_SERVER} \
    -sha256 \
    -extfile "${CERT_DIR}/server.ext"

echo "[OK] Server certificate generated (SAN: IP=${SERVER_IP}, DNS=localhost)"

# ── 3. Copy CA to installer runtime keys ─────────────────────────────────────
RUNTIME_CA="$(dirname "$0")/../../installer/runtime/keys/ca.pem"
if [ -d "$(dirname ${RUNTIME_CA})" ]; then
    cp "${CERT_DIR}/ca.pem" "${RUNTIME_CA}"
    echo "[OK] CA copied to installer/runtime/keys/ca.pem"
else
    echo "[WARN] Could not copy CA to installer — copy manually:"
    echo "       cp ${CERT_DIR}/ca.pem installer/runtime/keys/ca.pem"
fi

# ── 4. Verify ─────────────────────────────────────────────────────────────────
echo ""
echo "[VERIFY] Server certificate SAN:"
openssl x509 -in "${CERT_DIR}/server.pem" -noout -ext subjectAltName

echo ""
echo "[VERIFY] Chain valid:"
openssl verify -CAfile "${CERT_DIR}/ca.pem" "${CERT_DIR}/server.pem" && echo "✅ OK"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Next steps:"
echo "  1. cargo build --release   (in server/orchestration_agent/)"
echo "  2. Copy ca.pem to client:  ~/.federated/keys/ca.pem"
echo "  3. Remove ssl_target_name_override from grpc_client.py"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"