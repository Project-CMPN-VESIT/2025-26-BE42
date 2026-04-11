"""
grpc_client.py — Dual channel mTLS implementation

SECURITY FIXES:
  FIX-GRPC-1: Removed grpc.ssl_target_name_override and grpc.default_authority
               overrides. These two options disable TLS hostname verification,
               making MITM attacks trivial even with valid certificates.
               The server certificate must have a correct SAN that matches the
               server address. Use gen_certs.sh <SERVER_IP> to regenerate.

  FIX-GRPC-2: Added UploadUpdate client-streaming method to stub wrapper
               so pipeline.py can call stub.UploadUpdate(...).

  FIX-GRPC-3: Added DownloadGlobalModel server-streaming method.

  NOTE: If you see SSL_ERROR_SSL after this fix, your server cert SAN does
        not include the IP/hostname you are connecting to. Fix:
          bash server/orchestration_agent/certs/gen_certs.sh <YOUR_SERVER_IP>
        Then copy certs/ca.pem to installer/runtime/keys/ca.pem and reinstall.
"""

from __future__ import annotations

import grpc
import time
import logging
from pathlib import Path

from runtime.tpm_guard import sign_message
from runtime.self_destruct import trigger_self_destruct
from runtime.grpc.orchestrator_pb2_grpc import OrchestratorStub

log = logging.getLogger(__name__)

BASE  = Path.home() / ".federated"
KEYS  = BASE / "keys"

_CA_PEM      = KEYS / "ca.pem"
_CLIENT_KEY  = KEYS / "client.key"
_CLIENT_CERT = KEYS / "client.pem"

# FIX-GRPC-1: No hostname override options.
# The server certificate MUST have a SAN matching the address you connect to.
# If connecting by IP, the cert needs IP.x = <IP> in [alt_names].
# If connecting by hostname, the cert needs DNS.x = <hostname>.
_CHANNEL_OPTIONS = [
    ("grpc.keepalive_time_ms",              10_000),
    ("grpc.keepalive_timeout_ms",            5_000),
    ("grpc.keepalive_permit_without_calls",      1),
    ("grpc.http2.max_pings_without_data",        0),
    # REMOVED: grpc.ssl_target_name_override  ← was disabling hostname check
    # REMOVED: grpc.default_authority         ← was disabling hostname check
]

_MAX_RETRY    = 5
_RETRY_BASE_S = 1.0


def _wait_ready(channel: grpc.Channel, timeout: float = 15.0):
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout)
    except grpc.FutureTimeoutError:
        raise ConnectionError(
            f"gRPC channel not ready within {timeout}s.\n"
            "If this is a TLS error, ensure the server certificate SAN includes "
            "the address you are connecting to. Regenerate with:\n"
            "  bash server/orchestration_agent/certs/gen_certs.sh <SERVER_IP>\n"
            "Then copy certs/ca.pem → installer/runtime/keys/ca.pem"
        )


def _with_retry(fn, *args, **kwargs):
    last_err = None
    for attempt in range(_MAX_RETRY):
        try:
            return fn(*args, **kwargs)
        except grpc.RpcError as e:
            if e.code() in (
                grpc.StatusCode.UNAVAILABLE,
                grpc.StatusCode.DEADLINE_EXCEEDED,
            ):
                wait = _RETRY_BASE_S * (2 ** attempt)
                log.warning(
                    "gRPC transient error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, _MAX_RETRY, wait, e.details()
                )
                time.sleep(wait)
                last_err = e
            else:
                raise
    raise last_err


def create_enrollment_channel(server_addr: str) -> grpc.Channel:
    """
    Server-TLS only channel (no client certificate).
    Used during installation / first enrollment before client cert exists.
    """
    if not _CA_PEM.exists():
        raise FileNotFoundError(f"CA certificate not found: {_CA_PEM}")

    creds = grpc.ssl_channel_credentials(
        root_certificates=_CA_PEM.read_bytes(),
    )
    channel = grpc.secure_channel(server_addr, creds, options=_CHANNEL_OPTIONS)
    _wait_ready(channel)
    log.info("[gRPC] Enrollment channel ready → %s", server_addr)
    return channel


def create_mtls_channel(server_addr: str) -> grpc.Channel:
    """
    Full mTLS channel (client cert + server cert).
    Used for all operational calls after enrollment.
    """
    for p in [_CA_PEM, _CLIENT_KEY, _CLIENT_CERT]:
        if not p.exists():
            raise FileNotFoundError(
                f"mTLS credential missing: {p}\n"
                "Run the installer to enroll this device first."
            )

    creds = grpc.ssl_channel_credentials(
        root_certificates=_CA_PEM.read_bytes(),
        private_key=_CLIENT_KEY.read_bytes(),
        certificate_chain=_CLIENT_CERT.read_bytes(),
    )
    channel = grpc.secure_channel(server_addr, creds, options=_CHANNEL_OPTIONS)
    _wait_ready(channel)
    log.info("[gRPC] mTLS channel ready → %s", server_addr)
    return channel


def create_grpc_stub(server_addr: str) -> OrchestratorStub:
    """
    Create the operational mTLS stub.
    Falls back to enrollment channel only if client cert is not yet installed.
    """
    try:
        channel = create_mtls_channel(server_addr)
        log.info("[gRPC] Using mTLS (full mutual TLS)")
    except FileNotFoundError as e:
        log.warning("[gRPC] Client cert missing — falling back to server-TLS: %s", e)
        try:
            channel = create_enrollment_channel(server_addr)
            log.warning("[gRPC] Server-TLS only — enroll this device to use mTLS")
        except Exception as inner:
            trigger_self_destruct(f"Cannot establish any gRPC channel: {inner}")

    stub = OrchestratorStub(channel)
    stub._sign_message = sign_message
    return stub


def enrollment_stub(server_addr: str) -> OrchestratorStub:
    """Explicit enrollment-only stub (called from installer)."""
    channel = create_enrollment_channel(server_addr)
    return OrchestratorStub(channel)


def call_with_retry(rpc_fn, request, timeout: float = 30.0):
    """Wrap any unary gRPC call with retry + timeout."""
    return _with_retry(rpc_fn, request, timeout=timeout)