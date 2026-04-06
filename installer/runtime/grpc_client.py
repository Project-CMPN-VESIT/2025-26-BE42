"""
grpc_client.py — Dual channel mTLS implementation

Phase 3:  Two-channel design — enrollment (server-TLS) and operational (mTLS).
Phase 4:  Server certificate MUST have a Subject Alternative Name (SAN) for
          the server's IP address. Regenerate with:
              bash certs/gen_certs.sh <SERVER_IP>
          then copy certs/ca.pem to installer/runtime/keys/ca.pem and reinstall.
Phase 9:  Exponential backoff retry on transient gRPC errors.
"""

import grpc
import time
import logging
from pathlib import Path
from typing import Optional

from runtime.tpm_guard import sign_message
from runtime.self_destruct import trigger_self_destruct
from runtime.grpc.orchestrator_pb2_grpc import OrchestratorStub

log = logging.getLogger(__name__)

BASE = Path.home() / ".federated"
KEYS = BASE / "keys"

_CA_PEM      = KEYS / "ca.pem"
_CLIENT_KEY  = KEYS / "client.key"
_CLIENT_CERT = KEYS / "client.pem"

# Phase 4: Channel options — no hostname override needed when certs have correct SAN.
_CHANNEL_OPTIONS = [
    ("grpc.keepalive_time_ms",              10_000),
    ("grpc.keepalive_timeout_ms",            5_000),
    ("grpc.keepalive_permit_without_calls",      1),
    ("grpc.http2.max_pings_without_data",        0),
]

_MAX_RETRY    = 5
_RETRY_BASE_S = 1.0


# ── Internal helpers ──────────────────────────────────────────────────────────

def _wait_ready(channel: grpc.Channel, timeout: float = 15.0):
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout)
    except grpc.FutureTimeoutError:
        raise ConnectionError(f"gRPC channel not ready within {timeout}s")


def _with_retry(fn, *args, **kwargs):
    """Phase 9: exponential backoff on transient gRPC errors."""
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


# ── Public API ────────────────────────────────────────────────────────────────

def create_enrollment_channel(server_addr: str) -> grpc.Channel:
    """
    Phase 3 — Channel 1: server-TLS only (no client certificate).
    Used during installation when the device does not yet have a signed cert.
    """
    if not _CA_PEM.exists():
        raise FileNotFoundError(f"CA certificate not found: {_CA_PEM}")

    creds = grpc.ssl_channel_credentials(
        root_certificates=_CA_PEM.read_bytes(),
    )
    channel = grpc.secure_channel(server_addr, creds, options=_CHANNEL_OPTIONS)
    _wait_ready(channel)
    log.info("[gRPC] Enrollment channel ready -> %s", server_addr)
    return channel


def create_mtls_channel(server_addr: str) -> grpc.Channel:
    """
    Phase 3 — Channel 2: full mutual TLS (client cert + server cert).
    Used for all operational calls after device enrollment.
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
    log.info("[gRPC] mTLS channel ready -> %s", server_addr)
    return channel


def create_grpc_stub(server_addr: str) -> OrchestratorStub:
    """
    Create the operational mTLS stub.
    Falls back to enrollment channel gracefully when client cert is absent.
    """
    try:
        channel = create_mtls_channel(server_addr)
        log.info("[gRPC] Using full mutual TLS")
    except FileNotFoundError as e:
        log.warning("[gRPC] Client cert absent, falling back to server-TLS: %s", e)
        try:
            channel = create_enrollment_channel(server_addr)
            log.warning("[gRPC] Using server-TLS only — enroll this device first")
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
    """Phase 9: wrap any gRPC call with retry + timeout."""
    return _with_retry(rpc_fn, request, timeout=timeout)