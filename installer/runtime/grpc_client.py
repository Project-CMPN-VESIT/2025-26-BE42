from __future__ import annotations
"""
grpc_client.py — Dual channel mTLS implementation

Channel 1 (enrollment): server-TLS only (no client cert) — used during install
Channel 2 (operational): full mTLS (client cert + server cert) — used at runtime

This eliminates the "certificate required" error when no client cert exists yet.
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

# gRPC channel options shared by both channels
_CHANNEL_OPTIONS = [
    ("grpc.keepalive_time_ms",          10_000),
    ("grpc.keepalive_timeout_ms",        5_000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.max_pings_without_data",   0),
    # Remove override once server cert SAN includes the real IP
    # If you regenerated certs with correct SAN, delete the next two lines:
    ("grpc.ssl_target_name_override", "localhost"),
    ("grpc.default_authority",        "localhost"),
]

_MAX_RETRY = 5
_RETRY_BASE_S = 1.0   # exponential backoff base


# ── Internal helpers ──────────────────────────────────────────────────────────

def _wait_ready(channel: grpc.Channel, timeout: float = 15.0):
    """Block until channel is ready or raise."""
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout)
    except grpc.FutureTimeoutError:
        raise ConnectionError(
            f"gRPC channel not ready within {timeout}s"
        )


def _with_retry(fn, *args, **kwargs):
    """Call fn with exponential backoff on transient errors."""
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
                log.warning("gRPC transient error (attempt %d/%d), retrying in %.1fs: %s",
                            attempt + 1, _MAX_RETRY, wait, e.details())
                time.sleep(wait)
                last_err = e
            else:
                raise
    raise last_err


# ── Public API ────────────────────────────────────────────────────────────────

def create_enrollment_channel(server_addr: str) -> grpc.Channel:
    """
    Server-TLS only channel (no client certificate).
    Used during installation / first enrollment.
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
    Used for operational calls after enrollment.
    Raises FileNotFoundError if client cert not yet installed.
    """
    for p in [_CA_PEM, _CLIENT_KEY, _CLIENT_CERT]:
        if not p.exists():
            raise FileNotFoundError(
                f"mTLS credential missing: {p}\n"
                "Run the installer first to enroll this device."
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
    Falls back to enrollment channel if client cert is missing
    (graceful for first-run scenarios).
    """
    try:
        channel = create_mtls_channel(server_addr)
        log.info("[gRPC] Using mTLS (full mutual TLS)")
    except FileNotFoundError as e:
        log.warning("[gRPC] Client cert missing, falling back to server-TLS: %s", e)
        try:
            channel = create_enrollment_channel(server_addr)
            log.warning("[gRPC] Using server-TLS only (enroll this device first)")
        except Exception as inner:
            trigger_self_destruct(f"Cannot establish any gRPC channel: {inner}")

    stub = OrchestratorStub(channel)
    stub._sign_message = sign_message   # attach TPM signer for use in pipeline
    return stub


def enrollment_stub(server_addr: str) -> OrchestratorStub:
    """
    Explicit enrollment-only stub (called from installer).
    """
    channel = create_enrollment_channel(server_addr)
    return OrchestratorStub(channel)


def call_with_retry(rpc_fn, request, timeout: float = 30.0):
    """
    Wrap any gRPC call with retry + timeout.
    Usage: call_with_retry(stub.SubmitReceipt, receipt_msg, timeout=15)
    """
    return _with_retry(rpc_fn, request, timeout=timeout)