# installer/runtime/grpc_client.py

import grpc
from pathlib import Path
from runtime.tpm_guard import sign_message
from runtime.self_destruct import trigger_self_destruct
from grpc.orchestrator_pb2_grpc import OrchestratorStub

BASE = Path.home() / ".federated"

def create_grpc_stub(server_addr: str):
    try:
        creds = grpc.ssl_channel_credentials(
            root_certificates=(BASE / "keys" / "ca.pem").read_bytes()
        )

        channel = grpc.secure_channel(server_addr, creds)
        stub = OrchestratorStub(channel)

        # Attach TPM-backed signer
        stub.sign_message = sign_message

        return stub

    except Exception:
        trigger_self_destruct("Secure gRPC channel setup failed")
