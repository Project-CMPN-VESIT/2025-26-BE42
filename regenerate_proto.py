#!/usr/bin/env python3
"""
regenerate_proto.py
───────────────────
Run this script from the project root after modifying orchestrator.proto.
It regenerates the Python gRPC stubs (orchestrator_pb2.py + orchestrator_pb2_grpc.py)
that the installer and federated client use.

Usage:
    # From the repo root (BE-Major-Project/)
    python regenerate_proto.py

Requirements:
    pip install grpcio-tools   (already in requirements.txt)
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PROTO_FILE = REPO_ROOT / "server" / "orchestration_agent" / "proto" / "orchestrator.proto"
OUT_DIR    = REPO_ROOT / "installer" / "runtime" / "grpc"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Write a temporary __init__.py so the grpc module is importable
(OUT_DIR / "__init__.py").touch(exist_ok=True)

print(f"[PROTO] Regenerating stubs from: {PROTO_FILE}")
print(f"[PROTO] Output directory      : {OUT_DIR}")

result = subprocess.run(
    [
        sys.executable,
        "-m", "grpc_tools.protoc",
        f"-I{PROTO_FILE.parent}",
        f"--python_out={OUT_DIR}",
        f"--grpc_python_out={OUT_DIR}",
        str(PROTO_FILE.name),
    ],
    cwd=str(PROTO_FILE.parent),
    capture_output=True,
    text=True,
)

print(result.stdout or "", end="")
if result.returncode != 0:
    print(result.stderr or "", end="", file=sys.stderr)
    sys.exit(f"[FAIL] protoc exited with code {result.returncode}")

# Fix the relative import that grpc_tools generates
grpc_file = OUT_DIR / "orchestrator_pb2_grpc.py"
if grpc_file.exists():
    content = grpc_file.read_text()
    # grpc_tools generates `import orchestrator_pb2 as ...`
    # but we need `from . import orchestrator_pb2 as ...`
    fixed = content.replace(
        "import orchestrator_pb2 as orchestrator__pb2",
        "from . import orchestrator_pb2 as orchestrator__pb2",
    )
    grpc_file.write_text(fixed)
    print("[PROTO] Fixed relative import in orchestrator_pb2_grpc.py")

print("[OK] Proto stubs regenerated successfully")
print()
print("Rebuild the Rust server:")
print("  cd server/orchestration_agent && cargo build --release")