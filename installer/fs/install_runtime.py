import shutil
import stat
from pathlib import Path

BASE_DIR = Path.home() / ".federated"
INSTALLER_ROOT = Path(__file__).resolve().parents[1]

RUNTIME_SRC = INSTALLER_ROOT / "runtime"

def _chmod_exec(path: Path):
    path.chmod(stat.S_IRWXU)

def _chmod_tree(root: Path):
    for p in root.rglob("*"):
        try:
            p.chmod(stat.S_IRWXU)
        except Exception:
            pass


def install_runtime():
    # 1. bin/federated-client
    bin_dir = BASE_DIR / "bin"
    bin_dir.mkdir(exist_ok=True)

    src_client = RUNTIME_SRC / "federated_client.py"
    dst_client = bin_dir / "federated-client"

    shutil.copy2(src_client, dst_client)
    _chmod_exec(dst_client)

    # 2. agents
    if (BASE_DIR / "agents").exists():
        shutil.rmtree(BASE_DIR / "agents")
    shutil.copytree(
        RUNTIME_SRC / "agents",
        BASE_DIR / "agents",
    )

    agents_dst = BASE_DIR / "agents"
    _chmod_tree(agents_dst)


    # 3. grpc stubs
    if (BASE_DIR / "grpc").exists():
        shutil.rmtree(BASE_DIR / "grpc")
    shutil.copytree(
        RUNTIME_SRC / "grpc",
        BASE_DIR / "grpc",
    )

    grpc_dst = BASE_DIR / "grpc"
    _chmod_tree(grpc_dst)


    if (BASE_DIR / "configs").exists():
        shutil.rmtree(BASE_DIR / "configs")
    shutil.copytree(
        RUNTIME_SRC / "configs",
        BASE_DIR / "configs",
    )

    configs_dst = BASE_DIR / "configs"
    _chmod_tree(configs_dst)


    # 4. runtime guards & helpers
    if (BASE_DIR / "runtime").exists():
        shutil.rmtree(BASE_DIR / "runtime")
        
    runtime_dst = BASE_DIR / "runtime"
    runtime_dst.mkdir(exist_ok=True)

    for f in RUNTIME_SRC.glob("*.py"):
        if f.name == "federated_client.py":
            continue
        shutil.copy2(f, runtime_dst / f.name)

    _chmod_tree(runtime_dst)

    print("[OK] Runtime installed")
