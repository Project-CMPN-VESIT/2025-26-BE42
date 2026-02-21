import shutil
import stat
import platform
from pathlib import Path
import sys
from runtime.validate_deps import main as validate_deps

BASE_DIR = Path.home() / ".federated"

def get_installer_root() -> Path:
    """
    Returns the root directory of the installer.
    Works for:
    - normal python execution
    - PyInstaller --onefile execution
    """
    if getattr(sys, 'frozen', False):
        # PyInstaller onefile mode
        return Path(sys._MEIPASS)
    else:
        # Normal python execution
        return Path(__file__).resolve().parents[1]

INSTALLER_ROOT = get_installer_root()
RUNTIME_SRC = INSTALLER_ROOT / "runtime"


# --------------------------------------------------
# Permissions helpers
# --------------------------------------------------

def _chmod_exec(path: Path):
    try:
        path.chmod(stat.S_IRWXU)
    except Exception:
        pass


def _chmod_tree(root: Path):
    for p in root.rglob("*"):
        try:
            p.chmod(stat.S_IRWXU)
        except Exception:
            pass


# --------------------------------------------------
# Windows dependency installer
# --------------------------------------------------

def install_windows_deps():
    if platform.system().lower() != "windows":
        return

    src = RUNTIME_SRC / "deps" / "windows"
    dst = BASE_DIR / "deps" / "windows"

    if not src.exists():
        raise RuntimeError(
            "[INSTALLER] Missing runtime/deps/windows in installer"
        )

    # Fresh install to avoid mismatched binaries
    if dst.exists():
        shutil.rmtree(dst)

    shutil.copytree(src, dst)

    _chmod_tree(dst)

    # Validation (hard fail if missing)
    openface_bin = dst / "OpenFace" / "FeatureExtraction.exe"
    opensmile_bin = next(
        dst.glob("opensmile/**/SMILExtract.exe"),
        None
    )

    if not openface_bin.exists():
        raise RuntimeError(
            "[INSTALLER] FeatureExtraction.exe missing after install"
        )

    if opensmile_bin is None:
        raise RuntimeError(
            "[INSTALLER] SMILExtract.exe missing after install"
        )

    print("[OK] Windows OpenFace + openSMILE installed")


# --------------------------------------------------
# Runtime installer
# --------------------------------------------------

def install_runtime():
    # 1. bin/federated-client
    bin_dir = BASE_DIR / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    src_client = RUNTIME_SRC / "federated_client.py"
    dst_client = bin_dir / "federated-client"

    shutil.copy2(src_client, dst_client)
    _chmod_exec(dst_client)

    # 2. agents
    agents_dst = BASE_DIR / "agents"
    if agents_dst.exists():
        shutil.rmtree(agents_dst)

    shutil.copytree(
        RUNTIME_SRC / "agents",
        agents_dst,
    )
    _chmod_tree(agents_dst)

    # 3. grpc stubs
    grpc_dst = BASE_DIR / "grpc"
    if grpc_dst.exists():
        shutil.rmtree(grpc_dst)

    shutil.copytree(
        RUNTIME_SRC / "grpc",
        grpc_dst,
    )
    _chmod_tree(grpc_dst)

    # 4. configs
    configs_dst = BASE_DIR / "configs"
    if configs_dst.exists():
        shutil.rmtree(configs_dst)

    shutil.copytree(
        RUNTIME_SRC / "configs",
        configs_dst,
    )
    _chmod_tree(configs_dst)

    # 5. runtime guards & helpers
    runtime_dst = BASE_DIR / "runtime"
    if runtime_dst.exists():
        shutil.rmtree(runtime_dst)

    runtime_dst.mkdir(parents=True, exist_ok=True)

    for f in RUNTIME_SRC.glob("*.py"):
        if f.name == "federated_client.py":
            continue
        shutil.copy2(f, runtime_dst / f.name)

    _chmod_tree(runtime_dst)

    # 6. core shared modules
    core_src = RUNTIME_SRC / "core"
    core_dst = BASE_DIR / "core"

    if core_src.exists():
        if core_dst.exists():
            shutil.rmtree(core_dst)

        shutil.copytree(core_src, core_dst)
        _chmod_tree(core_dst)

    # 6. Windows native dependencies (OpenFace + openSMILE)
    install_windows_deps()

    if platform.system().lower() == "windows":
        print("[STEP] Validating runtime dependencies")
        validate_deps()
    else:
        print("[INFO] Skipping Windows dependency validation (Linux mode)")

    # 5. validation helper
    shutil.copy2(
        RUNTIME_SRC / "validate_deps.py",
        BASE_DIR / "runtime" / "validate_deps.py"
    )

    print("[OK] Runtime installed successfully")
