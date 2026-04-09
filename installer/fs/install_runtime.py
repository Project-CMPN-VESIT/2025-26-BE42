import shutil
import stat
import platform
import subprocess
import sys
from pathlib import Path

IS_WINDOWS = platform.system().lower() == "windows"

BASE_DIR = Path.home() / ".federated"
KEYS_DIR = Path.home() / ".federated" / "keys"


def get_installer_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]


INSTALLER_ROOT = get_installer_root()
RUNTIME_SRC = INSTALLER_ROOT / "runtime"


# ── Permissions helpers ───────────────────────────────────────────────────────

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


# ── Venv python path ──────────────────────────────────────────────────────────

def _venv_python() -> Path:
    if IS_WINDOWS:
        return BASE_DIR / "venv" / "Scripts" / "python.exe"
    return BASE_DIR / "venv" / "bin" / "python"


# ── MentalBERT installer ──────────────────────────────────────────────────────

def _is_real_model(directory: Path) -> bool:
    """
    Return True only if directory contains actual model weights
    (not git-lfs pointer files).  LFS pointers are tiny text files ~130 bytes.
    """
    model_files = (
        list(directory.glob("*.bin"))
        + list(directory.glob("*.safetensors"))
        + list(directory.glob("pytorch_model*.bin"))
    )
    if not model_files:
        return False
    # Real model files are megabytes; LFS pointers are ~100-200 bytes
    return max(f.stat().st_size for f in model_files) > 1_000_000


def install_mentalbert_model():
    MODEL_DST = BASE_DIR / "models" / "mentalbert"
    MODEL_SRC = RUNTIME_SRC / "models" / "mentalbert"

    print(f"[MODEL] Checking MentalBERT at {MODEL_DST}", flush=True)

    # ── Check existing install ────────────────────────────────────────────────
    if MODEL_DST.exists():
        if _is_real_model(MODEL_DST):
            print("[MODEL] Already installed and valid, skipping")
            return
        else:
            print(
                "[MODEL] Found incomplete model (git-lfs pointer files detected). "
                "Removing and re-downloading…",
                flush=True,
            )
            shutil.rmtree(MODEL_DST)

    # ── Try installer payload ─────────────────────────────────────────────────
    if MODEL_SRC.exists() and _is_real_model(MODEL_SRC):
        print("[MODEL] Installing from installer payload…")
        shutil.copytree(MODEL_SRC, MODEL_DST)
        print("[OK] MentalBERT model installed from installer payload")
        return

    if MODEL_SRC.exists():
        print(
            "[WARN] Installer payload contains git-lfs pointer files, not real weights. "
            "Will download from HuggingFace Hub instead.",
            flush=True,
        )

    # ── Download via venv python ──────────────────────────────────────────────
    python_cmd = str(_venv_python()) if _venv_python().exists() else sys.executable

    download_script = r'''
import sys, os
from pathlib import Path

dst = sys.argv[1]

def try_hub_download(repo_id, dst):
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=dst,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*", "*.ot"],
        )
        return True
    except Exception as e:
        print(f"[WARN] Hub download failed for {repo_id}: {e}", flush=True)
        return False

# Primary: mental/mental-bert-base-uncased
if try_hub_download("mental/mental-bert-base-uncased", dst):
    print("[OK] MentalBERT downloaded from HuggingFace Hub", flush=True)
    sys.exit(0)

# Fallback: standard BERT (compatible architecture)
print("[MODEL] Falling back to bert-base-uncased (compatible with MentalBERT architecture)", flush=True)
try:
    from transformers import AutoModel, AutoTokenizer
    Path(dst).mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model.save_pretrained(dst)
    tokenizer.save_pretrained(dst)
    print("[OK] bert-base-uncased installed as MentalBERT fallback", flush=True)
    sys.exit(0)
except Exception as e2:
    print(f"[ERROR] Fallback also failed: {e2}", flush=True)
    sys.exit(1)
'''

    print("[MODEL] Downloading MentalBERT from HuggingFace Hub…", flush=True)
    result = subprocess.run(
        [python_cmd, "-c", download_script, str(MODEL_DST)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    print(result.stdout or "", end="", flush=True)

    if result.returncode != 0:
        print(result.stderr or "", end="", file=sys.stderr)
        raise RuntimeError("MentalBERT model installation failed — check network or HuggingFace token.")

    print("[OK] MentalBERT model ready", flush=True)


# ── Windows native deps ───────────────────────────────────────────────────────

def install_windows_deps():
    if platform.system().lower() != "windows":
        return

    src = RUNTIME_SRC / "deps" / "windows"
    dst_root = BASE_DIR / "deps"
    dst = dst_root / "windows"

    print("[DEBUG] RUNTIME_SRC:", RUNTIME_SRC)
    deps_dir = RUNTIME_SRC / "deps"
    if deps_dir.exists():
        print("[DEBUG] Contents:", list(deps_dir.iterdir()))

    if dst.exists():
        shutil.rmtree(dst)

    if dst_root.exists():
        for item in dst_root.iterdir():
            if item.name == "windows":
                shutil.rmtree(item)

    shutil.copytree(src, dst)
    _chmod_tree(dst)

    openface_bin = dst / "OpenFace" / "FeatureExtraction.exe"
    opensmile_bin = next(dst.glob("opensmile/**/SMILExtract.exe"), None)

    if not openface_bin.exists():
        raise RuntimeError("[INSTALLER] FeatureExtraction.exe missing after install")

    if opensmile_bin is None:
        raise RuntimeError("[INSTALLER] SMILExtract.exe missing after install")

    print("[OK] Windows OpenFace + openSMILE installed")


# ── Runtime installer ─────────────────────────────────────────────────────────

def install_runtime():
    # 1. bin/federated-client
    bin_dir = BASE_DIR / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    src_client = RUNTIME_SRC / "federated_client.py"
    dst_client = bin_dir / "federated-client"

    shutil.copy2(src_client, dst_client)
    _chmod_exec(dst_client)

    # ── Windows TPM signer ────────────────────────────────────────────────────
    if IS_WINDOWS:
        signer_src = RUNTIME_SRC / "windows_signer.exe"
        signer_dst = bin_dir / "windows_signer.exe"

        print("[DEBUG] Copying Windows signer from:", signer_src)
        print("[DEBUG] Exists?:", signer_src.exists())

        if not signer_src.exists():
            raise RuntimeError("windows_signer.exe missing from runtime")

        shutil.copy2(signer_src, signer_dst)
        _chmod_exec(signer_dst)
        print("[OK] Windows TPM signer installed")

    # 2. agents
    agents_dst = BASE_DIR / "agents"
    if agents_dst.exists():
        shutil.rmtree(agents_dst)
    shutil.copytree(RUNTIME_SRC / "agents", agents_dst)
    _chmod_tree(agents_dst)

    # 3. configs
    configs_dst = BASE_DIR / "configs"
    if configs_dst.exists():
        shutil.rmtree(configs_dst)
    shutil.copytree(RUNTIME_SRC / "configs", configs_dst)
    _chmod_tree(configs_dst)

    # 4. runtime guards & helpers
    runtime_dst = BASE_DIR / "runtime"
    if runtime_dst.exists():
        shutil.rmtree(runtime_dst)
    runtime_dst.mkdir(parents=True, exist_ok=True)

    for f in RUNTIME_SRC.glob("*.py"):
        if f.name == "federated_client.py":
            continue
        shutil.copy2(f, runtime_dst / f.name)
    _chmod_tree(runtime_dst)

    # 5. grpc stubs
    grpc_dst = BASE_DIR / "runtime" / "grpc"
    if grpc_dst.exists():
        shutil.rmtree(grpc_dst)
    shutil.copytree(RUNTIME_SRC / "grpc", grpc_dst)
    _chmod_tree(grpc_dst)

    # 6. core shared modules
    core_src = RUNTIME_SRC / "core"
    core_dst = BASE_DIR / "core"
    if core_src.exists():
        if core_dst.exists():
            shutil.rmtree(core_dst)
        shutil.copytree(core_src, core_dst)
        _chmod_tree(core_dst)

    # 7. Windows native deps (OpenFace + openSMILE)
    install_windows_deps()

    # 8. CA certificate
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parent

    ca_src = base / "runtime" / "keys" / "ca.pem"
    ca_dst = KEYS_DIR / "ca.pem"
    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ca_src, ca_dst)
    print("[OK] CA certificate installed")

    # 9. validate_deps helper
    shutil.copy2(
        RUNTIME_SRC / "validate_deps.py",
        BASE_DIR / "runtime" / "validate_deps.py",
    )

    if IS_WINDOWS:
        shutil.copy2(
            RUNTIME_SRC / "windows_signer.exe",
            BASE_DIR / "bin" / "windows_signer.exe",
        )

    # 10. installer/security subset
    installer_security_src = INSTALLER_ROOT / "installer" / "security"
    installer_security_dst = BASE_DIR / "installer" / "security"

    if installer_security_src.exists():
        if installer_security_dst.exists():
            shutil.rmtree(installer_security_dst.parent)
        installer_security_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(installer_security_src, installer_security_dst)
        _chmod_tree(installer_security_dst)
        print("[OK] installer.security module installed")
    else:
        print("[WARN] installer/security not found in installer package")

    # 11. MentalBERT (LFS-aware download)
    install_mentalbert_model()

    print("[OK] Runtime installed successfully")