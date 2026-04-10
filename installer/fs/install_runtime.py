"""
install_runtime.py  — FIXED VERSION

Key fixes applied (every line reviewed):
  FIX-1: install_mentalbert_model() removed from install_runtime().
          It is now called by installer_core.setup_software() AFTER
          create_venv() and install_python_deps() so that
          huggingface_hub / transformers are available in the venv.
  FIX-2: Step 10 now also writes ~/.federated/installer/__init__.py
          so that `from installer.security.integrity import ...` works
          at runtime (federated_client.py import chain).
  FIX-3: Also writes ~/.federated/runtime/__init__.py so that relative
          imports inside runtime_guard.py (`from .tpm_guard import ...`)
          resolve correctly when runtime is imported as a package.
"""

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
    return max(f.stat().st_size for f in model_files) > 1_000_000


def install_mentalbert_model():
    """
    FIX-1: This function is NO LONGER called from install_runtime().
    It must be called by installer_core.setup_software() AFTER
    create_venv() and install_python_deps() complete, so that
    huggingface_hub and transformers are present in the venv.
    """
    MODEL_DST = BASE_DIR / "models" / "mentalbert"
    MODEL_SRC = RUNTIME_SRC / "models" / "mentalbert"
    MODEL_DST.parent.mkdir(parents=True, exist_ok=True)

    print(f"[MODEL] Checking MentalBERT at {MODEL_DST}", flush=True)
    print("[DEBUG] MODEL_SRC:", MODEL_SRC)
    print("[DEBUG] MODEL_SRC exists:", MODEL_SRC.exists())

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

    # ── venv python is now guaranteed to exist (called after create_venv+deps) ─
    python_cmd = str(_venv_python()) if _venv_python().exists() else sys.executable

    download_script = r'''
import sys, os
from pathlib import Path

dst = sys.argv[1]
Path(dst).mkdir(parents=True, exist_ok=True)

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

if try_hub_download("mental/mental-bert-base-uncased", dst):
    print("[OK] MentalBERT downloaded from HuggingFace Hub", flush=True)
    sys.exit(0)

print("[MODEL] Falling back to bert-base-uncased (compatible architecture)", flush=True)
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

    if not MODEL_DST.exists():
        raise RuntimeError("Model folder was not created at all")

    if not _is_real_model(MODEL_DST):
        raise RuntimeError("Model exists but is invalid (likely LFS pointer or failed download)")

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

    # FIX-2a: ensure agents package __init__.py files exist so imports work
    for pkg_dir in [agents_dst,
                    agents_dst / "lda",
                    agents_dst / "lda" / "pipelines",
                    agents_dst / "trainer",
                    agents_dst / "dp",
                    agents_dst / "enc"]:
        init_file = pkg_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("")
            try:
                init_file.chmod(0o600)
            except Exception:
                pass

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

    # FIX-3: create runtime/__init__.py so relative imports inside
    # runtime_guard.py (`from .tpm_guard import ...`) resolve correctly
    # when `runtime` is imported as a package from federated_client.py.
    runtime_init = runtime_dst / "__init__.py"
    if not runtime_init.exists():
        runtime_init.write_text("")
        try:
            runtime_init.chmod(0o600)
        except Exception:
            pass

    # 5. grpc stubs
    grpc_dst = BASE_DIR / "runtime" / "grpc"
    if grpc_dst.exists():
        shutil.rmtree(grpc_dst)
    shutil.copytree(RUNTIME_SRC / "grpc", grpc_dst)
    _chmod_tree(grpc_dst)

    # FIX-2b: grpc sub-package __init__.py
    grpc_init = grpc_dst / "__init__.py"
    if not grpc_init.exists():
        grpc_init.write_text("")
        try:
            grpc_init.chmod(0o600)
        except Exception:
            pass

    # 6. core shared modules
    core_src = RUNTIME_SRC / "core"
    core_dst = BASE_DIR / "core"
    if core_src.exists():
        if core_dst.exists():
            shutil.rmtree(core_dst)
        shutil.copytree(core_src, core_dst)
        _chmod_tree(core_dst)

    # FIX-2c: core package __init__.py
    core_init = core_dst / "__init__.py"
    if not core_init.exists():
        core_init.write_text("")
        try:
            core_init.chmod(0o600)
        except Exception:
            pass

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

    # FIX-2d: CRITICAL — create ~/.federated/installer/__init__.py so that
    # `from installer.security.integrity import integrity_guard` resolves.
    # Without this, every agent (dp_agent, enc_agent, trainer, lda/main)
    # that calls `from installer.security.integrity import integrity_guard`
    # raises ModuleNotFoundError at runtime.
    installer_pkg = BASE_DIR / "installer"
    installer_pkg.mkdir(parents=True, exist_ok=True)
    installer_init = installer_pkg / "__init__.py"
    if not installer_init.exists():
        installer_init.write_text("")
        try:
            installer_init.chmod(0o600)
        except Exception:
            pass

    # FIX-2e: also ensure installer/security/__init__.py is present
    # (it should be copied above, but guarantee it exists)
    sec_init = installer_security_dst / "__init__.py"
    if not sec_init.exists():
        sec_init.write_text(
            "from .anti_debug import anti_debug\n"
            "from .tpm_attestation import tpm_attestation\n"
        )
        try:
            sec_init.chmod(0o600)
        except Exception:
            pass

    # NOTE: install_mentalbert_model() is intentionally NOT called here.
    # It is called by installer_core.setup_software() AFTER install_python_deps()
    # so the venv python has huggingface_hub and transformers available.

    print("[OK] Runtime installed successfully")