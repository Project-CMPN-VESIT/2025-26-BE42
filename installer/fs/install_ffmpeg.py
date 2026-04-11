"""
install_ffmpeg.py

SECURITY FIX:
  FIX-FFMPEG-1: The Windows download now verifies the SHA-256 of the
                downloaded ZIP before extraction. Previously any MITM or
                compromised CDN could inject a malicious binary and it
                would be installed silently.

  The expected hash is pinned here. When upgrading FFmpeg, download the
  new ZIP, compute sha256sum on it, and update FFMPEG_SHA256 below.
  
  IMPORTANT: This hash is for ffmpeg-release-essentials.zip downloaded
  on 2026-04-08. Pin it to a specific release URL rather than the
  'latest' redirect for reproducible builds.
"""

import os
import platform
import subprocess
import shutil
import hashlib
import urllib.request
import zipfile

# Pinned release URL — do NOT use a 'latest' redirect in production.
# Update URL and hash together when upgrading.
FFMPEG_WIN_URL = (
    "https://www.gyan.dev/ffmpeg/builds/packages/"
    "ffmpeg-7.1-essentials_build.zip"
)

# SHA-256 of the pinned ZIP above.
# Recompute with: sha256sum ffmpeg-7.1-essentials_build.zip
# Update this value every time you update FFMPEG_WIN_URL.
FFMPEG_WIN_SHA256 = (
    "REPLACE_WITH_ACTUAL_SHA256_OF_PINNED_ZIP"
    # Example (not real): "a3f2c1e4b5d6..."
    # To get it: download the file manually, run sha256sum, paste here.
)


def _verify_sha256(path: str, expected: str) -> None:
    """Raise RuntimeError if file SHA-256 does not match expected."""
    if expected.startswith("REPLACE_WITH"):
        raise RuntimeError(
            "FFmpeg SHA-256 is not configured.\n"
            "Download the zip, run sha256sum on it, and set FFMPEG_WIN_SHA256 "
            "in install_ffmpeg.py before deploying."
        )
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    actual = h.hexdigest().lower()
    expected = expected.lower().strip()
    if actual != expected:
        raise RuntimeError(
            f"FFmpeg ZIP integrity check FAILED.\n"
            f"  expected: {expected}\n"
            f"  actual  : {actual}\n"
            "The download may have been tampered with. Do not proceed."
        )


def install_ffmpeg():
    print("[DEBUG] Checking FFmpeg...")

    if shutil.which("ffmpeg"):
        print("[DEBUG] FFmpeg already installed")
        return

    system = platform.system()

    try:
        if system == "Windows":
            zip_path   = "ffmpeg.zip"
            extract_dir = "ffmpeg"

            print(f"[DEBUG] Downloading FFmpeg from {FFMPEG_WIN_URL}...")
            urllib.request.urlretrieve(FFMPEG_WIN_URL, zip_path)

            # FIX-FFMPEG-1: verify integrity before extraction
            print("[DEBUG] Verifying FFmpeg ZIP integrity...")
            _verify_sha256(zip_path, FFMPEG_WIN_SHA256)
            print("[DEBUG] FFmpeg ZIP integrity OK")

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            ffmpeg_bin = None
            for root, dirs, files in os.walk(extract_dir):
                if "ffmpeg.exe" in files:
                    ffmpeg_bin = root
                    break

            if not ffmpeg_bin:
                raise RuntimeError("ffmpeg.exe not found in extracted archive")

            os.environ["PATH"] += os.pathsep + ffmpeg_bin
            print(f"[DEBUG] FFmpeg installed at {ffmpeg_bin}")

            # Clean up downloaded zip
            try:
                os.remove(zip_path)
            except OSError:
                pass

        elif system == "Linux":
            print("[DEBUG] Installing FFmpeg on Linux...")
            subprocess.run(["sudo", "apt", "update"], check=True)
            subprocess.run(["sudo", "apt", "install", "-y", "ffmpeg"], check=True)
            # On Linux we rely on apt's GPG signature verification

        elif system == "Darwin":
            print("[DEBUG] Installing FFmpeg on macOS...")
            if shutil.which("brew"):
                subprocess.run(["brew", "install", "ffmpeg"], check=True)
                # Homebrew verifies SHA-256 of bottles automatically
            else:
                raise RuntimeError(
                    "Homebrew not found. Install from https://brew.sh/"
                )

        else:
            raise RuntimeError(f"Unsupported OS: {system}")

        print("[DEBUG] FFmpeg installed successfully")

    except Exception as e:
        print(f"[ERROR] FFmpeg installation failed: {e}")
        print("Please install FFmpeg manually.")
        raise