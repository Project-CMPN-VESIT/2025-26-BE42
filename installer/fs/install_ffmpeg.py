import os
import platform
import subprocess
import shutil
import urllib.request
import zipfile

def install_ffmpeg():
    print("[DEBUG] Checking FFmpeg...")

    if shutil.which("ffmpeg"):
        print("[DEBUG] FFmpeg already installed")
        return

    system = platform.system()

    try:
        if system == "Windows":
            url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
            zip_path = "ffmpeg.zip"
            extract_dir = "ffmpeg"

            print("[DEBUG] Downloading FFmpeg...")

            urllib.request.urlretrieve(url, zip_path)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            # Find bin folder
            for root, dirs, files in os.walk(extract_dir):
                if "ffmpeg.exe" in files:
                    ffmpeg_bin = root
                    break

            # Add to PATH (current session)
            os.environ["PATH"] += os.pathsep + ffmpeg_bin

            print(f"[DEBUG] FFmpeg installed at {ffmpeg_bin}")

        elif system == "Linux":
            print("[DEBUG] Installing FFmpeg on Linux...")
            subprocess.run(["sudo", "apt", "update"], check=True)
            subprocess.run(["sudo", "apt", "install", "-y", "ffmpeg"], check=True)

        elif system == "Darwin":
            print("[DEBUG] Installing FFmpeg on macOS...")
            if shutil.which("brew"):
                subprocess.run(["brew", "install", "ffmpeg"], check=True)
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