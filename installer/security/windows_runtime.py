import winreg
import sys

def check_vc_runtime():
    """
    Checks for Visual C++ 2015–2022 x64 runtime
    """
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
        )
        installed, _ = winreg.QueryValueEx(key, "Installed")
        winreg.CloseKey(key)

        if installed != 1:
            raise RuntimeError

        print("[OK] Visual C++ Runtime present")

    except Exception:
        print("""
[INSTALLER] Missing Visual C++ Runtime (x64)

Download and install:
https://aka.ms/vs/17/release/vc_redist.x64.exe

Then re-run the installer.
""")
        sys.exit(1)
