import subprocess
import time
import platform

IDLE_THRESHOLD_SECONDS = 300  # 5 minutes


def is_system_idle() -> bool:
    system = platform.system().lower()

    # ✅ Windows: no idle blocking
    if system == "windows":
        return True

    # ✅ Linux: xprintidle
    try:
        output = subprocess.check_output(
            ["xprintidle"], stderr=subprocess.DEVNULL
        )
        idle_ms = int(output.strip())
        return idle_ms > IDLE_THRESHOLD_SECONDS * 1000
    except Exception:
        # Fail-safe: assume NOT idle
        return False


def wait_until_idle():
    while not is_system_idle():
        time.sleep(30)
