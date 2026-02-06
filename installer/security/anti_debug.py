import os
import sys
import ctypes
from .self_destruct import trigger_self_destruct

def anti_debug(strict=True, installer_mode=False):
    import time

    try:
        libc = ctypes.CDLL("libc.so.6")
        PTRACE_TRACEME = 0
        PTRACE_DETACH = 17

        ret = 0
        ret = libc.ptrace(PTRACE_TRACEME, 0, None, None)
        libc.ptrace(PTRACE_DETACH, 0, None, None)

        if ret != 0:
            sys.exit("[SECURITY] Debugger detected (ptrace)")

    except Exception:
        trigger_self_destruct("Debugger probe failed")

    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("TracerPid"):
                    tracer_pid = int(line.split(":")[1].strip())
                    if tracer_pid != 0:
                        if strict and not installer_mode:
                            sys.exit("[SECURITY] Debugger detected (TracerPid)")
                        else:
                            print("[WARN] Debugger detected (installer mode)")
    except FileNotFoundError:
        trigger_self_destruct("Debugger probe failed")

    for env in ["LD_PRELOAD", "LD_DEBUG", "PYTHONINSPECT", "PYTHONDEBUG"]:
        if env in os.environ:
            trigger_self_destruct("Suspicious environment detected")

    t1 = time.time()
    for _ in range(1000000):
        pass
    t2 = time.time()

    if (t2 - t1) > 1.2:
        trigger_self_destruct("[SECURITY] Timing anomaly detected")
