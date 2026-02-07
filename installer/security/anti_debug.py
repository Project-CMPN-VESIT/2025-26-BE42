def anti_debug(strict=True, installer_mode=False):
    import platform
    import time
    import os
    import sys

    system = platform.system().lower()

    # --------------------------------------------------
    # Windows: SAFE CHECKS ONLY
    # --------------------------------------------------
    if system == "windows":
        # Windows does NOT support ptrace or /proc
        # We only do lightweight checks here

        suspicious_envs = [
            "PYTHONINSPECT",
            "PYTHONDEBUG",
            "PYDEVD_LOAD_VALUES_ASYNC",
        ]

        for env in suspicious_envs:
            if env in os.environ:
                if strict and not installer_mode:
                    sys.exit("[SECURITY] Debug environment detected")
                else:
                    print("[WARN] Debug environment detected (installer mode)")

        # Timing check (very relaxed on Windows)
        t1 = time.time()
        time.sleep(0.01)
        t2 = time.time()

        if (t2 - t1) > 0.5:
            if strict and not installer_mode:
                sys.exit("[SECURITY] Timing anomaly detected")
            else:
                print("[WARN] Timing anomaly (installer mode)")

        return  # ✅ EXIT CLEANLY ON WINDOWS

    # --------------------------------------------------
    # Linux: FULL ANTI-DEBUG
    # --------------------------------------------------
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        PTRACE_TRACEME = 0
        PTRACE_DETACH = 17

        ret = libc.ptrace(PTRACE_TRACEME, 0, None, None)
        libc.ptrace(PTRACE_DETACH, 0, None, None)

        if ret != 0:
            sys.exit("[SECURITY] Debugger detected (ptrace)")

    except Exception:
        if strict:
            from .self_destruct import trigger_self_destruct
            trigger_self_destruct("Debugger probe failed")
        return

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
        if strict:
            from .self_destruct import trigger_self_destruct
            trigger_self_destruct("Debugger probe failed")

    for env in ["LD_PRELOAD", "LD_DEBUG"]:
        if env in os.environ:
            from .self_destruct import trigger_self_destruct
            trigger_self_destruct("Suspicious environment detected")

    t1 = time.time()
    for _ in range(1000000):
        pass
    t2 = time.time()

    if (t2 - t1) > 1.2:
        from .self_destruct import trigger_self_destruct
        trigger_self_destruct("Timing anomaly detected")
