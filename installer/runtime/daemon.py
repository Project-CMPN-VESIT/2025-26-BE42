import time
from runtime.idle import wait_until_idle
from runtime.pipeline import run_pipeline

UPLOAD_INTERVAL = 60 * 60  # 1 hour

def daemon_loop(stub, device_id, master_secret):
    while True:
        try:
            wait_until_idle()
            run_pipeline(stub, device_id, master_secret)
        except Exception:
            # Silent failure; retry later
            pass

        time.sleep(UPLOAD_INTERVAL)
