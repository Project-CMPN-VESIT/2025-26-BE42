import tkinter as tk
from tkinter import messagebox, scrolledtext
import threading
import sys
import io
import traceback

import installer_core


class InstallerGUI:
    def __init__(self, root):
        self.root = root
        root.title("Federated Client Installer")
        root.geometry("700x500")
        root.resizable(False, False)

        tk.Label(
            root,
            text="Federated Learning Client Installer",
            font=("Segoe UI", 16, "bold")
        ).pack(pady=10)

        tk.Label(root, text="Enrollment OTP").pack()
        self.otp_entry = tk.Entry(root, width=40)
        self.otp_entry.pack(pady=5)

        tk.Label(root, text="Server Address (host:port)").pack()
        self.server_entry = tk.Entry(root, width=40)
        self.server_entry.pack(pady=5)

        self.start_btn = tk.Button(
            root,
            text="Install",
            command=self.start_install,
            width=20,
            bg="#2ecc71",
            fg="white"
        )
        self.start_btn.pack(pady=10)

        self.log = scrolledtext.ScrolledText(
            root,
            width=85,
            height=20,
            state="disabled"
        )
        self.log.pack(padx=10, pady=10)

    def log_write(self, text):
        self.log.configure(state="normal")
        self.log.insert(tk.END, text)
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def start_install(self):
        otp    = self.otp_entry.get().strip()
        server = self.server_entry.get().strip()

        if len(otp) < 6:
            messagebox.showerror("Error", "OTP must be at least 6 digits")
            return
        if not server:
            messagebox.showerror("Error", "Server address is required (e.g. 192.168.1.7:50051)")
            return

        self.start_btn.config(state="disabled")
        self.log_write(f"[INFO] Starting installation...\n"
                       f"  OTP: {otp}\n"
                       f"  Server: {server}\n\n")

        threading.Thread(
            target=self.run_installer,
            args=(otp, server),
            daemon=True
        ).start()

    def run_installer(self, otp, server):
        buffer = io.StringIO()

        # Redirect stdout/stderr so print() output appears in the log widget.
        # logging goes to the log FILE (not stdout), so those entries are
        # visible in ~/.federated/logs/installer.log for diagnosis.
        sys.stdout = buffer
        sys.stderr = buffer

        success = False
        error_msg = ""

        try:
            installer_core.main(otp, server)
            success = True

        except SystemExit as e:
            # installer_core calls sys.exit() with a human-readable message
            error_msg = str(e)

        except Exception as e:
            # Catch everything else (FutureTimeoutError, RpcError, etc.)
            # and produce a full traceback so the problem is diagnosable.
            error_msg = (
                f"{type(e).__name__}: {e}\n\n"
                f"Full traceback:\n{traceback.format_exc()}"
            )

        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            captured = buffer.getvalue()

        # Schedule all UI updates back on the main thread
        if success:
            self.root.after(0, lambda: self._on_success(captured))
        else:
            self.root.after(0, lambda: self._on_failure(captured, error_msg))

    def _on_success(self, captured):
        self.log_write(captured)
        self.log_write("\n[OK] Installation completed successfully!\n")
        messagebox.showinfo("Success", "Installation completed successfully")
        self.start_btn.config(state="normal")

    def _on_failure(self, captured, error_msg):
        self.log_write(captured)
        self.log_write(f"\n[ERROR] Installation failed:\n{error_msg}\n")
        self.log_write(
            "\nCheck the full log at:\n"
            "  C:\\Users\\<you>\\.federated\\logs\\installer.log\n"
        )
        messagebox.showerror(
            "Installer Failed",
            f"{error_msg}\n\nSee log: ~/.federated/logs/installer.log"
        )
        self.start_btn.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    InstallerGUI(root)
    root.mainloop()