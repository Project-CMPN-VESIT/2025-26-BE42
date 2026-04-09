"""
installer_gui.py — Two-phase enrollment GUI.

Phase 1 (Software Setup):
  ┌─────────────────────────────────────────────┐
  │ Server address: [192.168.1.7:50051         ] │
  │ [Install Software]                           │
  │ ┌── progress log ──────────────────────────┐ │
  │ │ ...                                      │ │
  │ └──────────────────────────────────────────┘ │
  └─────────────────────────────────────────────┘

Phase 2 (Enrollment):  appears AFTER Phase 1 succeeds
  ┌─────────────────────────────────────────────┐
  │ ✅ Software installed!                       │
  │ Device ID: a1b2c3d4e5f6a7b8                 │
  │ Your administrator has been notified.        │
  │ Enter the OTP they give you:                 │
  │ OTP: [      ]   [Complete Enrollment]        │
  └─────────────────────────────────────────────┘
"""

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
        root.geometry("700x620")
        root.resizable(False, False)

        # ── State ─────────────────────────────────────────────────────────────
        self._device_pubkey: bytes = b""
        self._device_fingerprint: str = ""
        self._server_addr: str = ""

        # ── Title ─────────────────────────────────────────────────────────────
        tk.Label(
            root,
            text="Federated Learning Client Installer",
            font=("Segoe UI", 15, "bold"),
        ).pack(pady=10)

        # ── Phase 1 frame ──────────────────────────────────────────────────────
        self._phase1_frame = tk.LabelFrame(
            root, text="Phase 1 — Software Setup", padx=10, pady=6
        )
        self._phase1_frame.pack(fill="x", padx=15, pady=(0, 4))

        tk.Label(self._phase1_frame, text="Server address (host:port)").pack(anchor="w")
        self.server_entry = tk.Entry(self._phase1_frame, width=50)
        self.server_entry.pack(pady=(0, 6), anchor="w")

        self.install_btn = tk.Button(
            self._phase1_frame,
            text="Install Software",
            command=self._start_phase1,
            width=20,
            bg="#2ecc71",
            fg="white",
        )
        self.install_btn.pack(anchor="w")

        # ── Log ───────────────────────────────────────────────────────────────
        tk.Label(root, text="Installation log:", anchor="w").pack(fill="x", padx=15)
        self.log = scrolledtext.ScrolledText(root, width=84, height=18, state="disabled")
        self.log.pack(padx=15, pady=(0, 6))

        # ── Phase 2 frame (hidden until Phase 1 succeeds) ─────────────────────
        self._phase2_frame = tk.LabelFrame(
            root, text="Phase 2 — Enrollment", padx=10, pady=6
        )
        # Not packed yet — shown dynamically

        self._fingerprint_var = tk.StringVar(value="")
        tk.Label(
            self._phase2_frame,
            text="✅ Software installed. Give your administrator this Device ID:",
        ).pack(anchor="w")

        fp_row = tk.Frame(self._phase2_frame)
        fp_row.pack(fill="x", pady=(0, 6))
        self._fp_entry = tk.Entry(
            fp_row,
            textvariable=self._fingerprint_var,
            state="readonly",
            width=36,
            font=("Courier New", 11),
        )
        self._fp_entry.pack(side="left")
        tk.Button(fp_row, text="Copy", command=self._copy_fingerprint).pack(
            side="left", padx=6
        )

        tk.Label(
            self._phase2_frame,
            text="Your administrator has been notified. Enter the OTP they provide:",
        ).pack(anchor="w")

        otp_row = tk.Frame(self._phase2_frame)
        otp_row.pack(fill="x", pady=(0, 4))
        self.otp_entry = tk.Entry(otp_row, width=18, font=("Courier New", 12))
        self.otp_entry.pack(side="left")

        self.enroll_btn = tk.Button(
            otp_row,
            text="Complete Enrollment",
            command=self._start_phase2,
            width=22,
            bg="#2980b9",
            fg="white",
        )
        self.enroll_btn.pack(side="left", padx=8)

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, text: str):
        self.log.configure(state="normal")
        self.log.insert(tk.END, text)
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def _copy_fingerprint(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self._fingerprint_var.get())

    # ── Phase 1 ───────────────────────────────────────────────────────────────

    def _start_phase1(self):
        server = self.server_entry.get().strip()
        if not server:
            messagebox.showerror("Error", "Server address is required (e.g. 192.168.1.7:50051)")
            return

        self._server_addr = server
        self.install_btn.config(state="disabled")
        self._log(f"[INFO] Starting software setup…\n  Server: {server}\n\n")

        threading.Thread(target=self._run_phase1, daemon=True).start()

    def _run_phase1(self):
        buf = io.StringIO()
        sys.stdout = buf
        sys.stderr = buf

        pubkey = None
        fingerprint = None
        error_msg = ""

        try:
            pubkey = installer_core.setup_software(self._server_addr)
            fingerprint = installer_core.request_enrollment_otp(pubkey, self._server_addr)
        except SystemExit as e:
            error_msg = str(e)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            captured = buf.getvalue()

        if pubkey and fingerprint and not error_msg:
            self.root.after(0, lambda: self._on_phase1_success(captured, pubkey, fingerprint))
        else:
            self.root.after(0, lambda: self._on_phase1_failure(captured, error_msg))

    def _on_phase1_success(self, captured: str, pubkey: bytes, fingerprint: str):
        self._log(captured)
        self._log("\n[OK] Software installed. Waiting for OTP from administrator.\n")
        self._device_pubkey = pubkey
        self._device_fingerprint = fingerprint
        self._fingerprint_var.set(fingerprint)
        # Show Phase 2 panel
        self._phase2_frame.pack(fill="x", padx=15, pady=(0, 10))
        self.otp_entry.focus_set()

    def _on_phase1_failure(self, captured: str, error_msg: str):
        self._log(captured)
        self._log(f"\n[ERROR] Setup failed:\n{error_msg}\n")
        self._log(f"\nSee full log: {installer_core.LOG_FILE}\n")
        messagebox.showerror(
            "Setup Failed",
            f"{error_msg}\n\nSee: {installer_core.LOG_FILE}",
        )
        self.install_btn.config(state="normal")

    # ── Phase 2 ───────────────────────────────────────────────────────────────

    def _start_phase2(self):
        otp = self.otp_entry.get().strip()
        if len(otp) < 6:
            messagebox.showerror("Error", "OTP must be at least 6 characters")
            return
        if not self._device_pubkey:
            messagebox.showerror("Error", "Software setup must complete first")
            return

        self.enroll_btn.config(state="disabled")
        self._log(f"\n[INFO] Completing enrollment with OTP…\n")

        threading.Thread(target=self._run_phase2, args=(otp,), daemon=True).start()

    def _run_phase2(self, otp: str):
        buf = io.StringIO()
        sys.stdout = buf
        sys.stderr = buf

        success = False
        error_msg = ""

        try:
            installer_core.finalize_install(
                self._device_pubkey, otp, self._server_addr
            )
            success = True
        except SystemExit as e:
            error_msg = str(e)
        except PermissionError as e:
            error_msg = str(e)   # OTP rejected — clear message
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            captured = buf.getvalue()

        if success:
            self.root.after(0, lambda: self._on_phase2_success(captured))
        else:
            self.root.after(0, lambda: self._on_phase2_failure(captured, error_msg))

    def _on_phase2_success(self, captured: str):
        self._log(captured)
        self._log("\n[OK] Enrollment complete! Client daemon registered.\n")
        messagebox.showinfo(
            "Success",
            "Enrollment complete!\n\nThe federated learning client will start automatically on next login.",
        )
        self.enroll_btn.config(state="normal")

    def _on_phase2_failure(self, captured: str, error_msg: str):
        self._log(captured)
        self._log(f"\n[ERROR] Enrollment failed:\n{error_msg}\n")
        messagebox.showerror(
            "Enrollment Failed",
            f"{error_msg}\n\nIf the OTP expired, ask your administrator for a new one.",
        )
        self.enroll_btn.config(state="normal")
        # Allow OTP re-entry; no need to redo Phase 1


if __name__ == "__main__":
    root = tk.Tk()
    InstallerGUI(root)
    root.mainloop()