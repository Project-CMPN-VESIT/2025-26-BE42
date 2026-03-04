import tkinter as tk
from tkinter import messagebox, scrolledtext
import threading
import sys
import io

# import your existing installer
import installer_core


class InstallerGUI:
    def __init__(self, root):
        self.root = root
        root.title("Federated Client Installer")
        root.geometry("700x450")
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
            height=18,
            state="disabled"
        )
        self.log.pack(padx=10, pady=10)

    def log_write(self, text):
        self.log.configure(state="normal")
        self.log.insert(tk.END, text)
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def start_install(self):
        otp = self.otp_entry.get().strip()
        server = self.server_entry.get().strip()
        if len(otp) < 6:
            messagebox.showerror("Error", "Invalid OTP")
            return

        if not server:
            messagebox.showerror("Error", "Server address required")
            return

        self.start_btn.config(state="disabled")
        threading.Thread(
            target=self.run_installer,
            args=(otp, server),
            daemon=True
        ).start()

    def run_installer(self, otp, server):
        print("INSTALLER CORE FILE:", installer_core.__file__)

        buffer = io.StringIO()
        sys.stdout = buffer
        sys.stderr = buffer

        try:
            installer_core.main(otp, server)

            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Success",
                    "Installation completed successfully"
                )
            )

        except SystemExit as e:
            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "Installer Failed",
                    str(e)
                )
            )

        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            self.root.after(0, lambda: self.log_write(buffer.getvalue()))


if __name__ == "__main__":
    root = tk.Tk()
    InstallerGUI(root)
    root.mainloop()
