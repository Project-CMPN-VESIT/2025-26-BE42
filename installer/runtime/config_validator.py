"""
config_validator.py — Phase 11: config schema validation.

Validates local_config.yaml before the pipeline starts so bad config
causes a clear error immediately rather than a cryptic crash mid-run.

Usage:
    from runtime.config_validator import validate_config
    validate_config(cfg)     # raises ConfigValidationError on problems
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger(__name__)


class ConfigValidationError(ValueError):
    """Raised when local_config.yaml fails validation."""


# ── Rules ─────────────────────────────────────────────────────────────────────

def _require(cfg: dict, *path: str) -> Any:
    """Traverse nested dict by path, raise ConfigValidationError if missing."""
    node = cfg
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise ConfigValidationError(
                f"Required config key missing: {' → '.join(path)}"
            )
        node = node[key]
    return node


def _warn_missing(cfg: dict, *path: str) -> bool:
    """Return False (and log a warning) if a path is missing."""
    node = cfg
    for key in path:
        if not isinstance(node, dict) or key not in node:
            log.warning("Config key missing (optional): %s", " → ".join(path))
            return False
        node = node[key]
    return True


def _require_file(path_str: str, label: str):
    """Raise ConfigValidationError if a referenced file does not exist."""
    p = Path(path_str).expanduser()
    if not p.exists():
        raise ConfigValidationError(
            f"Config references missing file [{label}]: {p}"
        )


# ── Main validator ────────────────────────────────────────────────────────────

def validate_config(cfg: dict) -> None:
    """
    Validate the loaded local_config.yaml dict.
    Raises ConfigValidationError with a descriptive message on any problem.
    """
    errors: List[str] = []

    def _check(fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except ConfigValidationError as e:
            errors.append(str(e))

    # ── storage ───────────────────────────────────────────────────────────────
    _check(_require, cfg, "storage")
    _check(_require, cfg, "storage", "root")

    storage_root = cfg.get("storage", {}).get("root", "")
    if not storage_root:
        errors.append("storage.root must not be empty")

    # ── mode ──────────────────────────────────────────────────────────────────
    valid_modes = {"interactive", "batch", "continuous", "session", "text"}
    mode = cfg.get("mode", "")
    if mode not in valid_modes:
        errors.append(
            f"Invalid mode '{mode}'. Must be one of: {sorted(valid_modes)}"
        )

    # ── ingest.video ─────────────────────────────────────────────────────────
    video_cfg = cfg.get("ingest", {}).get("video", {})
    if video_cfg.get("enabled", False):
        of_cfg = video_cfg.get("params", {}).get("openface", {})
        bin_path = of_cfg.get("binary_path", "")
        if bin_path:
            p = Path(bin_path).expanduser()
            if not p.exists():
                log.warning(
                    "OpenFace binary not found: %s "
                    "(OK if running on Linux without bundled binaries)", p
                )

    # ── ingest.audio ─────────────────────────────────────────────────────────
    audio_cfg = cfg.get("ingest", {}).get("audio", {})
    if audio_cfg.get("enabled", False):
        sr = audio_cfg.get("sr", 16000)
        if sr not in (8000, 16000, 22050, 44100, 48000):
            errors.append(f"ingest.audio.sr={sr} is unusual. Expected 8000/16000/22050/44100/48000.")

    # ── audio_pipe.features.egemaps ───────────────────────────────────────────
    egemaps_cfg = cfg.get("audio_pipe", {}).get("features", {}).get("egemaps", {})
    if egemaps_cfg.get("enabled", False):
        for key in ("opensmile_binary", "opensmile_config"):
            val = egemaps_cfg.get(key, "")
            if not val:
                errors.append(f"audio_pipe.features.egemaps.{key} must be set when egemaps.enabled=true")
            else:
                p = Path(val).expanduser()
                if not p.exists():
                    log.warning("egemaps.%s not found: %s", key, p)

    # ── text_pipe ─────────────────────────────────────────────────────────────
    text_pipe = cfg.get("text_pipe", {})
    backend = text_pipe.get("asr_backend", "whisper")
    if backend not in ("whisper", "hf"):
        errors.append(
            f"text_pipe.asr_backend='{backend}' invalid. Must be 'whisper' or 'hf'."
        )
    if not text_pipe.get("asr_model"):
        errors.append("text_pipe.asr_model must be set (e.g. 'small')")
    if not text_pipe.get("asr_hf_model"):
        log.warning("text_pipe.asr_hf_model not set — HF fallback ASR unavailable")

    # ── limits ────────────────────────────────────────────────────────────────
    limits = cfg.get("limits", {})
    max_sessions = limits.get("max_concurrent_sessions", 4)
    if not isinstance(max_sessions, int) or max_sessions < 1:
        errors.append("limits.max_concurrent_sessions must be a positive integer")

    # ── Report ────────────────────────────────────────────────────────────────
    if errors:
        msg = "\n".join(f"  • {e}" for e in errors)
        raise ConfigValidationError(
            f"local_config.yaml has {len(errors)} validation error(s):\n{msg}"
        )

    log.info("[config_validator] Configuration validated OK (mode=%s)", mode)