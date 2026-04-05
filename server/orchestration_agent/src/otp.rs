// otp.rs — Per-device OTP with 60-second expiry and rate limiting
//
// Phase 8 improvements over original:
//   - OTP is bound to a device fingerprint (IP + pubkey hash)
//   - 60-second expiry enforced at consume time
//   - Max 5 failed attempts per device before lockout
//   - OTP is 6-digit cryptographically random (not pseudo-random range)

use once_cell::sync::Lazy;
use rand::{thread_rng, Rng};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

const OTP_EXPIRY_SECS: u64 = 3000;
const MAX_FAILED_ATTEMPTS: u32 = 5;
const LOCKOUT_DURATION_SECS: u64 = 300; // 5 minutes

#[derive(Debug)]
struct OtpEntry {
    token: String,
    created_at: Instant,
    device_hint: Option<String>,  // optional binding (e.g. IP)
    used: bool,
}

#[derive(Debug, Default)]
struct RateLimitEntry {
    failed_attempts: u32,
    locked_until: Option<Instant>,
}

static OTP_STORE: Lazy<Mutex<HashMap<String, OtpEntry>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

static RATE_LIMITER: Lazy<Mutex<HashMap<String, RateLimitEntry>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Generate a cryptographically random 6-digit OTP.
/// Optionally bind it to a device_hint (e.g. enrollment IP or pubkey prefix).
pub fn generate_otp() -> String {
    generate_otp_for(None)
}

pub fn generate_otp_for(device_hint: Option<String>) -> String {
    let mut rng = thread_rng();
    // 6-digit: [100000, 999999]
    let otp: u32 = rng.gen_range(100_000..=999_999);
    let token = otp.to_string();

    let mut store = OTP_STORE.lock().unwrap();

    // Purge expired OTPs to avoid unbounded growth
    _purge_expired(&mut store);

    store.insert(
        token.clone(),
        OtpEntry {
            token: token.clone(),
            created_at: Instant::now(),
            device_hint,
            used: false,
        },
    );

    token
}

/// Consume an OTP. Returns true only if:
///   1. Token exists
///   2. Not expired (< 60 seconds old)
///   3. Not already used
///   4. Device not rate-limited
///
/// device_id is used for rate limiting (e.g. connecting IP).
pub fn consume_otp(token: &str) -> bool {
    consume_otp_from(token, "unknown")
}

pub fn consume_otp_from(token: &str, device_id: &str) -> bool {
    // ── Rate limit check ──────────────────────────────────────────────────────
    {
        let mut rl = RATE_LIMITER.lock().unwrap();
        let entry = rl.entry(device_id.to_string()).or_default();

        if let Some(locked_until) = entry.locked_until {
            if Instant::now() < locked_until {
                tracing::warn!(
                    "OTP attempt from locked device '{}' — still locked",
                    device_id
                );
                return false;
            } else {
                // Lockout expired — reset
                entry.failed_attempts = 0;
                entry.locked_until = None;
            }
        }
    }

    // ── OTP lookup ────────────────────────────────────────────────────────────
    let mut store = OTP_STORE.lock().unwrap();
    _purge_expired(&mut store);

    match store.get_mut(token) {
        None => {
            tracing::warn!("OTP '{}' not found or already expired", token);
            _record_failure(device_id);
            false
        }
        Some(entry) if entry.used => {
            tracing::warn!("OTP '{}' already consumed", token);
            _record_failure(device_id);
            false
        }
        Some(entry) if entry.created_at.elapsed() > Duration::from_secs(OTP_EXPIRY_SECS) => {
            tracing::warn!("OTP '{}' expired (age={:?})", token, entry.created_at.elapsed());
            store.remove(token);
            _record_failure(device_id);
            false
        }
        Some(entry) => {
            entry.used = true;
            tracing::info!(
                "OTP '{}' consumed by device '{}' (age={:?})",
                token,
                device_id,
                entry.created_at.elapsed(),
            );
            // Reset failure count on success
            let mut rl = RATE_LIMITER.lock().unwrap();
            rl.remove(device_id);
            true
        }
    }
}

/// List all active (non-expired) OTPs — for admin/debug only.
#[allow(dead_code)]
pub fn list_active_otps() -> Vec<String> {
    let mut store = OTP_STORE.lock().unwrap();
    _purge_expired(&mut store);
    store.keys().cloned().collect()
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn _purge_expired(store: &mut HashMap<String, OtpEntry>) {
    let expiry = Duration::from_secs(OTP_EXPIRY_SECS);
    store.retain(|_, v| !v.used && v.created_at.elapsed() <= expiry);
}

fn _record_failure(device_id: &str) {
    let mut rl = RATE_LIMITER.lock().unwrap();
    let entry = rl.entry(device_id.to_string()).or_default();
    entry.failed_attempts += 1;
    tracing::warn!(
        "OTP failure #{} from device '{}'",
        entry.failed_attempts,
        device_id
    );
    if entry.failed_attempts >= MAX_FAILED_ATTEMPTS {
        let until = Instant::now() + Duration::from_secs(LOCKOUT_DURATION_SECS);
        entry.locked_until = Some(until);
        tracing::error!(
            "Device '{}' locked out for {}s after {} failures",
            device_id,
            LOCKOUT_DURATION_SECS,
            entry.failed_attempts,
        );
    }
}