use std::collections::HashMap;
use std::sync::Mutex;
use sha2::{Sha256, Digest};

lazy_static::lazy_static! {
    static ref OTP_STORE: Mutex<HashMap<Vec<u8>, bool>> = Mutex::new(HashMap::new());
}

pub fn add_otp(token: &str) {
    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    let hash = hasher.finalize().to_vec();

    OTP_STORE.lock().unwrap().insert(hash, false);
}

pub fn consume_otp(token: &str) -> bool {
    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    let hash = hasher.finalize().to_vec();

    let mut store = OTP_STORE.lock().unwrap();
    match store.get_mut(&hash) {
        Some(used) if !*used => {
            *used = true;
            true
        }
        _ => false,
    }
}
