use std::collections::HashSet;
use std::sync::Mutex;
use once_cell::sync::Lazy;
use rand::{thread_rng, Rng};

static OTP_STORE: Lazy<Mutex<HashSet<String>>> =
    Lazy::new(|| Mutex::new(HashSet::new()));

pub fn generate_otp() -> String {
    let mut rng = thread_rng();
    let otp: u32 = rng.gen_range(100000..999999);

    let token = otp.to_string();

    OTP_STORE
        .lock()
        .unwrap()
        .insert(token.clone());

    token
}

pub fn consume_otp(token: &str) -> bool {
    OTP_STORE
        .lock()
        .unwrap()
        .remove(token)
}