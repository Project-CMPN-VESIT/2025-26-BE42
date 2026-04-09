use dashmap::DashMap;
use std::sync::Arc;
use crate::round::{Round, RoundState};

pub type DeviceId = Vec<u8>;

/// device_fingerprint -> (device_pubkey_bytes, csr_bytes)
pub type PendingEnrollment = (Vec<u8>, Vec<u8>);

pub struct OrchestratorState {
    /// Registered devices: device_id (SHA-256 of pubkey) → pubkey bytes
    pub devices: DashMap<DeviceId, Vec<u8>>,
    /// Active federated rounds
    pub rounds: DashMap<u64, Round>,
    /// Legacy: single enrollment tokens (kept for backward compat)
    pub enrollment_tokens: DashMap<String, ()>,
    /// Multi-device pending enrollments: fingerprint (8-byte hex) → (pubkey, csr)
    pub pending_enrollments: DashMap<String, PendingEnrollment>,
}

impl OrchestratorState {
    pub fn new() -> Arc<Self> {
        let s = Arc::new(Self {
            devices: DashMap::new(),
            rounds: DashMap::new(),
            enrollment_tokens: DashMap::new(),
            pending_enrollments: DashMap::new(),
        });

        s.rounds.insert(1, Round {
            id: 1,
            model_version: "v1".into(),
            epsilon_max: 1.0,
            upload_uri: "objectstore://round-1".into(),
            state: RoundState::Collecting,
            updates: Vec::new(),
            aggregation_receipt: None,
            epsilon_spent: 0.0,
        });

        s
    }
}