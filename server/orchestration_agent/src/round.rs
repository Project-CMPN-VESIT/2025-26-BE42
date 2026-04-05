// round.rs — RoundState derives Debug so format!("{:?}", round.state) compiles.

use crate::state::DeviceId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoundState {
    Open,
    Collecting,
    Aggregating,
    Complete,
}

#[derive(Clone, Debug)]
pub struct UpdateMeta {
    pub device_id: DeviceId,
    pub enc_uri:   String,
    pub scheme:    String,
    pub nonce:     Option<String>,
}

pub struct Round {
    pub id:                  u64,
    pub model_version:       String,
    pub epsilon_max:         f64,
    pub upload_uri:          String,
    pub state:               RoundState,
    pub updates:             Vec<UpdateMeta>,
    pub aggregation_receipt: Option<AggregationReceipt>,
    pub epsilon_spent:       f64,
}

#[derive(Clone, Debug)]
pub struct AggregationReceipt {
    pub round_id:         u64,
    pub num_updates:      usize,
    pub aggregation_mode: String,
    pub aggregated_uri:   String,
}