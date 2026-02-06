mod config;
mod crypto;
mod errors;
mod grpc;
mod identity;
mod ledger;
mod pubsub;
mod receipts;
mod round;
mod state;
mod otp;

use crate::config::Config;
use crate::state::OrchestratorState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let cfg = Config::load("config/orchestrator.toml")?;
    let state = OrchestratorState::new();

    pubsub::start(state.clone());
    grpc::server::serve(cfg, state).await?;

    Ok(())
}
