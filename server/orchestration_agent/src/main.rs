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

use crate::otp::generate_otp;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // --------------------------------------------------
    // 1. Init logging FIRST
    // --------------------------------------------------
    tracing_subscriber::fmt::init();

    // --------------------------------------------------
    // 2. Load config
    // --------------------------------------------------
    let cfg = Config::load("config/orchestrator.toml")?;

    // --------------------------------------------------
    // 3. Create shared orchestrator state
    // --------------------------------------------------
    let state = OrchestratorState::new();

    // --------------------------------------------------
    // 4. DEV ONLY: bootstrap enrollment OTP
    // --------------------------------------------------
    let otp = crate::otp::generate_otp();
    state.enrollment_tokens.insert(otp.to_string(), ());
    println!("[DEV] Enrollment OTP enabled: {}", otp);

    // --------------------------------------------------
    // 5. Start background systems
    // --------------------------------------------------
    pubsub::start(state.clone());

    // --------------------------------------------------
    // 6. Start gRPC server (blocks)
    // --------------------------------------------------
    grpc::server::serve(cfg, state).await?;

    Ok(())
}
