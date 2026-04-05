// main.rs — Fixed: removed unused `use crate::otp::generate_otp` import.

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
    // ── 1. Init logging ───────────────────────────────────────────────────────
    tracing_subscriber::fmt::init();

    // ── 2. Load config ────────────────────────────────────────────────────────
    let cfg = Config::load("config/orchestrator.toml")?;

    // ── 3. Create shared orchestrator state ───────────────────────────────────
    let state = OrchestratorState::new();

    // ── 4. Bootstrap enrollment OTP ───────────────────────────────────────────
    // FIX: was importing `use crate::otp::generate_otp` but calling it via the
    // full path anyway — removed the redundant import to silence the warning.
    let otp = crate::otp::generate_otp();

    // Keep the enrollment_tokens map in sync for any code that reads it directly.
    state.enrollment_tokens.insert(otp.clone(), ());

    println!("[ENROLLMENT] OTP: {} (valid for 50 minutes)", otp);
    tracing::info!("Enrollment OTP generated — valid 50 minutes");

    // ── 5. Start background systems ───────────────────────────────────────────
    pubsub::start(state.clone());

    // ── 6. Start gRPC server (blocks until shutdown) ──────────────────────────
    grpc::server::serve(cfg, state).await?;

    Ok(())
}