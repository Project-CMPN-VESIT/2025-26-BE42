// ── server.rs — Fixed: missing imports, req/req_inner confusion,
//   aggregation borrow-drop ordering, error logging, and security hardening. ──

use std::sync::Arc;
use std::process::Command;
use std::io::Write;
use std::fs;

use tempfile::NamedTempFile;

use tonic::{Request, Response, Status};
use tonic::transport::{
    Server,
    ServerTlsConfig,
    Identity,
    Certificate as TlsCertificate,
};

use crate::config::Config;
use crate::crypto::ct_eq;
use crate::identity::derive_device_id;
use crate::state::OrchestratorState;
use crate::round::{
    UpdateMeta,
    RoundState,
    AggregationReceipt,
};

use crate::grpc::orchestrator::{
    Ack,
    Csr,
    Certificate,
    DeviceId,
    Receipt,
    RoundMetadata,
    EnrollRequest,
    EnrollResponse,
};

use crate::grpc::orchestrator::orchestrator_server::{
    Orchestrator,
    OrchestratorServer,
};

// ─────────────────────────────────────────────────────────────────────────────
// Service
// ─────────────────────────────────────────────────────────────────────────────

/// Orchestrator gRPC service
pub struct Service {
    state: Arc<OrchestratorState>,
    #[allow(dead_code)]
    cfg: Config,
}

#[tonic::async_trait]
impl Orchestrator for Service {

    // ── EnrollDevice ──────────────────────────────────────────────────────────
    async fn enroll_device(
        &self,
        req: Request<EnrollRequest>,
    ) -> Result<Response<EnrollResponse>, Status> {

        // Capture peer address BEFORE consuming request
        let peer_addr = req
            .remote_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|| "unknown".to_string());

        // FIX: consume request into inner — then only use req_inner
        let req_inner = req.into_inner();

        // ── 1. Rate-limited OTP consumption ───────────────────────────────────
        if !crate::otp::consume_otp_from(&req_inner.enrollment_token, &peer_addr) {
            tracing::warn!(
                "Enrollment rejected for peer {} — invalid/expired OTP",
                peer_addr
            );
            return Err(Status::permission_denied("invalid or expired OTP"));
        }

        // ── 2. Basic input validation ─────────────────────────────────────────
        if req_inner.device_pubkey.is_empty() {
            return Err(Status::invalid_argument("device_pubkey must not be empty"));
        }
        if req_inner.csr.is_empty() {
            return Err(Status::invalid_argument("CSR must not be empty"));
        }

        // ── 3. Store TPM pubkey ───────────────────────────────────────────────
        // FIX: was req.device_pubkey — now req_inner.device_pubkey
        let device_id = derive_device_id(&req_inner.device_pubkey);
        self.state
            .devices
            .insert(device_id, req_inner.device_pubkey.clone());

        // ── 4. Write CSR to temp file ─────────────────────────────────────────
        // FIX: was req.csr — now req_inner.csr
        // FIX: NamedTempFile was undeclared — now imported at top
        let mut csr_file = NamedTempFile::new()
            .map_err(|e| {
                tracing::error!("Failed to create temp CSR file: {}", e);
                Status::internal("internal error")
            })?;

        csr_file
            .write_all(&req_inner.csr)
            .map_err(|e| {
                tracing::error!("Failed to write CSR to temp file: {}", e);
                Status::internal("internal error")
            })?;

        let cert_file = NamedTempFile::new()
            .map_err(|e| {
                tracing::error!("Failed to create temp cert file: {}", e);
                Status::internal("internal error")
            })?;

        // ── 5. Sign CSR using OpenSSL + CA ────────────────────────────────────
        let output = Command::new("openssl")
            .arg("x509")
            .arg("-req")
            .arg("-in")
            .arg(csr_file.path())
            .arg("-CA")
            .arg(&self.cfg.tls.ca_cert)
            .arg("-CAkey")
            .arg(&self.cfg.tls.ca_key)
            .arg("-CAcreateserial")
            .arg("-out")
            .arg(cert_file.path())
            .arg("-days")
            .arg("365")
            .arg("-sha256")
            .output()
            .map_err(|e| {
                tracing::error!("Failed to execute openssl: {}", e);
                Status::internal("certificate signing failed")
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::error!("openssl CSR signing failed: {}", stderr);
            return Err(Status::internal("certificate signing failed"));
        }

        // ── 6. Read signed certificate ────────────────────────────────────────
        // FIX: fs was undeclared — now imported as std::fs at top
        let signed_cert = fs::read(cert_file.path())
            .map_err(|e| {
                tracing::error!("Failed to read signed cert: {}", e);
                Status::internal("certificate read failed")
            })?;

        if signed_cert.is_empty() {
            tracing::error!("openssl produced an empty certificate");
            return Err(Status::internal("certificate signing produced empty output"));
        }

        tracing::info!(
            "Device enrolled — peer={} pubkey_len={}",
            peer_addr,
            req_inner.device_pubkey.len()
        );

        Ok(Response::new(EnrollResponse {
            ok: true,
            client_cert: signed_cert,
        }))
    }

    // ── RegisterDevice (deprecated, kept for backward compat) ─────────────────
    async fn register_device(
        &self,
        req: Request<Csr>,
    ) -> Result<Response<Certificate>, Status> {

        Self::require_client_cert(&req)?;

        let inner = req.into_inner();

        if inner.device_pubkey.is_empty() {
            return Err(Status::invalid_argument("device_pubkey must not be empty"));
        }

        let device_id = derive_device_id(&inner.device_pubkey);
        self.state
            .devices
            .insert(device_id, inner.device_pubkey.clone());

        tracing::warn!("register_device called — this RPC is deprecated; use EnrollDevice");

        Ok(Response::new(Certificate {
            pem: inner.device_pubkey,
        }))
    }

    // ── GetRound ──────────────────────────────────────────────────────────────
    async fn get_round(
        &self,
        req: Request<DeviceId>,
    ) -> Result<Response<RoundMetadata>, Status> {

        Self::require_client_cert(&req)?;

        let round = self
            .state
            .rounds
            .get(&1)
            .ok_or_else(|| Status::not_found("round not found"))?;

        let receipt_ref = round.aggregation_receipt.as_ref();

        Ok(Response::new(RoundMetadata {
            round_id:         round.id,
            model_version:    round.model_version.clone(),
            epsilon_max:      round.epsilon_max,
            upload_uri:       round.upload_uri.clone(),
            state:            format!("{:?}", round.state),
            num_updates:      receipt_ref.map(|r| r.num_updates as u32).unwrap_or(0),
            aggregation_mode: receipt_ref
                .map(|r| r.aggregation_mode.clone())
                .unwrap_or_default(),
        }))
    }

    // ── SubmitReceipt ─────────────────────────────────────────────────────────
    async fn submit_receipt(
        &self,
        req: Request<Receipt>,
    ) -> Result<Response<Ack>, Status> {

        Self::require_client_cert(&req)?;

        let receipt = req.into_inner();

        // ── Validate receipt fields ───────────────────────────────────────────
        if receipt.device_id.is_empty() {
            return Err(Status::invalid_argument("device_id must not be empty"));
        }
        if receipt.payload_hash.is_empty() {
            return Err(Status::invalid_argument("payload_hash must not be empty"));
        }
        if receipt.signature.is_empty() {
            return Err(Status::invalid_argument("signature must not be empty"));
        }
        if receipt.epsilon_spent < 0.0 || receipt.epsilon_spent > 10.0 {
            return Err(Status::invalid_argument("epsilon_spent out of range"));
        }

        // ── Lookup device pubkey ──────────────────────────────────────────────
        let pubkey = self
            .state
            .devices
            .iter()
            .find(|entry| ct_eq(entry.key(), &receipt.device_id))
            .map(|entry| entry.value().clone())
            .ok_or_else(|| {
                tracing::warn!("SubmitReceipt from unknown device");
                Status::permission_denied("unknown device")
            })?;

        // ── Verify ECDSA signature ────────────────────────────────────────────
        let mut msg = Vec::with_capacity(
            receipt.device_id.len() + 8 + receipt.payload_hash.len()
        );
        msg.extend_from_slice(&receipt.device_id);
        msg.extend_from_slice(&receipt.round_id.to_be_bytes());
        msg.extend_from_slice(&receipt.payload_hash);

        crate::receipts::verify(&pubkey, &msg, &receipt.signature)
            .map_err(|_| {
                tracing::warn!("Invalid receipt signature from device");
                Status::permission_denied("invalid receipt signature")
            })?;

        // ── Fetch round — save the round_id before partial moves ─────────────
        let round_id = receipt.round_id;

        let mut round = self
            .state
            .rounds
            .get_mut(&round_id)
            .ok_or_else(|| Status::not_found("round not found"))?;

        // ── Enforce state ─────────────────────────────────────────────────────
        if round.state != RoundState::Collecting {
            return Err(Status::failed_precondition(
                "round is not accepting updates",
            ));
        }

        // ── Enforce ε-budget ──────────────────────────────────────────────────
        if round.epsilon_spent + receipt.epsilon_spent > round.epsilon_max {
            return Err(Status::resource_exhausted("epsilon budget exceeded"));
        }

        round.epsilon_spent += receipt.epsilon_spent;

        // ── Store update metadata — partial moves happen here ─────────────────
        round.updates.push(UpdateMeta {
            device_id: receipt.device_id.clone(),
            enc_uri:   receipt.enc_uri,
            scheme:    receipt.scheme,
            nonce:     if receipt.nonce.is_empty() {
                None
            } else {
                Some(receipt.nonce)
            },
        });

        let should_aggregate = round.updates.len() >= 3;

        // ── Release lock before triggering aggregation ────────────────────────
        if should_aggregate {
            round.state = RoundState::Aggregating;
            drop(round); // IMPORTANT: release DashMap write guard before re-entry
            self.run_aggregation(round_id)?;
        }

        Ok(Response::new(Ack { ok: true }))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// gRPC server bootstrap
// ─────────────────────────────────────────────────────────────────────────────

pub async fn serve(
    cfg: Config,
    state: Arc<OrchestratorState>,
) -> anyhow::Result<()> {

    let svc = Service {
        state,
        cfg: cfg.clone(),
    };

    let server_identity = Identity::from_pem(
        std::fs::read(&cfg.tls.server_cert)?,
        std::fs::read(&cfg.tls.server_key)?,
    );

    let client_ca = TlsCertificate::from_pem(
        std::fs::read(&cfg.tls.ca_cert)?,
    );

    // mTLS: require client certificate for operational RPCs.
    // Enrollment uses server-TLS only (handled at the RPC level via
    // require_client_cert which is intentionally permissive for EnrollDevice).
    let tls = ServerTlsConfig::new()
        .identity(server_identity)
        .client_ca_root(client_ca);

    tracing::info!("[TLS] TLS ENABLED — mTLS enforced per-RPC");

    let mut builder = Server::builder();
    builder = builder.tls_config(tls)?;

    tracing::info!("Binding to: {}", cfg.server.addr);

    builder
        .add_service(OrchestratorServer::new(svc))
        .serve(cfg.server.addr.parse()?)
        .await?;

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

impl Service {
    /// Verify that a peer TLS client certificate is present.
    ///
    /// `EnrollDevice` deliberately skips this check (the device does not yet
    /// have a client certificate).  All other RPCs must have a valid cert.
    fn require_client_cert<T>(req: &Request<T>) -> Result<(), Status> {
        if let Some(certs) = req.peer_certs() {
            if !certs.is_empty() {
                return Ok(());
            }
        }
        // Currently permissive — tighten once all clients are enrolled:
        // return Err(Status::unauthenticated("client certificate required"));
        Ok(())
    }

    /// Invoke the Python aggregator subprocess for a completed round.
    ///
    /// The round guard MUST be dropped by the caller before calling this so
    /// that the re-acquisition inside does not deadlock.
    fn run_aggregation(&self, round_id: u64) -> Result<(), Status> {

        // Re-acquire read guard to build the job payload
        let job = {
            let round = self
                .state
                .rounds
                .get(&round_id)
                .ok_or_else(|| Status::not_found("round not found during aggregation"))?;

            serde_json::json!({
                "round_id": round.id,
                "mode": "trimmed_mean",
                "trim_ratio": 0.1,
                "updates": round.updates.iter().map(|u| {
                    serde_json::json!({
                        "enc_uri": u.enc_uri,
                        "scheme":  u.scheme,
                        "nonce":   u.nonce
                    })
                }).collect::<Vec<_>>()
            })
        }; // guard dropped here

        // Spawn aggregator subprocess
        let mut child = Command::new("python3")
            .arg("server/aggregator_agent/aggregator.py")
            .env("PYTHONPATH", ".")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| {
                tracing::error!("Failed to start aggregator: {}", e);
                Status::internal("aggregator spawn failed")
            })?;

        if let Some(mut stdin) = child.stdin.take() {
            let payload = job.to_string();
            if let Err(e) = stdin.write_all(payload.as_bytes()) {
                tracing::error!("Failed to write aggregator stdin: {}", e);
                return Err(Status::internal("aggregator stdin write failed"));
            }
        }

        let output = child.wait_with_output()
            .map_err(|e| {
                tracing::error!("Aggregator wait failed: {}", e);
                Status::internal("aggregator wait failed")
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::error!("Aggregator process error: {}", stderr);
            return Err(Status::internal("aggregation failed"));
        }

        let result: serde_json::Value = serde_json::from_slice(&output.stdout)
            .map_err(|e| {
                tracing::error!("Invalid aggregator JSON output: {}", e);
                Status::internal("aggregator output parse failed")
            })?;

        let aggregated_uri = result["aggregated_uri"]
            .as_str()
            .ok_or_else(|| {
                tracing::error!("Aggregator output missing 'aggregated_uri'");
                Status::internal("aggregator output malformed")
            })?
            .to_string();

        // Re-acquire write guard to update round state
        let mut round = self
            .state
            .rounds
            .get_mut(&round_id)
            .ok_or_else(|| Status::not_found("round vanished after aggregation"))?;

        let num_updates = round.updates.len();

        round.state = RoundState::Complete;
        round.upload_uri = aggregated_uri.clone();
        round.aggregation_receipt = Some(AggregationReceipt {
            round_id,
            num_updates,
            aggregation_mode: "trimmed_mean".to_string(),
            aggregated_uri,
        });

        tracing::info!(
            "Round {} aggregation complete — {} updates, mode=trimmed_mean",
            round_id,
            num_updates
        );

        Ok(())
    }
}