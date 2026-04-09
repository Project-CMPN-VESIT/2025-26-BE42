// server.rs — Multi-device OTP flow with RequestEnrollment + EnrollDevice

use std::sync::Arc;
use std::process::Command;
use std::io::Write;
use std::fs;

use tempfile::NamedTempFile;

use tonic::{Request, Response, Status};
use tonic::transport::Server;

use crate::config::Config;
use crate::crypto::{ct_eq, hash_bytes};
use crate::identity::derive_device_id;
use crate::state::OrchestratorState;
use crate::round::{UpdateMeta, RoundState, AggregationReceipt};

use crate::grpc::orchestrator::{
    Ack, Csr, Certificate, DeviceId, Receipt, RoundMetadata,
    EnrollmentRequest, EnrollmentRequestAck,
    EnrollRequest, EnrollResponse,
};

use crate::grpc::orchestrator::orchestrator_server::{Orchestrator, OrchestratorServer};

// ─────────────────────────────────────────────────────────────────────────────
// Service
// ─────────────────────────────────────────────────────────────────────────────

pub struct Service {
    state: Arc<OrchestratorState>,
    #[allow(dead_code)]
    cfg: Config,
}

#[tonic::async_trait]
impl Orchestrator for Service {

    // ── RequestEnrollment (Phase 1 — no OTP needed) ───────────────────────────
    //
    // Device announces itself.  Server generates a unique OTP for this device
    // and prints it to the admin console.  Admin communicates OTP to device
    // operator out-of-band (email, phone, admin portal, etc.).
    //
    // Multiple simultaneous enrollment requests are supported because each
    // device gets its own OTP entry in OTP_STORE.
    async fn request_enrollment(
        &self,
        req: Request<EnrollmentRequest>,
    ) -> Result<Response<EnrollmentRequestAck>, Status> {

        let peer_addr = req
            .remote_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let inner = req.into_inner();

        if inner.device_pubkey.is_empty() {
            return Err(Status::invalid_argument("device_pubkey must not be empty"));
        }
        if inner.csr.is_empty() {
            return Err(Status::invalid_argument("csr must not be empty"));
        }

        // Derive stable fingerprint from public key
        let fingerprint_bytes = hash_bytes(&inner.device_pubkey);
        let fingerprint = hex::encode(&fingerprint_bytes[..8]); // first 8 bytes → 16 hex chars

        // Generate a unique OTP bound to this device fingerprint
        let otp = crate::otp::generate_otp_for(Some(fingerprint.clone()));

        // Store pending CSR so EnrollDevice can use it (optional optimisation)
        // We store the raw CSR bytes keyed by fingerprint so EnrollDevice can
        // skip re-submission if needed. For now we just store pubkey + csr.
        self.state.pending_enrollments.insert(
            fingerprint.clone(),
            (inner.device_pubkey.clone(), inner.csr.clone()),
        );

        // ── Admin notification (console) ──────────────────────────────────────
        let device_info = if inner.device_info.is_empty() {
            format!("peer={}", peer_addr)
        } else {
            format!("{} / peer={}", inner.device_info, peer_addr)
        };

        println!(
            "\n╔══════════════════════════════════════════════════════════════╗"
        );
        println!(
            "║  NEW ENROLLMENT REQUEST                                      ║"
        );
        println!(
            "║  Device fingerprint : {:<38} ║", fingerprint
        );
        println!(
            "║  Device info        : {:<38} ║", &device_info[..device_info.len().min(38)]
        );
        println!(
            "║  OTP (send to user) : {:<38} ║", otp
        );
        println!(
            "╚══════════════════════════════════════════════════════════════╝\n"
        );

        tracing::info!(
            "Enrollment requested — fingerprint={} info={} peer={}",
            fingerprint, inner.device_info, peer_addr
        );

        Ok(Response::new(EnrollmentRequestAck {
            accepted: true,
            device_fingerprint: fingerprint,
        }))
    }

    // ── EnrollDevice (Phase 2 — OTP required) ────────────────────────────────
    async fn enroll_device(
        &self,
        req: Request<EnrollRequest>,
    ) -> Result<Response<EnrollResponse>, Status> {

        let peer_addr = req
            .remote_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let req_inner = req.into_inner();

        // Rate-limited OTP consumption
        if !crate::otp::consume_otp_from(&req_inner.enrollment_token, &peer_addr) {
            tracing::warn!(
                "Enrollment rejected for peer {} — invalid/expired OTP",
                peer_addr
            );
            return Err(Status::permission_denied("invalid or expired OTP"));
        }

        if req_inner.device_pubkey.is_empty() {
            return Err(Status::invalid_argument("device_pubkey must not be empty"));
        }
        if req_inner.csr.is_empty() {
            return Err(Status::invalid_argument("CSR must not be empty"));
        }

        // Register device
        let device_id = derive_device_id(&req_inner.device_pubkey);
        self.state.devices.insert(device_id, req_inner.device_pubkey.clone());

        // Sign CSR
        let mut csr_file = NamedTempFile::new()
            .map_err(|e| {
                tracing::error!("Failed to create temp CSR file: {}", e);
                Status::internal("internal error")
            })?;

        csr_file.write_all(&req_inner.csr)
            .map_err(|e| {
                tracing::error!("Failed to write CSR: {}", e);
                Status::internal("internal error")
            })?;

        let cert_file = NamedTempFile::new()
            .map_err(|_| Status::internal("internal error"))?;

        let output = Command::new("openssl")
            .args([
                "x509", "-req",
                "-in",  csr_file.path().to_str().unwrap(),
                "-CA",  &self.cfg.tls.ca_cert,
                "-CAkey", &self.cfg.tls.ca_key,
                "-CAcreateserial",
                "-out", cert_file.path().to_str().unwrap(),
                "-days", "365",
                "-sha256",
            ])
            .output()
            .map_err(|e| {
                tracing::error!("openssl exec failed: {}", e);
                Status::internal("certificate signing failed")
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::error!("openssl signing failed: {}", stderr);
            return Err(Status::internal("certificate signing failed"));
        }

        let signed_cert = fs::read(cert_file.path())
            .map_err(|e| {
                tracing::error!("Failed to read signed cert: {}", e);
                Status::internal("certificate read failed")
            })?;

        if signed_cert.is_empty() {
            return Err(Status::internal("certificate signing produced empty output"));
        }

        // Clean up pending enrollment entry
        let fingerprint_bytes = hash_bytes(&req_inner.device_pubkey);
        let fingerprint = hex::encode(&fingerprint_bytes[..8]);
        self.state.pending_enrollments.remove(&fingerprint);

        tracing::info!(
            "Device enrolled — fingerprint={} peer={}",
            fingerprint, peer_addr
        );

        println!(
            "[ENROLLED] Device fingerprint={} from peer={}",
            fingerprint, peer_addr
        );

        Ok(Response::new(EnrollResponse {
            ok: true,
            client_cert: signed_cert,
        }))
    }

    // ── RegisterDevice (deprecated) ───────────────────────────────────────────
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
        self.state.devices.insert(device_id, inner.device_pubkey.clone());

        tracing::warn!("register_device called — deprecated; use RequestEnrollment + EnrollDevice");

        Ok(Response::new(Certificate { pem: inner.device_pubkey }))
    }

    // ── GetRound ──────────────────────────────────────────────────────────────
    async fn get_round(
        &self,
        req: Request<DeviceId>,
    ) -> Result<Response<RoundMetadata>, Status> {

        Self::require_client_cert(&req)?;

        let round = self.state.rounds.get(&1)
            .ok_or_else(|| Status::not_found("round not found"))?;

        let receipt_ref = round.aggregation_receipt.as_ref();

        Ok(Response::new(RoundMetadata {
            round_id:         round.id,
            model_version:    round.model_version.clone(),
            epsilon_max:      round.epsilon_max,
            upload_uri:       round.upload_uri.clone(),
            state:            format!("{:?}", round.state),
            num_updates:      receipt_ref.map(|r| r.num_updates as u32).unwrap_or(0),
            aggregation_mode: receipt_ref.map(|r| r.aggregation_mode.clone()).unwrap_or_default(),
        }))
    }

    // ── SubmitReceipt ─────────────────────────────────────────────────────────
    async fn submit_receipt(
        &self,
        req: Request<Receipt>,
    ) -> Result<Response<Ack>, Status> {

        Self::require_client_cert(&req)?;

        let receipt = req.into_inner();

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

        let pubkey = self.state.devices.iter()
            .find(|entry| ct_eq(entry.key(), &receipt.device_id))
            .map(|entry| entry.value().clone())
            .ok_or_else(|| {
                tracing::warn!("SubmitReceipt from unknown device");
                Status::permission_denied("unknown device")
            })?;

        let mut msg = Vec::with_capacity(
            receipt.device_id.len() + 8 + receipt.payload_hash.len()
        );
        msg.extend_from_slice(&receipt.device_id);
        msg.extend_from_slice(&receipt.round_id.to_be_bytes());
        msg.extend_from_slice(&receipt.payload_hash);

        crate::receipts::verify(&pubkey, &msg, &receipt.signature)
            .map_err(|_| {
                tracing::warn!("Invalid receipt signature");
                Status::permission_denied("invalid receipt signature")
            })?;

        let round_id = receipt.round_id;

        let mut round = self.state.rounds.get_mut(&round_id)
            .ok_or_else(|| Status::not_found("round not found"))?;

        if round.state != RoundState::Collecting {
            return Err(Status::failed_precondition("round is not accepting updates"));
        }

        if round.epsilon_spent + receipt.epsilon_spent > round.epsilon_max {
            return Err(Status::resource_exhausted("epsilon budget exceeded"));
        }

        round.epsilon_spent += receipt.epsilon_spent;

        round.updates.push(UpdateMeta {
            device_id: receipt.device_id.clone(),
            enc_uri:   receipt.enc_uri,
            scheme:    receipt.scheme,
            nonce:     if receipt.nonce.is_empty() { None } else { Some(receipt.nonce) },
        });

        let should_aggregate = round.updates.len() >= 3;

        if should_aggregate {
            round.state = RoundState::Aggregating;
            drop(round);
            self.run_aggregation(round_id)?;
        }

        Ok(Response::new(Ack { ok: true }))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Server bootstrap
// ─────────────────────────────────────────────────────────────────────────────

pub async fn serve(cfg: Config, state: Arc<OrchestratorState>) -> anyhow::Result<()> {
    let svc = Service { state, cfg: cfg.clone() };
    let addr = cfg.server.addr.parse()?;

    if cfg.server.enable_tls {
        let server_identity = tonic::transport::Identity::from_pem(
            std::fs::read(&cfg.tls.server_cert)?,
            std::fs::read(&cfg.tls.server_key)?,
        );
        let tls = tonic::transport::ServerTlsConfig::new().identity(server_identity);

        tracing::info!("[SERVER] TLS mode — binding to {}", addr);
        println!("[SERVER] Running in TLS mode on {}", addr);

        Server::builder()
            .tls_config(tls)?
            .add_service(OrchestratorServer::new(svc))
            .serve(addr)
            .await?;
    } else {
        tracing::warn!("[SERVER] INSECURE mode (enable_tls=false) — binding to {}", addr);
        println!("[SERVER] Running in INSECURE mode on {}", addr);

        Server::builder()
            .add_service(OrchestratorServer::new(svc))
            .serve(addr)
            .await?;
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

impl Service {
    fn require_client_cert<T>(req: &Request<T>) -> Result<(), Status> {
        if let Some(certs) = req.peer_certs() {
            if !certs.is_empty() {
                return Ok(());
            }
        }
        // Permissive during transition; tighten once all clients are enrolled:
        // return Err(Status::unauthenticated("client certificate required"));
        Ok(())
    }

    fn run_aggregation(&self, round_id: u64) -> Result<(), Status> {
        let job = {
            let round = self.state.rounds.get(&round_id)
                .ok_or_else(|| Status::not_found("round not found during aggregation"))?;

            serde_json::json!({
                "round_id": round.id,
                "mode": "trimmed_mean",
                "trim_ratio": 0.1,
                "updates": round.updates.iter().map(|u| serde_json::json!({
                    "enc_uri": u.enc_uri,
                    "scheme":  u.scheme,
                    "nonce":   u.nonce,
                })).collect::<Vec<_>>()
            })
        };

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
            if let Err(e) = stdin.write_all(job.to_string().as_bytes()) {
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
            tracing::error!("Aggregator error: {}", stderr);
            return Err(Status::internal("aggregation failed"));
        }

        let result: serde_json::Value = serde_json::from_slice(&output.stdout)
            .map_err(|e| {
                tracing::error!("Invalid aggregator JSON: {}", e);
                Status::internal("aggregator output parse failed")
            })?;

        let aggregated_uri = result["aggregated_uri"]
            .as_str()
            .ok_or_else(|| Status::internal("aggregator output malformed"))?
            .to_string();

        let mut round = self.state.rounds.get_mut(&round_id)
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
            "Round {} aggregation complete — {} updates",
            round_id, num_updates
        );

        Ok(())
    }
}