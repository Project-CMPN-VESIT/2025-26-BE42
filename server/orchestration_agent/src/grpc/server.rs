use std::sync::Arc;
use std::process::Command;
use std::io::Write;

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

// Generated protobuf types
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

/// Orchestrator gRPC service
pub struct Service {
    state: Arc<OrchestratorState>,
    #[allow(dead_code)]
    cfg: Config,
}

#[tonic::async_trait]
impl Orchestrator for Service {

    async fn enroll_device(
    &self,
    req: Request<EnrollRequest>,
) -> Result<Response<EnrollResponse>, Status> {

        use std::fs;
        use std::process::Command;
        use std::io::Write;
        use tempfile::NamedTempFile;

        let req = req.into_inner();

        // 1. Validate OTP
        if !crate::otp::consume_otp(&req.enrollment_token) {
            return Err(Status::permission_denied("invalid or expired OTP"));
        }

        // 2. Store TPM pubkey
        let device_id = derive_device_id(&req.device_pubkey);
        self.state.devices.insert(device_id, req.device_pubkey.clone());

        // 3. Write CSR to temp file
        let mut csr_file = NamedTempFile::new()
            .map_err(|_| Status::internal("failed to create temp CSR file"))?;

        csr_file.write_all(&req.csr)
            .map_err(|_| Status::internal("failed to write CSR"))?;

        let cert_file = NamedTempFile::new()
            .map_err(|_| Status::internal("failed to create temp cert file"))?;

        // 4. Sign CSR using OpenSSL + CA
        let output = Command::new("openssl")
            .arg("x509")
            .arg("-req")
            .arg("-in").arg(csr_file.path())
            .arg("-CA").arg(&self.cfg.tls.ca_cert)
            .arg("-CAkey").arg(&self.cfg.tls.ca_key)
            .arg("-CAcreateserial")
            .arg("-out").arg(cert_file.path())
            .arg("-days").arg("365")
            .arg("-sha256")
            .output()
            .map_err(|_| Status::internal("failed to execute openssl"))?;

        if !output.status.success() {
            return Err(Status::internal("openssl CSR signing failed"));
        }

        // 5. Read signed certificate
        let signed_cert = fs::read(cert_file.path())
            .map_err(|_| Status::internal("failed to read signed cert"))?;

        tracing::info!("Device enrolled + client cert issued");

        Ok(Response::new(EnrollResponse {
            ok: true,
            client_cert: signed_cert,
        }))
    }

    // --------------------------------------------------
    // Device registration (identity bootstrap)
    // --------------------------------------------------
    async fn register_device(
        &self,
        req: Request<Csr>,
    ) -> Result<Response<Certificate>, Status> {

        let pubkey = req.into_inner().device_pubkey;
        let device_id = derive_device_id(&pubkey);

        self.state.devices.insert(device_id, pubkey.clone());

        tracing::warn!("register_device is deprecated; use EnrollDevice");

        Ok(Response::new(Certificate { pem: pubkey }))
    }

    // --------------------------------------------------
    // Query current round metadata
    // --------------------------------------------------
    async fn get_round(
        &self,
        _req: Request<DeviceId>,
    ) -> Result<Response<RoundMetadata>, Status> {

        let round = self.state
            .rounds
            .get(&1)
            .ok_or_else(|| Status::not_found("round not found"))?;

        let receipt = round.aggregation_receipt.as_ref();

        Ok(Response::new(RoundMetadata {
            round_id: round.id,
            model_version: round.model_version.clone(),
            epsilon_max: round.epsilon_max,
            upload_uri: round.upload_uri.clone(),
            state: format!("{:?}", round.state),

            num_updates: receipt
                .map(|r| r.num_updates)
                .unwrap_or(0) as u32,

            aggregation_mode: receipt
                .map(|r| r.aggregation_mode.clone())
                .unwrap_or_else(|| "".to_string()),
        }))
    }

    // --------------------------------------------------
    // Submit receipt + update metadata
    // --------------------------------------------------
    async fn submit_receipt(
        &self,
        req: Request<Receipt>,
    ) -> Result<Response<Ack>, Status> {

        let receipt = req.into_inner();

        // 2. Verify receipt signature
        let pubkey = self.state.devices.iter()
            .find(|entry| ct_eq(entry.key(), &receipt.device_id))
            .map(|entry| entry.value().clone())
            .ok_or_else(|| Status::permission_denied("unknown device"))?;

        let mut msg = Vec::new();
        msg.extend_from_slice(&receipt.device_id);
        msg.extend_from_slice(&receipt.round_id.to_be_bytes());
        msg.extend_from_slice(&receipt.payload_hash);

        crate::receipts::verify(
            &pubkey,
            &msg,
            &receipt.signature,
        )
        .map_err(|_| Status::permission_denied("invalid receipt signature"))?;

        // 3. Fetch round
        let mut round = self.state.rounds
            .get_mut(&receipt.round_id)
            .ok_or_else(|| Status::not_found("round not found"))?;

        // 4. Enforce round state
        if round.state != RoundState::Collecting {
            return Err(Status::failed_precondition("round not accepting updates"));
        }

        // 5. Enforce ε-budget
        if round.epsilon_spent + receipt.epsilon_spent > round.epsilon_max {
            return Err(Status::resource_exhausted("epsilon budget exceeded"));
        }

        round.epsilon_spent += receipt.epsilon_spent;

        // 6. Store update metadata (NO DATA PLANE)
        round.updates.push(UpdateMeta {
            device_id: receipt.device_id.clone(),
            enc_uri: receipt.enc_uri,
            scheme: receipt.scheme,
            nonce: if receipt.nonce.is_empty() {
                None
            } else {
                Some(receipt.nonce)
            },
        });

        // 7. Trigger aggregation (temporary threshold)
        if round.updates.len() >= 3 {
            round.state = RoundState::Aggregating;
            drop(round);
            self.run_aggregation(receipt.round_id)?;
        }

        Ok(Response::new(Ack { ok: true }))
    }
}

// --------------------------------------------------
// gRPC server bootstrap
// --------------------------------------------------
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

    let tls = ServerTlsConfig::new()
        .identity(server_identity)
        .client_ca_root(client_ca);

    let mut builder = Server::builder();

    println!("[DEBUG] TLS is being configured");
    builder = builder.tls_config(tls)?;
    println!("[DEBUG] TLS forced ON");

    println!("Binding to: {}", cfg.server.addr);

    builder
        .add_service(OrchestratorServer::new(svc))
        .serve(cfg.server.addr.parse()?)
        .await?;

    Ok(())
}

// --------------------------------------------------
// Aggregation trigger (control-plane → worker)
// --------------------------------------------------
impl Service {

    fn run_aggregation(&self, round_id: u64) -> Result<(), Status> {

        let round = self.state.rounds
            .get(&round_id)
            .ok_or_else(|| Status::not_found("round not found"))?;

        let job = serde_json::json!({
            "round_id": round.id,
            "mode": "trimmed_mean",
            "trim_ratio": 0.1,
            "updates": round.updates.iter().map(|u| {
                serde_json::json!({
                    "enc_uri": u.enc_uri,
                    "scheme": u.scheme,
                    "nonce": u.nonce
                })
            }).collect::<Vec<_>>()
        });

        let mut child = Command::new("python3")
            .arg("server/aggregator_agent/aggregator.py")
            .env("PYTHONPATH", ".")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .map_err(|_| Status::internal("failed to start aggregator"))?;

        if let Some(mut stdin) = child.stdin.take() {
            let payload = job.to_string();
            stdin.write_all(payload.as_bytes())
                .map_err(|_| Status::internal("failed to write to aggregator stdin"))?;
        }

        let output = child.wait_with_output()
            .map_err(|_| Status::internal("aggregator execution failed"))?;

        if !output.status.success() {
            return Err(Status::internal("aggregation process error"));
        }

        let result: serde_json::Value = serde_json::from_slice(&output.stdout)
            .map_err(|_| Status::internal("invalid aggregator output"))?;

        let mut round = self.state.rounds
            .get_mut(&round_id)
            .ok_or_else(|| Status::not_found("round not found"))?;

        round.state = RoundState::Complete;
        round.upload_uri = result["aggregated_uri"]
            .as_str()
            .unwrap()
            .to_string();

        round.aggregation_receipt = Some(AggregationReceipt {
            round_id,
            num_updates: round.updates.len(),
            aggregation_mode: "trimmed_mean".to_string(),
            aggregated_uri: round.upload_uri.clone(),
        });

        Ok(())
    }
}
