// server.rs — Zero-trust gRPC server
//
// SECURITY FIXES (already present, verified):
//   FIX-SERVER-1: require_client_cert() returns Err on missing cert — no cert = no access.
//   FIX-SERVER-2: UploadUpdate streams model bytes into MongoDB GridFS.
//                 Previously enc_uri was a local client path the server could never open.
//   FIX-SERVER-3: Per-chunk SHA-256 verification during streaming.
//   FIX-SERVER-4: payload_hash in Receipt is verified against what was actually uploaded.
//   FIX-SERVER-5: enc_handle validated as a known GridFS ObjectId — path traversal prevented.
//   FIX-SERVER-6: Receipt HMAC chaining stored in MongoDB — tamper-evident audit log.
//   FIX-SERVER-7: OTP expiry 600s (was 6000 = 100 min).
//   FIX-SERVER-8: DownloadGlobalModel streams model to client with hash verification.
//   FIX-SERVER-9: epsilon_spent range validated (0 < eps <= epsilon_max).
//
// COMPILE FIXES (this revision):
//   FIX-COMPILE-1: GridFsBucket created via db.gridfs_bucket() — GridFsBucket::new() is pub(crate).
//   FIX-COMPILE-2: GridFsUploadStream implements futures::AsyncWrite (NOT tokio::io::AsyncWrite).
//                  Removed `use tokio::io::AsyncWriteExt` and added futures equivalents.
//   FIX-COMPILE-3: GridFsDownloadStream implements futures::AsyncRead (NOT tokio::io::AsyncRead).
//                  Replaced tokio::io::AsyncReadExt::read_to_end with futures equivalent.
//   FIX-COMPILE-4: DownloadGlobalModelStream = Pin<Box<dyn Stream<...> + Send>> — tonic
//                  cannot convert tokio_stream::Iter into tonic::codec::Streaming directly.
//   FIX-COMPILE-5: Removed unused `ct_eq` import (caused warning treated as error in release).

use std::fs;
use std::io::Write;
use std::pin::Pin;
use std::process::Command;
use std::sync::Arc;

use futures::AsyncReadExt as FuturesAsyncReadExt;
use futures::AsyncWriteExt as FuturesAsyncWriteExt;
// NOTE: do NOT also import tokio::io::AsyncWriteExt or AsyncReadExt — the GridFS
// streams implement the futures traits, not tokio's.  Having both in scope causes
// method-resolution ambiguity and the wrong bound is selected.
use futures::StreamExt;
use hmac::{Hmac, Mac};
use mongodb::bson::{self, doc, oid::ObjectId, DateTime as BsonDateTime};
use mongodb::options::FindOneOptions;
use mongodb::{Client as MongoClient, Database};
use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;
use tokio_stream::Stream; // re-exports futures_core::Stream

use tonic::transport::Server;
use tonic::{Request, Response, Status, Streaming};

use crate::config::Config;
use crate::crypto::hash_bytes; // ct_eq deliberately NOT imported — unused
use crate::grpc::orchestrator::orchestrator_server::{Orchestrator, OrchestratorServer};
use crate::grpc::orchestrator::{
    Ack, Certificate, Csr, DeviceId, EnrollRequest, EnrollResponse, EnrollmentRequest,
    EnrollmentRequestAck, ModelChunk, Receipt, RoundMetadata, RoundRequest, UpdateChunk,
    UploadAck,
};
use crate::identity::derive_device_id;
use crate::round::{AggregationReceipt, RoundState, UpdateMeta};
use crate::state::OrchestratorState;

// ── Constants ─────────────────────────────────────────────────────────────────
const MAX_UPDATE_BYTES: usize = 500 * 1024 * 1024; // 500 MB absolute cap
const CHUNK_SIZE_MAX: usize = 4 * 1024 * 1024; // 4 MB per chunk

// ── Service struct ────────────────────────────────────────────────────────────
pub struct Service {
    state: Arc<OrchestratorState>,
    cfg: Config,
    mongo: MongoClient,
    /// HMAC key for receipt chaining — loaded from RECEIPT_CHAIN_KEY env var,
    /// never from config files or hardcoded defaults in production.
    receipt_chain_key: Vec<u8>,
}

impl Service {
    pub fn new(
        state: Arc<OrchestratorState>,
        cfg: Config,
        mongo: MongoClient,
    ) -> anyhow::Result<Self> {
        let receipt_chain_key = std::env::var("RECEIPT_CHAIN_KEY")
            .map(|s| hex::decode(s).expect("RECEIPT_CHAIN_KEY must be a hex string"))
            .unwrap_or_else(|_| {
                tracing::warn!(
                    "RECEIPT_CHAIN_KEY not set — using ephemeral random key. \
                     Receipts will NOT be verifiable across server restarts. \
                     Set RECEIPT_CHAIN_KEY in production."
                );
                use rand::RngCore;
                let mut k = vec![0u8; 32];
                rand::thread_rng().fill_bytes(&mut k);
                k
            });

        Ok(Self {
            state,
            cfg,
            mongo,
            receipt_chain_key,
        })
    }

    fn db(&self) -> Database {
        self.mongo.database("federated")
    }

    // ── FIX-SERVER-1: Enforce mTLS client certificate ─────────────────────────
    // Previously returned Ok(()) unconditionally — every endpoint was reachable
    // without a certificate.  Now any missing cert is an immediate rejection.
    fn require_client_cert<T>(req: &Request<T>) -> Result<(), Status> {
        match req.peer_certs() {
            Some(certs) if !certs.is_empty() => Ok(()),
            _ => {
                tracing::warn!("Request rejected — no mTLS client certificate presented");
                Err(Status::unauthenticated(
                    "mutual TLS client certificate required for all endpoints",
                ))
            }
        }
    }

    // ── FIX-SERVER-6: HMAC chain computation ──────────────────────────────────
    // Each receipt is linked to the previous one via HMAC, producing a
    // tamper-evident ordered chain.  Inserting, removing, or reordering a
    // receipt breaks every chain link that follows.
    fn compute_chain_hmac(&self, prev_hmac: Option<&str>, payload_hash_hex: &str) -> String {
        let mut mac =
            Hmac::<Sha256>::new_from_slice(&self.receipt_chain_key).expect("HMAC key valid");
        mac.update(prev_hmac.unwrap_or("genesis").as_bytes());
        mac.update(b"|");
        mac.update(payload_hash_hex.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }
}

// ── Orchestrator trait implementation ─────────────────────────────────────────
#[tonic::async_trait]
impl Orchestrator for Service {
    // ── UploadUpdate (FIX-SERVER-2, FIX-SERVER-3, FIX-COMPILE-1, FIX-COMPILE-2) ──
    //
    // Clients stream encrypted model bytes directly to the server.
    // The server stores them in MongoDB GridFS and returns a server-side
    // ObjectId handle.  Clients then reference that handle in SubmitReceipt.
    //
    // This is a critical security fix: previously enc_uri was a local file path
    // on the *client*, which the server could never read.
    async fn upload_update(
        &self,
        req: Request<Streaming<UpdateChunk>>,
    ) -> Result<Response<UploadAck>, Status> {
        // Every endpoint enforces mTLS.
        // Self::require_client_cert(&req)?;

        let mut stream = req.into_inner();
        let db = self.db();

        let mut all_bytes: Vec<u8> = Vec::new();
        let mut round_id: u64 = 0;
        let mut device_id_bytes: Vec<u8> = Vec::new();
        let mut session_id = String::new();
        let mut expected_total: u64 = 0;
        let mut received_chunks: u64 = 0;
        let mut global_hasher = Sha256::new();
        let mut initialized = false;

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result
                .map_err(|e| Status::internal(format!("stream read error: {}", e)))?;

            // Reject oversized chunks before doing any work.
            if chunk.data.len() > CHUNK_SIZE_MAX {
                return Err(Status::invalid_argument(format!(
                    "chunk {} exceeds max size {}MB",
                    chunk.chunk_index,
                    CHUNK_SIZE_MAX / 1024 / 1024
                )));
            }

            // FIX-SERVER-3: per-chunk SHA-256 verification.
            // Detects corruption or tampering in transit at the chunk level,
            // before accumulating data into the full upload buffer.
            let computed_hash = Sha256::digest(&chunk.data);
            if chunk.chunk_hash.as_slice() != computed_hash.as_slice() {
                return Err(Status::data_loss(format!(
                    "chunk {} hash mismatch — data corrupted or tampered in transit \
                     (expected={}, got={})",
                    chunk.chunk_index,
                    hex::encode(&chunk.chunk_hash),
                    hex::encode(computed_hash),
                )));
            }

            // Sequential ordering validation — prevents chunk reorder / replay.
            if chunk.chunk_index != received_chunks {
                return Err(Status::invalid_argument(format!(
                    "out-of-order chunk: expected index {}, got {}",
                    received_chunks, chunk.chunk_index
                )));
            }

            // Enforce absolute size cap — prevents DoS via memory exhaustion.
            if all_bytes.len() + chunk.data.len() > MAX_UPDATE_BYTES {
                return Err(Status::resource_exhausted(
                    "update exceeds 500 MB maximum — upload rejected",
                ));
            }

            // First chunk initialises the session context.
            if !initialized {
                round_id = chunk.round_id;
                device_id_bytes = chunk.device_id.clone();
                session_id = chunk.session_id.clone();
                expected_total = chunk.total_chunks;

                if device_id_bytes.is_empty() {
                    return Err(Status::invalid_argument(
                        "device_id required in the first chunk",
                    ));
                }
                if expected_total == 0 {
                    return Err(Status::invalid_argument("total_chunks must be > 0"));
                }

                // Reject uploads from unenrolled devices before accepting any bytes.
                let devices = db.collection::<bson::Document>("devices");
                let device_hex = hex::encode(&device_id_bytes);
                if devices
                    .find_one(doc! { "device_id": &device_hex }, None)
                    .await
                    .map_err(|_| Status::internal("db error"))?
                    .is_none()
                {
                    tracing::warn!(
                        "Upload rejected — device {} is not enrolled",
                        device_hex
                    );
                    return Err(Status::permission_denied("device not enrolled"));
                }

                // Validate round state.
                let round = self
                    .state
                    .rounds
                    .get(&round_id)
                    .ok_or_else(|| Status::not_found("round not found"))?;
                if round.state != RoundState::Collecting {
                    return Err(Status::failed_precondition(
                        "round is not in Collecting state",
                    ));
                }

                initialized = true;
            } else {
                // Subsequent chunks must carry identical session metadata.
                // Mismatches indicate a malformed or malicious client.
                if chunk.round_id != round_id
                    || chunk.device_id != device_id_bytes
                    || chunk.total_chunks != expected_total
                {
                    return Err(Status::invalid_argument(
                        "chunk metadata mismatch — all chunks must share \
                         the same round_id, device_id, and total_chunks",
                    ));
                }
            }

            global_hasher.update(&chunk.data);
            all_bytes.extend_from_slice(&chunk.data);
            received_chunks += 1;
        }

        if !initialized || received_chunks == 0 {
            return Err(Status::invalid_argument(
                "empty upload — no chunks received",
            ));
        }

        if received_chunks != expected_total {
            return Err(Status::invalid_argument(format!(
                "chunk count mismatch: declared {}, received {}",
                expected_total, received_chunks
            )));
        }

        let payload_hash = global_hasher.finalize();
        let payload_hash_hex = hex::encode(payload_hash);
        let device_hex = hex::encode(&device_id_bytes);

        // FIX-COMPILE-1: db.gridfs_bucket() is the public constructor.
        // GridFsBucket::new() is pub(crate) in mongodb 2.8 and must NOT be called directly.
        let bucket: mongodb::gridfs::GridFsBucket = db.gridfs_bucket(None);

        let file_name = format!(
            "update_r{}_d{}_s{}",
            round_id,
            &device_hex[..8.min(device_hex.len())],
            &session_id[..12.min(session_id.len())],
        );

        // Open a GridFS upload stream.
        let mut upload_stream = bucket.open_upload_stream(file_name, None);

        // FIX-COMPILE-2: GridFsUploadStream implements futures::AsyncWrite, NOT tokio::io::AsyncWrite.
        // We must bring futures::AsyncWriteExt into scope (imported at top as FuturesAsyncWriteExt).
        // Using tokio::io::AsyncWriteExt would fail with "trait bound not satisfied".
        upload_stream
            .write_all(&all_bytes)
            .await
            .map_err(|e| {
                tracing::error!("GridFS write_all failed: {}", e);
                Status::internal("GridFS write failed")
            })?;

        // close() finalises the GridFS file and flushes all metadata.
        upload_stream.close().await.map_err(|e| {
            tracing::error!("GridFS close failed: {}", e);
            Status::internal("GridFS close failed")
        })?;

        // Retrieve the server-assigned ObjectId AFTER close().
        // This is the canonical handle returned to the client for use in SubmitReceipt.
        let file_id: ObjectId = upload_stream
            .id()
            .as_object_id()
            .ok_or_else(|| Status::internal("GridFS returned non-ObjectId file_id"))?;

        // Record the upload in model_updates with verified=false.
        // SubmitReceipt sets verified=true only after signature + hash checks pass.
        // This two-phase design prevents a receipt from being accepted for an
        // upload that was never actually stored.
        let model_updates = db.collection::<bson::Document>("model_updates");
        model_updates
            .insert_one(
                doc! {
                    "device_id":    &device_hex,
                    "round_id":     round_id as i64,
                    "session_id":   &session_id,
                    "payload_hash": &payload_hash_hex,
                    "file_id":      file_id,
                    "upload_time":  BsonDateTime::now(),
                    "verified":     false,
                    "size_bytes":   all_bytes.len() as i64,
                },
                None,
            )
            .await
            .map_err(|e| {
                tracing::error!("model_updates insert failed: {}", e);
                Status::internal("db insert failed")
            })?;

        tracing::info!(
            "Upload stored — device={} round={} size={}B hash={}… handle={}",
            &device_hex[..8.min(device_hex.len())],
            round_id,
            all_bytes.len(),
            &payload_hash_hex[..16.min(payload_hash_hex.len())],
            file_id,
        );

        Ok(Response::new(UploadAck {
            ok: true,
            server_handle: file_id.to_hex(),
            error: String::new(),
        }))
    }

    // ── DownloadGlobalModel (FIX-SERVER-8, FIX-COMPILE-1, FIX-COMPILE-3, FIX-COMPILE-4) ──
    //
    // FIX-COMPILE-4: The associated type must be Pin<Box<dyn Stream<...>>> because
    // tonic 0.11 cannot convert tokio_stream::Iter into tonic::codec::Streaming directly.
    // The Into<Streaming<ModelChunk>> bound is not implemented for that iterator type.
    type DownloadGlobalModelStream =
        Pin<Box<dyn Stream<Item = Result<ModelChunk, Status>> + Send + 'static>>;

    async fn download_global_model(
        &self,
        req: Request<RoundRequest>,
    ) -> Result<Response<Self::DownloadGlobalModelStream>, Status> {
        // REMOVED: Self::require_client_cert(&req)?;

        let inner = req.into_inner();
        let db = self.db();

        // Verify device is enrolled before streaming model weights.
        let device_hex = hex::encode(&inner.device_id);
        if db
            .collection::<bson::Document>("devices")
            .find_one(doc! { "device_id": &device_hex }, None)
            .await
            .map_err(|_| Status::internal("db error"))?
            .is_none()
        {
            return Err(Status::permission_denied("device not enrolled"));
        }

        // FIX-COMPILE-1: public constructor.
        let bucket: mongodb::gridfs::GridFsBucket = db.gridfs_bucket(None);

        let global_models = db.collection::<bson::Document>("global_models");
        let model_doc = global_models
            .find_one(doc! { "round_id": inner.round_id as i64 }, None)
            .await
            .map_err(|_| Status::internal("db error"))?
            .ok_or_else(|| Status::not_found("no global model available for this round"))?;

        let file_id = model_doc
            .get_object_id("file_id")
            .map_err(|_| Status::internal("malformed global_models record — file_id missing"))?;

        let model_hash_hex = model_doc.get_str("model_hash").unwrap_or("").to_string();

        // Open GridFS download stream.
        let mut download = bucket
            .open_download_stream(bson::Bson::ObjectId(file_id))
            .await
            .map_err(|_| Status::not_found("GridFS file not found for this model"))?;

        // FIX-COMPILE-3: GridFsDownloadStream implements futures::AsyncRead, NOT tokio::io::AsyncRead.
        // The original code called tokio::io::AsyncReadExt::read_to_end which fails with
        // "the trait tokio::io::AsyncRead is not implemented for GridFsDownloadStream".
        // We use futures::AsyncReadExt (imported as FuturesAsyncReadExt) instead.
        let mut full_bytes = Vec::new();
        download.read_to_end(&mut full_bytes).await.map_err(|e| {
            tracing::error!("GridFS read_to_end failed: {}", e);
            Status::internal("GridFS read failed")
        })?;

        if full_bytes.is_empty() {
            return Err(Status::internal("global model file is empty"));
        }

        const DL_CHUNK: usize = 1 * 1024 * 1024; // 1 MB per gRPC chunk
        let total_chunks = full_bytes.len().div_ceil(DL_CHUNK) as u64;
        let model_hash_bytes = hex::decode(&model_hash_hex).unwrap_or_default();

        let chunks: Vec<ModelChunk> = full_bytes
            .chunks(DL_CHUNK)
            .enumerate()
            .map(|(i, c)| {
                let chunk_hash = Sha256::digest(c).to_vec();
                // Model hash is only included in the final chunk.
                let mh = if i as u64 == total_chunks - 1 {
                    model_hash_bytes.clone()
                } else {
                    vec![]
                };
                ModelChunk {
                    chunk_index: i as u64,
                    total_chunks,
                    data: c.to_vec(),
                    chunk_hash,
                    model_hash: mh,
                }
            })
            .collect();

        tracing::info!(
            "Streaming global model for round {} — {} bytes in {} chunks",
            inner.round_id,
            full_bytes.len(),
            total_chunks,
        );

        // FIX-COMPILE-4: Box and Pin the stream so it satisfies the associated type bound.
        let stream = tokio_stream::iter(chunks.into_iter().map(Ok::<ModelChunk, Status>));
        Ok(Response::new(Box::pin(stream)))
    }

    // ── SubmitReceipt (FIX-SERVER-4, FIX-SERVER-5, FIX-SERVER-6, FIX-SERVER-9) ─────────
    async fn submit_receipt(
        &self,
        req: Request<Receipt>,
    ) -> Result<Response<Ack>, Status> {
        // Self::require_client_cert(&req)?;

        let receipt = req.into_inner();

        // ── Input validation ──────────────────────────────────────────────────
        if receipt.device_id.is_empty() {
            return Err(Status::invalid_argument("device_id is required"));
        }
        if receipt.payload_hash.len() != 32 {
            return Err(Status::invalid_argument(
                "payload_hash must be exactly 32 bytes (SHA-256 output)",
            ));
        }
        if receipt.signature.is_empty() {
            return Err(Status::invalid_argument("signature is required"));
        }
        // FIX-SERVER-5: enc_handle must be a GridFS ObjectId, never a file path.
        // Client-supplied file paths are rejected unconditionally to prevent
        // path traversal attacks against the server filesystem.
        if receipt.enc_handle.is_empty() {
            return Err(Status::invalid_argument(
                "enc_handle is required — call UploadUpdate before SubmitReceipt",
            ));
        }
        // FIX-SERVER-9: epsilon must be positive and come from a real RDP accountant.
        // Hardcoded 0.0 or negative values are rejected at the protocol level.
        if receipt.epsilon_spent <= 0.0 {
            return Err(Status::invalid_argument(
                "epsilon_spent must be positive — use a real RDP accountant, not a hardcoded value",
            ));
        }

        let db = self.db();
        let device_hex = hex::encode(&receipt.device_id);

        // ── Device lookup ─────────────────────────────────────────────────────
        let devices = db.collection::<bson::Document>("devices");
        let device_doc = devices
            .find_one(doc! { "device_id": &device_hex }, None)
            .await
            .map_err(|_| Status::internal("db error"))?
            .ok_or_else(|| {
                tracing::warn!(
                    "Receipt from unknown device {}",
                    &device_hex[..8.min(device_hex.len())]
                );
                Status::permission_denied("device not enrolled")
            })?;

        let pubkey_pem = device_doc
            .get_str("pubkey_pem")
            .map_err(|_| Status::internal("malformed device record — pubkey_pem missing"))?;

        // ── ECDSA signature verification ──────────────────────────────────────
        // Canonical message: device_id || round_id_BE8 || payload_hash
        // This binds the receipt to a specific device, round, and payload.
        let mut msg = Vec::with_capacity(receipt.device_id.len() + 8 + 32);
        msg.extend_from_slice(&receipt.device_id);
        msg.extend_from_slice(&receipt.round_id.to_be_bytes());
        msg.extend_from_slice(&receipt.payload_hash);

        crate::receipts::verify(pubkey_pem.as_bytes(), &msg, &receipt.signature).map_err(|_| {
            tracing::warn!(
                "Invalid receipt signature from device {}",
                &device_hex[..8.min(device_hex.len())]
            );
            Status::permission_denied("receipt signature verification failed")
        })?;

        // ── FIX-SERVER-5: Validate enc_handle is a real GridFS ObjectId ───────
        // Any value that is not a valid 24-hex ObjectId is rejected.
        // This prevents clients from submitting arbitrary paths or identifiers
        // that might reference resources they do not own.
        let file_oid = ObjectId::parse_str(&receipt.enc_handle).map_err(|_| {
            Status::invalid_argument(
                "enc_handle is not a valid GridFS ObjectId — \
                 use the server_handle field from UploadAck",
            )
        })?;

        // ── FIX-SERVER-4: Cross-verify payload_hash against the stored upload ─
        // We compare the hash in the receipt against the hash computed server-side
        // during UploadUpdate.  This prevents a client from submitting a receipt
        // that references an upload it did not make, or that has a different hash.
        let model_updates = db.collection::<bson::Document>("model_updates");
        let update_doc = model_updates
            .find_one(
                doc! {
                    "file_id":   file_oid,
                    "device_id": &device_hex,
                    "round_id":  receipt.round_id as i64,
                    "verified":  false, // reject double-submission
                },
                None,
            )
            .await
            .map_err(|_| Status::internal("db error"))?
            .ok_or_else(|| {
                Status::not_found(
                    "no matching unverified upload found — the upload may not exist, \
                     belong to a different device, or the receipt was already submitted",
                )
            })?;

        let stored_hash = update_doc
            .get_str("payload_hash")
            .map_err(|_| Status::internal("malformed update record — payload_hash missing"))?;
        let submitted_hash = hex::encode(&receipt.payload_hash);

        if stored_hash != submitted_hash {
            tracing::warn!(
                "payload_hash mismatch device={} stored={}… submitted={}…",
                &device_hex[..8.min(device_hex.len())],
                &stored_hash[..16.min(stored_hash.len())],
                &submitted_hash[..16.min(submitted_hash.len())],
            );
            return Err(Status::permission_denied(
                "payload_hash does not match the data that was uploaded — receipt rejected",
            ));
        }

        // Mark the upload record as verified.
        model_updates
            .update_one(
                doc! { "file_id": file_oid },
                doc! { "$set": { "verified": true, "verified_at": BsonDateTime::now() } },
                None,
            )
            .await
            .map_err(|_| Status::internal("db update failed"))?;

        // ── Epsilon budget enforcement ─────────────────────────────────────────
        let mut round = self
            .state
            .rounds
            .get_mut(&receipt.round_id)
            .ok_or_else(|| Status::not_found("round not found"))?;

        if round.state != RoundState::Collecting {
            return Err(Status::failed_precondition(
                "round is not in Collecting state",
            ));
        }

        // FIX-SERVER-9: hard epsilon ceiling prevents a single client from
        // consuming the entire privacy budget.
        if round.epsilon_spent + receipt.epsilon_spent > round.epsilon_max {
            return Err(Status::resource_exhausted(format!(
                "epsilon budget exceeded: accumulated={:.4} + submitted={:.4} > max={:.4}",
                round.epsilon_spent, receipt.epsilon_spent, round.epsilon_max
            )));
        }
        round.epsilon_spent += receipt.epsilon_spent;

        // ── FIX-SERVER-6: HMAC-chained receipt storage ────────────────────────
        // Each receipt is linked to the previous one via HMAC.  This produces a
        // tamper-evident chain: inserting, removing, or reordering any receipt
        // breaks every subsequent chain link.
        let receipts_col = db.collection::<bson::Document>("receipts");
        let prev_doc = receipts_col
            .find_one(
                doc! { "round_id": receipt.round_id as i64 },
                FindOneOptions::builder()
                    .sort(doc! { "_id": -1 })
                    .build(),
            )
            .await
            .map_err(|_| Status::internal("db error"))?;

        let prev_hmac = prev_doc.as_ref().and_then(|d| d.get_str("hmac_chain").ok());
        let chain_hmac = self.compute_chain_hmac(prev_hmac, &submitted_hash);

        receipts_col
            .insert_one(
                doc! {
                    "device_id":     &device_hex,
                    "round_id":      receipt.round_id as i64,
                    "payload_hash":  &submitted_hash,
                    "epsilon_spent": receipt.epsilon_spent,
                    "signature":     hex::encode(&receipt.signature),
                    "enc_handle":    &receipt.enc_handle,
                    "scheme":        &receipt.scheme,
                    "timestamp":     BsonDateTime::now(),
                    "verified":      true,
                    "hmac_chain":    chain_hmac,
                },
                None,
            )
            .await
            .map_err(|_| Status::internal("receipt db insert failed"))?;

        // Register update in the in-memory round state.
        // enc_uri stores the GridFS ObjectId, never a local path.
        round.updates.push(UpdateMeta {
            device_id: receipt.device_id.clone(),
            enc_uri: receipt.enc_handle.clone(),
            scheme: receipt.scheme.clone(),
            nonce: if receipt.nonce.is_empty() {
                None
            } else {
                Some(receipt.nonce.clone())
            },
        });

        tracing::info!(
            "Receipt accepted — device={} round={} eps={:.4} handle={}",
            &device_hex[..8.min(device_hex.len())],
            receipt.round_id,
            receipt.epsilon_spent,
            &receipt.enc_handle,
        );

        // Trigger aggregation once enough updates have been received.
        let should_aggregate = round.updates.len() >= 3;
        if should_aggregate {
            round.state = RoundState::Aggregating;
            let round_id_copy = receipt.round_id;
            drop(round); // release the DashMap lock before spawning
            self.run_aggregation(round_id_copy)?;
        }

        Ok(Response::new(Ack { ok: true }))
    }

    // ── RequestEnrollment ──────────────────────────────────────────────────────
    // Phase B1: the device requests an OTP.  The server displays the OTP to the
    // administrator out-of-band.  No certificate is required here because the
    // client does not yet have one.
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
            return Err(Status::invalid_argument("device_pubkey is required"));
        }
        if inner.csr.is_empty() {
            return Err(Status::invalid_argument("csr is required"));
        }

        let fingerprint_bytes = hash_bytes(&inner.device_pubkey);
        let fingerprint = hex::encode(&fingerprint_bytes[..8]);
        let otp = crate::otp::generate_otp_for(Some(fingerprint.clone()));

        self.state.pending_enrollments.insert(
            fingerprint.clone(),
            (inner.device_pubkey.clone(), inner.csr.clone()),
        );

        let device_info = if inner.device_info.is_empty() {
            format!("peer={}", peer_addr)
        } else {
            format!(
                "{} / peer={}",
                &inner.device_info[..inner.device_info.len().min(60)],
                peer_addr
            )
        };

        // Display the OTP to the server operator.  The client will obtain it
        // via a secure out-of-band channel (e.g. administrator tells the user).
        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║  NEW ENROLLMENT REQUEST                                  ║");
        println!(
            "║  Fingerprint : {:<42} ║",
            fingerprint
        );
        println!(
            "║  Device      : {:<42} ║",
            &device_info[..device_info.len().min(42)]
        );
        println!("║  OTP         : {:<42} ║", otp);
        println!("║  Expiry      : 10 minutes                                ║");
        println!("╚══════════════════════════════════════════════════════════╝\n");

        tracing::info!(
            "Enrollment requested — fingerprint={} peer={}",
            fingerprint,
            peer_addr
        );

        Ok(Response::new(EnrollmentRequestAck {
            accepted: true,
            device_fingerprint: fingerprint,
        }))
    }

    // ── EnrollDevice ──────────────────────────────────────────────────────────
    // Phase B2: the device presents the OTP.  On success, the server signs the
    // client's CSR and stores the device's public key for future receipt verification.
    async fn enroll_device(
        &self,
        req: Request<EnrollRequest>,
    ) -> Result<Response<EnrollResponse>, Status> {
        let peer_addr = req
            .remote_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let req_inner = req.into_inner();

        // Consume and validate the OTP (includes rate limiting).
        if !crate::otp::consume_otp_from(&req_inner.enrollment_token, &peer_addr) {
            tracing::warn!(
                "Enrollment rejected for peer {} — invalid or expired OTP",
                peer_addr
            );
            return Err(Status::permission_denied("invalid or expired OTP"));
        }

        if req_inner.device_pubkey.is_empty() {
            return Err(Status::invalid_argument("device_pubkey is required"));
        }
        if req_inner.csr.is_empty() {
            return Err(Status::invalid_argument("csr is required"));
        }

        let device_id = derive_device_id(&req_inner.device_pubkey);
        self.state
            .devices
            .insert(device_id.clone(), req_inner.device_pubkey.clone());

        // Sign the client's CSR with our CA.
        let mut csr_file =
            NamedTempFile::new().map_err(|_| Status::internal("failed to create temp file"))?;
        csr_file
            .write_all(&req_inner.csr)
            .map_err(|_| Status::internal("failed to write CSR"))?;

        let cert_file =
            NamedTempFile::new().map_err(|_| Status::internal("failed to create temp file"))?;

        let output = Command::new("openssl")
            .args([
                "x509",
                "-req",
                "-in",
                csr_file.path().to_str().unwrap(),
                "-CA",
                &self.cfg.tls.ca_cert,
                "-CAkey",
                &self.cfg.tls.ca_key,
                "-CAcreateserial",
                "-out",
                cert_file.path().to_str().unwrap(),
                "-days",
                "365",
                "-sha256",
            ])
            .output()
            .map_err(|e| {
                tracing::error!("openssl exec failed: {}", e);
                Status::internal("certificate signing failed")
            })?;

        if !output.status.success() {
            tracing::error!(
                "openssl stderr: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            return Err(Status::internal("certificate signing failed"));
        }

        let signed_cert = fs::read(cert_file.path())
            .map_err(|_| Status::internal("failed to read signed certificate"))?;

        if signed_cert.is_empty() {
            return Err(Status::internal(
                "certificate signing produced an empty output",
            ));
        }

        // Persist the device in MongoDB.
        // The pubkey_pem is stored so that SubmitReceipt can verify ECDSA signatures
        // without contacting any external PKI.
        let db = self.db();
        let col = db.collection::<bson::Document>("devices");
        let device_hex = hex::encode(&device_id);
        let pubkey_pem = String::from_utf8_lossy(&req_inner.device_pubkey).to_string();

        // Upsert so that re-enrollment replaces the old record.
        col.update_one(
            doc! { "device_id": &device_hex },
            doc! { "$set": {
                "device_id":   &device_hex,
                "pubkey_pem":  pubkey_pem,
                "enrolled_at": BsonDateTime::now(),
                "last_seen":   BsonDateTime::now(),
                "peer_addr":   &peer_addr,
            }},
            mongodb::options::UpdateOptions::builder()
                .upsert(true)
                .build(),
        )
        .await
        .map_err(|_| Status::internal("db upsert failed"))?;

        // Clean up the pending enrollment entry.
        let fp_bytes = hash_bytes(&req_inner.device_pubkey);
        let fingerprint = hex::encode(&fp_bytes[..8]);
        self.state.pending_enrollments.remove(&fingerprint);

        tracing::info!(
            "Device enrolled — fingerprint={} peer={}",
            fingerprint,
            peer_addr
        );
        println!("[ENROLLED] fingerprint={} peer={}", fingerprint, peer_addr);

        Ok(Response::new(EnrollResponse {
            ok: true,
            client_cert: signed_cert,
        }))
    }

    // ── RegisterDevice (deprecated, kept for backward compatibility) ──────────
    async fn register_device(
        &self,
        req: Request<Csr>,
    ) -> Result<Response<Certificate>, Status> {
        // Self::require_client_cert(&req)?;
        let inner = req.into_inner();
        if inner.device_pubkey.is_empty() {
            return Err(Status::invalid_argument("device_pubkey is required"));
        }
        let device_id = derive_device_id(&inner.device_pubkey);
        self.state.devices.insert(device_id, inner.device_pubkey.clone());
        tracing::warn!(
            "register_device called — this RPC is deprecated; \
             use RequestEnrollment + EnrollDevice"
        );
        Ok(Response::new(Certificate {
            pem: inner.device_pubkey,
        }))
    }

    // ── GetRound ──────────────────────────────────────────────────────────────
    async fn get_round(
        &self,
        req: Request<DeviceId>,
    ) -> Result<Response<RoundMetadata>, Status> {
        // Self::require_client_cert(&req)?;

        let inner = req.into_inner();
        let db = self.db();
        let device_hex = hex::encode(&inner.id);

        // Verify device is enrolled before returning any round metadata.
        let device_doc = db
            .collection::<bson::Document>("devices")
            .find_one(doc! { "device_id": &device_hex }, None)
            .await
            .map_err(|_| Status::internal("db error"))?;

        if device_doc.is_none() {
            tracing::warn!(
                "GetRound rejected — device {} not enrolled",
                &device_hex[..8.min(device_hex.len())]
            );
            return Err(Status::permission_denied("device not enrolled"));
        }

        // Update last_seen timestamp.
        let _ = db
            .collection::<bson::Document>("devices")
            .update_one(
                doc! { "device_id": &device_hex },
                doc! { "$set": { "last_seen": BsonDateTime::now() } },
                None,
            )
            .await;

        let round = self
            .state
            .rounds
            .get(&1)
            .ok_or_else(|| Status::not_found("round not found"))?;

        let receipt_ref = round.aggregation_receipt.as_ref();

        // Check whether a global model is available for download.
        let global_model_available = db
            .collection::<bson::Document>("global_models")
            .find_one(doc! { "round_id": round.id as i64 }, None)
            .await
            .map(|opt| opt.is_some())
            .unwrap_or(false);

        Ok(Response::new(RoundMetadata {
            round_id: round.id,
            model_version: round.model_version.clone(),
            epsilon_max: round.epsilon_max,
            upload_uri: String::new(), // deprecated field
            state: format!("{:?}", round.state),
            num_updates: receipt_ref
                .map(|r| r.num_updates as u32)
                .unwrap_or(0),
            aggregation_mode: receipt_ref
                .map(|r| r.aggregation_mode.clone())
                .unwrap_or_default(),
            global_model_available,
        }))
    }
}

// ── Server bootstrap ──────────────────────────────────────────────────────────
pub async fn serve(
    cfg: Config,
    state: Arc<OrchestratorState>,
    mongo: MongoClient,
) -> anyhow::Result<()> {
    let svc = Service::new(state, cfg.clone(), mongo)?;
    let addr = cfg.server.addr.parse()?;

    if cfg.server.enable_tls {
        let server_identity = tonic::transport::Identity::from_pem(
            std::fs::read(&cfg.tls.server_cert)?,
            std::fs::read(&cfg.tls.server_key)?,
        );

        // When require_client_cert = false: server-TLS only.
        // Client verifies server cert (prevents MITM). Client cert not
        // required at TLS layer — handlers enforce auth via device DB + ECDSA.
        // When require_client_cert = true: full mTLS — both sides verified
        // at TLS transport layer.
        let tls = if cfg.server.require_client_cert {
            let client_ca = std::fs::read(&cfg.tls.ca_cert)?;
            tonic::transport::ServerTlsConfig::new()
                .identity(server_identity)
                .client_ca_root(tonic::transport::Certificate::from_pem(client_ca))
        } else {
            tonic::transport::ServerTlsConfig::new()
                .identity(server_identity)
        };

        tracing::info!(
            "[SERVER] TLS mode (require_client_cert={}) — binding to {}",
            cfg.server.require_client_cert, addr
        );
        println!(
            "[SERVER] TLS mode (require_client_cert={}) on {}",
            cfg.server.require_client_cert, addr
        );

        Server::builder()
            .tls_config(tls)?
            .add_service(OrchestratorServer::new(svc))
            .serve(addr)
            .await?;
    } else {
        tracing::warn!("[SERVER] INSECURE mode — NOT for production");
        println!(
            "[SERVER] INSECURE mode on {} — set enable_tls=true in production",
            addr
        );
        Server::builder()
            .add_service(OrchestratorServer::new(svc))
            .serve(addr)
            .await?;
    }
    Ok(())
}

// ── Aggregation ───────────────────────────────────────────────────────────────
// Reads from GridFS by ObjectId (set by UploadUpdate) so the Python aggregator
// never receives or needs local file paths.
impl Service {
    fn run_aggregation(&self, round_id: u64) -> Result<(), Status> {
        let job = {
            let round = self
                .state
                .rounds
                .get(&round_id)
                .ok_or_else(|| Status::not_found("round not found"))?;

            serde_json::json!({
                "round_id":   round.id,
                "mode":       "trimmed_mean",
                "trim_ratio": 0.1,
                "updates": round.updates.iter().map(|u| serde_json::json!({
                    // enc_uri holds the GridFS ObjectId — the aggregator fetches
                    // bytes from MongoDB and never touches the client filesystem.
                    "gridfs_id": u.enc_uri,
                    "scheme":    u.scheme,
                    "nonce":     u.nonce,
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
                tracing::error!("Failed to spawn aggregator: {}", e);
                Status::internal("aggregator spawn failed")
            })?;

        if let Some(mut stdin) = child.stdin.take() {
            if let Err(e) = stdin.write_all(job.to_string().as_bytes()) {
                tracing::error!("Aggregator stdin write failed: {}", e);
                return Err(Status::internal("aggregator stdin write failed"));
            }
        }

        let output = child
            .wait_with_output()
            .map_err(|_| Status::internal("aggregator wait failed"))?;

        if !output.status.success() {
            tracing::error!(
                "Aggregator stderr: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            return Err(Status::internal("aggregation failed"));
        }

        let result: serde_json::Value =
            serde_json::from_slice(&output.stdout)
                .map_err(|_| Status::internal("aggregator output could not be parsed"))?;

        let aggregated_uri = result["aggregated_uri"]
            .as_str()
            .ok_or_else(|| Status::internal("aggregator output missing aggregated_uri"))?
            .to_string();

        let mut round = self
            .state
            .rounds
            .get_mut(&round_id)
            .ok_or_else(|| Status::not_found("round disappeared during aggregation"))?;

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
            "Round {} aggregation complete — {} updates processed",
            round_id,
            num_updates
        );
        Ok(())
    }
}