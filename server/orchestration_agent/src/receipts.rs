use p256::ecdsa::{VerifyingKey, Signature, signature::Verifier};
use crate::errors::OrchestratorError;

pub fn verify(
    device_pubkey: &[u8],
    msg: &[u8],
    sig: &[u8],
) -> Result<(), OrchestratorError> {

    let verifying_key = VerifyingKey::from_sec1_bytes(device_pubkey)
        .map_err(|_| OrchestratorError::InvalidIdentity)?;

    let signature = Signature::from_der(sig)
        .map_err(|_| OrchestratorError::CryptoError)?;

    verifying_key
        .verify(msg, &signature)
        .map_err(|_| OrchestratorError::CryptoError)?;

    Ok(())
}