use std::io::{Read, Write};
use windows::core::*;
use windows::Win32::Security::Cryptography::*;
use sha2::{Sha256, Digest};
use base64::Engine;

const KEY_NAME: &str = "FederatedDeviceKey";

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  windows_signer --init");
        eprintln!("  windows_signer --pubkey <output_file>");
        eprintln!("  windows_signer --sign");
        std::process::exit(1);
    }

    let result = match args[1].as_str() {
        "--init" => cmd_init(),
        "--pubkey" => {
            if args.len() < 3 {
                eprintln!("[ERROR] --pubkey requires an output file path");
                std::process::exit(1);
            }
            cmd_pubkey(&args[2])
        }
        "--sign" => cmd_sign(),
        _ => {
            eprintln!("[ERROR] Unknown command: {}", args[1]);
            std::process::exit(1);
        }
    };

    if let Err(e) = result {
        eprintln!("[ERROR] windows_signer failed: {:?}", e);
        std::process::exit(1);
    }
}

// ── --init: open or create the persistent TPM key ────────────────────────────
fn cmd_init() -> Result<()> {
    let key = open_or_create_key()?;
    unsafe { NCryptFreeObject(key); }
    eprintln!("[OK] TPM key '{}' ready", KEY_NAME);
    Ok(())
}

// ── --pubkey <file>: export public key as PEM SPKI ───────────────────────────
fn cmd_pubkey(out_path: &str) -> Result<()> {
    let pem_bytes = export_public_key_pem()?;
    std::fs::write(out_path, &pem_bytes)
        .map_err(|e| Error::from_win32())?;
    eprintln!("[OK] Public key written to {}", out_path);
    Ok(())
}

// ── --sign: read stdin, sign SHA-256 digest, write DER to stdout ─────────────
fn cmd_sign() -> Result<()> {
    let mut input = Vec::new();
    std::io::stdin()
        .read_to_end(&mut input)
        .map_err(|_| Error::from_win32())?;

    if input.is_empty() {
        eprintln!("[ERROR] Empty input for signing");
        std::process::exit(1);
    }

    let mut hasher = Sha256::new();
    hasher.update(&input);
    let digest = hasher.finalize();

    let signature = sign_digest(&digest)?;
    std::io::stdout()
        .write_all(&signature)
        .map_err(|_| Error::from_win32())?;

    eprintln!("[OK] Signed {} bytes", input.len());
    Ok(())
}

// ── Key management ────────────────────────────────────────────────────────────
fn open_or_create_key() -> Result<NCRYPT_KEY_HANDLE> {
    unsafe {
        let mut provider = NCRYPT_PROV_HANDLE::default();
        NCryptOpenStorageProvider(
            &mut provider,
            w!("Microsoft Platform Crypto Provider"),
            0,
        )?;

        let mut key = NCRYPT_KEY_HANDLE::default();

        // Try to open existing persisted key
        if NCryptOpenKey(
            provider,
            &mut key,
            w!("FederatedDeviceKey"),
            CERT_KEY_SPEC(0),
            NCRYPT_FLAGS(0),
        ).is_ok() {
            NCryptFreeObject(provider);
            eprintln!("[TPM] Opened existing key '{}'", KEY_NAME);
            return Ok(key);
        }

        // Create new persisted key
        eprintln!("[TPM] Creating new key '{}'", KEY_NAME);
        NCryptCreatePersistedKey(
            provider,
            &mut key,
            w!("ECDSA_P256"),
            w!("FederatedDeviceKey"),
            CERT_KEY_SPEC(0),
            NCRYPT_FLAGS(0),
        )?;

        NCryptFinalizeKey(key, NCRYPT_FLAGS(0))?;
        NCryptFreeObject(provider);
        eprintln!("[TPM] Key '{}' created and finalized", KEY_NAME);
        Ok(key)
    }
}

fn sign_digest(digest: &[u8]) -> Result<Vec<u8>> {
    unsafe {
        let key = open_or_create_key()?;
        let mut sig_len = 0u32;

        NCryptSignHash(
            key,
            None,
            digest,
            None,
            &mut sig_len,
            NCRYPT_FLAGS(0),
        )?;

        let mut signature = vec![0u8; sig_len as usize];
        NCryptSignHash(
            key,
            None,
            digest,
            Some(&mut signature),
            &mut sig_len,
            NCRYPT_FLAGS(0),
        )?;

        NCryptFreeObject(key);
        Ok(encode_der_ecdsa(&signature))
    }
}

fn export_public_key_pem() -> Result<Vec<u8>> {
    unsafe {
        let key = open_or_create_key()?;
        let mut len = 0u32;

        NCryptExportKey(
            key, None,
            w!("PUBLICBLOB"),
            None, None,
            &mut len,
            NCRYPT_FLAGS(0),
        )?;

        let mut buf = vec![0u8; len as usize];
        NCryptExportKey(
            key, None,
            w!("PUBLICBLOB"),
            None, Some(&mut buf),
            &mut len,
            NCRYPT_FLAGS(0),
        )?;

        NCryptFreeObject(key);

        // CNG BCRYPT_ECCKEY_BLOB layout for P-256:
        // [4 bytes magic][4 bytes key_size=32][32 bytes X][32 bytes Y]
        if buf.len() < 72 {
            eprintln!("[ERROR] Unexpected key blob size: {}", buf.len());
            return Err(Error::from_win32());
        }

        let x = &buf[8..40];
        let y = &buf[40..72];

        // Uncompressed EC point: 0x04 || X || Y
        let mut ec_point = vec![0x04u8];
        ec_point.extend_from_slice(x);
        ec_point.extend_from_slice(y);

        // SPKI DER wrapper for P-256:
        // SEQUENCE {
        //   SEQUENCE { OID id-ecPublicKey, OID prime256v1 }
        //   BIT STRING { ec_point }
        // }
        let spki_prefix: [u8; 26] = [
            0x30, 0x59,
            0x30, 0x13,
            0x06, 0x07, 0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x02, 0x01,  // id-ecPublicKey
            0x06, 0x08, 0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x03, 0x01, 0x07, // prime256v1
            0x03, 0x42, 0x00,  // BIT STRING, 66 bytes, 0 unused bits
        ];
        let mut spki = Vec::new();
        spki.extend_from_slice(&spki_prefix);
        spki.extend_from_slice(&ec_point);

        let b64 = base64::engine::general_purpose::STANDARD.encode(&spki);
        let pem = format!(
            "-----BEGIN PUBLIC KEY-----\n{}\n-----END PUBLIC KEY-----\n",
            b64,
        );
        Ok(pem.into_bytes())
    }
}

// ── DER ECDSA signature encoding ─────────────────────────────────────────────
fn encode_der_ecdsa(rs: &[u8]) -> Vec<u8> {
    assert_eq!(rs.len(), 64, "Expected 64-byte raw R||S from CNG");
    fn encode_int(bytes: &[u8]) -> Vec<u8> {
        let mut v = bytes.to_vec();
        while v.len() > 1 && v[0] == 0 {
            v.remove(0);
        }
        if v[0] & 0x80 != 0 {
            let mut prefixed = vec![0u8];
            prefixed.extend(v);
            v = prefixed;
        }
        let mut out = vec![0x02, v.len() as u8];
        out.extend(v);
        out
    }
    let r = encode_int(&rs[0..32]);
    let s = encode_int(&rs[32..64]);
    let total = r.len() + s.len();
    let mut der = vec![0x30, total as u8];
    der.extend(r);
    der.extend(s);
    der
}