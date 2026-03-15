use std::io::{Read, Write};
use windows::core::*;
use windows::Win32::Security::Cryptography::*;
use sha2::{Sha256, Digest};
use base64::Engine;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  windows_signer --init");
        eprintln!("  windows_signer --pubkey <file>");
        eprintln!("  windows_signer --sign");
        std::process::exit(1);
    }

    match args[1].as_str() {

        "--init" => {
            open_or_create_key()?;
            println!("[TPM] Windows TPM key initialized");
        }

        "--pubkey" => {
            if args.len() < 3 {
                eprintln!("Missing output file");
                std::process::exit(1);
            }

            let pem = export_public_key_bytes()?;
            std::fs::write(&args[2], pem)?;
        }

        "--sign" => {
            sign_stdin()?;
        }

        _ => {
            eprintln!("Unknown command");
            std::process::exit(1);
        }
    }

    Ok(())
}

fn export_public_key_bytes() -> Result<Vec<u8>> {
    unsafe {
        let key = open_or_create_key()?;

        let mut len = 0u32;

        NCryptExportKey(
            key,
            None,
            w!("PUBLICBLOB"),
            None,
            None,
            &mut len,
            NCRYPT_FLAGS(0),
        )?;

        let mut buf = vec![0u8; len as usize];

        NCryptExportKey(
            key,
            None,
            w!("PUBLICBLOB"),
            None,
            Some(&mut buf),
            &mut len,
            NCRYPT_FLAGS(0),
        )?;

        let x = &buf[8..40];
        let y = &buf[40..72];

        let mut ec_point = vec![0x04];
        ec_point.extend(x);
        ec_point.extend(y);

        let spki_prefix: [u8; 26] = [
            0x30, 0x59,
            0x30, 0x13,
            0x06, 0x07, 0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x02, 0x01,
            0x06, 0x08, 0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x03, 0x01, 0x07,
            0x03, 0x42, 0x00
        ];

        let mut spki = Vec::new();
        spki.extend(spki_prefix);
        spki.extend(ec_point);

        let pem = format!(
            "-----BEGIN PUBLIC KEY-----\n{}\n-----END PUBLIC KEY-----",
            base64::engine::general_purpose::STANDARD.encode(spki)
        );
        
        NCryptFreeObject(key);
        Ok(pem.into_bytes())
    }
}

fn open_or_create_key() -> Result<NCRYPT_KEY_HANDLE> {
    unsafe {

        let mut provider = NCRYPT_PROV_HANDLE::default();

        NCryptOpenStorageProvider(
            &mut provider,
            w!("Microsoft Platform Crypto Provider"),
            0,
        )?;

        let mut key = NCRYPT_KEY_HANDLE::default();

        if NCryptOpenKey(
            provider,
            &mut key,
            w!("FederatedDeviceKey"),
            0,
            NCRYPT_MACHINE_KEY_FLAG,
        ).is_ok()
        {
            NCryptFreeObject(provider);
            return Ok(key);
        }

        NCryptCreatePersistedKey(
            provider,
            &mut key,
            w!("ECDSA_P256"),
            w!("FederatedDeviceKey"),
            0,
            NCRYPT_MACHINE_KEY_FLAG,
        )?;

        NCryptFinalizeKey(key, NCRYPT_FLAGS(0))?;

        NCryptFreeObject(provider);

        Ok(key)
    }
}

fn sign_stdin() -> Result<()> {
    let mut input = Vec::new();
    std::io::stdin().read_to_end(&mut input)?;

    let mut hasher = Sha256::new();
    hasher.update(&input);
    let digest = hasher.finalize();

    unsafe {
        let key = open_or_create_key()?;

        let mut sig_len = 0u32;

        NCryptSignHash(
            key,
            None,
            &digest,
            None,
            &mut sig_len,
            NCRYPT_FLAGS(0),
        )?;

        let mut signature = vec![0u8; sig_len as usize];

        NCryptSignHash(
            key,
            None,
            &digest,
            Some(&mut signature),
            &mut sig_len,
            NCRYPT_FLAGS(0),
        )?;

        let der = encode_der_ecdsa(&signature);
        std::io::stdout().write_all(&der)?;
        NCryptFreeObject(key);
    }

    Ok(())
}

fn encode_der_ecdsa(rs: &[u8]) -> Vec<u8> {
    assert_eq!(rs.len(), 64);

    fn encode_int(bytes: &[u8]) -> Vec<u8> {
        let mut v = bytes.to_vec();

        // strip leading zeros
        while v.len() > 1 && v[0] == 0 {
            v.remove(0);
        }

        // if high bit set → prepend 0x00
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

    let total_len = r.len() + s.len();
    let mut der = vec![0x30, total_len as u8];
    der.extend(r);
    der.extend(s);

    der
}