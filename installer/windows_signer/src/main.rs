use std::io::{Read, Write};
use windows::core::*;
use windows::Win32::Security::Cryptography::*;
use sha2::{Sha256, Digest};
use base64::Engine;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 && args[1] == "--export-pub" {
        export_public_key()?;
        return Ok(());
    }

    sign_stdin()?;
    Ok(())
}

fn open_or_create_key() -> Result<NCRYPT_KEY_HANDLE> {
    unsafe {
        let mut provider = NCRYPT_PROV_HANDLE::default();

        // Use TPM-backed provider
        NCryptOpenStorageProvider(
            &mut provider,
            w!("Microsoft Platform Crypto Provider"),
            0,
        )?;

        let mut key = NCRYPT_KEY_HANDLE::default();

        // Try open with MACHINE scope
        let open = NCryptOpenKey(
            provider,
            &mut key,
            w!("FederatedDeviceKey"),
            CERT_KEY_SPEC(0),
            NCRYPT_MACHINE_KEY_FLAG,
        );

        if open.is_ok() {
            return Ok(key);
        }

        // Try create
        let create = NCryptCreatePersistedKey(
            provider,
            &mut key,
            w!("ECDSA_P256"),
            w!("FederatedDeviceKey"),
            CERT_KEY_SPEC(0),
            NCRYPT_MACHINE_KEY_FLAG,
        );

        match create {
            Ok(_) => {
                NCryptFinalizeKey(key, NCRYPT_FLAGS(0))?;
                return Ok(key);
            }
            Err(e) => {
                // If key already exists, try opening again
                if e.code().0 as u32 == 0x8009000F {
                    NCryptOpenKey(
                        provider,
                        &mut key,
                        w!("FederatedDeviceKey"),
                        CERT_KEY_SPEC(0),
                        NCRYPT_MACHINE_KEY_FLAG,
                    )?;
                    return Ok(key);
                } else {
                    return Err(e);
                }
            }
        }
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
    }

    Ok(())
}

fn export_public_key() -> Result<()> {
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

        // PUBLICBLOB structure:
        // BCRYPT_ECCPUBLIC_BLOB header (8 bytes)
        // then X (32 bytes)
        // then Y (32 bytes)

        let x = &buf[8..40];
        let y = &buf[40..72];

        // Uncompressed EC point format
        let mut ec_point = vec![0x04];
        ec_point.extend(x);
        ec_point.extend(y);

        // ASN.1 header for P-256 SPKI
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

        println!("{}", pem);
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