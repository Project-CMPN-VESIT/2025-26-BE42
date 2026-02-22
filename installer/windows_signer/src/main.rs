use std::io::{Read, Write};
use windows::core::*;
use windows::Win32::Security::Cryptography::*;
use sha2::{Sha256, Digest};

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

        NCryptOpenStorageProvider(
            &mut provider,
            w!("Microsoft Platform Crypto Provider"),
            0,
        )?;

        let mut key = NCRYPT_KEY_HANDLE::default();

        let open_result = NCryptOpenKey(
            provider,
            &mut key,
            w!("FederatedDeviceKey"),
            CERT_KEY_SPEC(0),
            NCRYPT_FLAGS(0),
        );

        if open_result.is_err() {
            NCryptCreatePersistedKey(
                provider,
                &mut key,
                w!("ECDSA_P256"),
                w!("FederatedDeviceKey"),
                CERT_KEY_SPEC(0),
                NCRYPT_MACHINE_KEY_FLAG,
            )?;

            NCryptFinalizeKey(key, NCRYPT_FLAGS(0))?;
        }

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

        println!("PUBLIC KEY BLOB SIZE: {}", len);
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