use serde::Deserialize;

#[derive(Deserialize, Clone)]
pub struct Config {
    pub server: Server,
    pub tls: Tls,
}

#[derive(Deserialize, Clone)]
pub struct Tls {
    pub ca_cert: String,
    pub ca_key: String,
    pub server_cert: String,
    pub server_key: String,
}

#[derive(Deserialize, Clone)]
pub struct Server {
    pub addr: String,
    pub enable_tls: bool
}

impl Config {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let cfg = toml::from_str(&content)?;
        Ok(cfg)
    }
}
