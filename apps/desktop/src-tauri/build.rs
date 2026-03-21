use std::fs;
use std::path::PathBuf;

fn ensure_dev_sidecar_stubs(target: &str, profile: &str) {
    if profile == "release" {
        return;
    }

    let binaries_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("binaries");
    fs::create_dir_all(&binaries_dir).unwrap();

    for base_name in ["gpt-rag-api", "gpt-rag-worker"] {
        let path = binaries_dir.join(format!("{base_name}-{target}"));
        if !path.exists() {
            fs::write(path, b"dev-sidecar-placeholder").unwrap();
        }
    }
}

fn main() {
    let target = std::env::var("TARGET").unwrap();
    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    println!("cargo:rustc-env=TARGET_TRIPLE={target}");
    ensure_dev_sidecar_stubs(&target, &profile);
    tauri_build::build();
}
