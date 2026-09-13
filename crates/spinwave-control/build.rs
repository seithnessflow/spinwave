//! The engine's fingerprint, for every measurement the knowledge base
//! stores (`notes/knowledge-base-design.md`): a SHA-256 over the sources
//! AND data of the engine crates (the parameter table and the factory
//! wavetable live there) plus the toolchain (`rustc -vV` and the target:
//! two compilers can differ in the ulps), and a second one over the
//! descriptors alone, so a new descriptor stales no render. The git
//! commit and a dirty flag ride along as provenance, never as the key.

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Command;

fn files_under(dir: &Path, out: &mut Vec<PathBuf>) {
    if dir.is_file() {
        out.push(dir.to_path_buf());
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            files_under(&path, out);
        } else {
            out.push(path);
        }
    }
}

fn hash_trees(root: &Path, trees: &[&str], salt: &[String]) -> String {
    let mut files = Vec::new();
    for tree in trees {
        files_under(&root.join(tree), &mut files);
    }
    files.sort();
    let mut hasher = Sha256::new();
    for file in files {
        let relative = file.strip_prefix(root).unwrap_or(&file);
        hasher.update(relative.to_string_lossy().replace('\\', "/").as_bytes());
        hasher.update([0u8]);
        hasher.update(std::fs::read(&file).unwrap_or_default());
        hasher.update([0u8]);
    }
    for s in salt {
        hasher.update(s.as_bytes());
        hasher.update([0u8]);
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn command_output(cmd: &str, args: &[&str], dir: &Path) -> String {
    Command::new(cmd)
        .args(args)
        .current_dir(dir)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let root = manifest.parent().and_then(|p| p.parent()).expect("repo root").to_path_buf();

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let toolchain = command_output(&rustc, &["-vV"], &root);
    let target = std::env::var("TARGET").unwrap_or_default();
    let salt = vec![toolchain, target];

    let engine_trees = [
        "crates/spinwave-poly/src",
        "crates/spinwave-dsp/src",
        "crates/spinwave-engine/src",
        "crates/spinwave-params/src",
    ];
    let engine = hash_trees(&root, &engine_trees, &salt);
    let descriptor_files = [
        "crates/spinwave-control/src/analysis.rs",
        "crates/spinwave-control/src/ops/descriptors.rs",
        "crates/spinwave-control/src/ops/distance.rs",
    ];
    let descriptors = hash_trees(&root, &descriptor_files, &[]);

    let git = command_output("git", &["rev-parse", "--short", "HEAD"], &root);
    let dirty = !command_output("git", &["status", "--porcelain", "--untracked-files=no"], &root).is_empty();

    println!("cargo:rustc-env=SPINWAVE_ENGINE_FINGERPRINT={engine}");
    println!("cargo:rustc-env=SPINWAVE_DESCRIPTORS_FINGERPRINT={descriptors}");
    println!("cargo:rustc-env=SPINWAVE_GIT_COMMIT={git}");
    println!("cargo:rustc-env=SPINWAVE_GIT_DIRTY={}", if dirty { "1" } else { "0" });
    for tree in engine_trees.iter().chain(descriptor_files.iter()) {
        println!("cargo:rerun-if-changed={}", root.join(tree).display());
    }
    println!("cargo:rerun-if-changed={}", root.join(".git/HEAD").display());
    println!("cargo:rerun-if-changed={}", root.join(".git/index").display());
}
