//! # CPU Feature Detection Loader
//!
//! This binary serves as a "trampoline" or "loader" for the `chunker` application.
//! Its primary purpose is to detect the available CPU features (AVX2, AVX-512, SVE, etc.)
//! at runtime and execute the most optimized binary variant available for the current hardware.
//!
//! ## How it works
//!
//! 1.  It inspects the directory where it is running.
//! 2.  It checks for the existence of optimized binaries (e.g., `chunker-x86_64-v4`).
//! 3.  It checks if the current CPU supports the required features for those binaries.
//! 4.  It `exec`s (replaces the current process image) with the best matching binary.
//!
//! This allows us to ship a single "fat" distribution that runs optimally on everything
//! from a 10-year-old laptop to a modern HPC cluster, without requiring the user to
//! manually select a binary.

use std::env;
use std::process::Command;
use std::path::PathBuf;

#[cfg(unix)]
use std::os::unix::process::CommandExt;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let current_exe = env::current_exe().unwrap_or_else(|_| PathBuf::from("./chunker"));
    let dir = current_exe.parent().unwrap_or_else(|| std::path::Path::new("."));

    // ------------------------------------------------------------------------
    // x86_64 Candidates (Intel/AMD) - Linux & Windows
    // ------------------------------------------------------------------------
    #[cfg(target_arch = "x86_64")]
    let candidates = [
        ("avx512f", "chunker-x86_64-v4"),
        ("avx2", "chunker-x86_64-v3"),
        ("generic", "chunker-x86_64-generic"),
    ];

    // ------------------------------------------------------------------------
    // AArch64 Candidates (ARM64 / Apple Silicon / Windows Dev Kit)
    // ------------------------------------------------------------------------
    #[cfg(target_arch = "aarch64")]
    let candidates = [
        ("sve", "chunker-aarch64-sve"),
        ("generic", "chunker-aarch64-generic"),
    ];

    for (feature, binary_name) in candidates {
        let supported = match feature {
            "generic" => true,
            #[cfg(target_arch = "x86_64")]
            "avx512f" => std::arch::is_x86_feature_detected!("avx512f"),
            #[cfg(target_arch = "x86_64")]
            "avx2" => std::arch::is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "aarch64")]
            "sve" => std::arch::is_aarch64_feature_detected!("sve"),
            _ => false,
        };

        if supported {
            // Handle Windows .exe extension
            let binary_name_with_ext = if cfg!(windows) {
                format!("{binary_name}.exe")
            } else {
                binary_name.to_string()
            };

            let binary_path = dir.join(&binary_name_with_ext);
            if binary_path.exists() {
                eprintln!("[Loader] Detected support for '{feature}'. Executing: {binary_name_with_ext:?}");
                
                let mut cmd = Command::new(binary_path);
                let _ = cmd.args(&args);

                #[cfg(unix)]
                {
                    let err = cmd.exec();
                    eprintln!("[Loader] Failed to exec {binary_name_with_ext:?}: {err}");
                    std::process::exit(1);
                }

                #[cfg(windows)]
                {
                    // Windows doesn't support exec(), so we spawn and wait.
                    match cmd.status() {
                        Ok(status) => std::process::exit(status.code().unwrap_or(1)),
                        Err(e) => {
                            eprintln!("[Loader] Failed to run {:?}: {}", binary_name_with_ext, e);
                            std::process::exit(1);
                        }
                    }
                }
            }
        }
    }

    eprintln!("[Loader] No suitable binary found.");
    std::process::exit(1);
}
