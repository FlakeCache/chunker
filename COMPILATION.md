# Compilation Strategy

## Runners

We use **self-hosted** Linux runners for all builds. The build environment is managed by **Nix**.

| Target | Runner OS | Compiler | Notes |
|--------|-----------|----------|-------|
| **Linux x86_64** | `self-hosted` | `cargo build` | Native compilation inside Nix shell. |
| **Linux ARM64** | `self-hosted` | `cross` | Uses QEMU/Docker via `cargo-cross` (provided by Nix). |
| **macOS ARM64** | `self-hosted` | `cargo zigbuild` | Cross-compiles to macOS using Zig (provided by Nix). |
| **Windows x86_64** | `self-hosted` | `cargo zigbuild` | Cross-compiles to Windows MSVC using Zig (provided by Nix). |
| **Windows ARM64** | `self-hosted` | `cargo zigbuild` | Cross-compiles to Windows ARM64 MSVC using Zig (provided by Nix). |
| **Linux Musl** | `self-hosted` | `cross` | Static binary (Alpine compatible). |
| **Linux ARMv7** | `self-hosted` | `cross` | 32-bit ARM (Raspberry Pi 2/3). |
| **Linux RISC-V** | `self-hosted` | `cross` | 64-bit RISC-V (VisionFive 2). |

## Environment Setup (Nix)

We use `flake.nix` to provide reproducible, minimal build environments. Instead of one giant environment, we define specialized shells:

- **`nix develop .#linux`**: Native Linux & Musl tools.
- **`nix develop .#cross-linux`**: Tools for ARM64, ARMv7, RISC-V.
- **`nix develop .#cross-macos`**: Zig & macOS SDKs.
- **`nix develop .#cross-windows`**: Zig & Windows SDKs.

### Security & Compliance

Every build artifact is:
1.  **Scanned**: An SBOM (Software Bill of Materials) is generated using `syft` in SPDX JSON format.
2.  **Signed**: Artifacts are signed using `cosign` (configured in CI).
- `cargo-cross`
- `openssl`, `pkg-config`

The CI workflow uses `nix develop` to enter this environment before building.
