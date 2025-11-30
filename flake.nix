{
  description = "Chunker development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        # Base tools for all environments
        baseTools = with pkgs; [
          openssl
          pkg-config
          syft    # SBOM generation
          cosign  # Artifact signing
          zig
          cargo-zigbuild
          nodejs
        ];

        # Helper to create a Rust toolchain with specific targets
        mkRust = targets: pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
          inherit targets;
        };

      in
      {
        devShells = {
          # 1. Linux Native (x86_64) & Musl (Static)
          linux = pkgs.mkShell {
            buildInputs = baseTools ++ [
              (mkRust [ "x86_64-unknown-linux-gnu" "x86_64-unknown-linux-musl" ])
              pkgs.cargo-audit
            ];
          };

          # 2. Linux Cross (ARM64, ARMv7, RISC-V)
          cross-linux = pkgs.mkShell {
            buildInputs = baseTools ++ [
              (mkRust [ 
                "aarch64-unknown-linux-gnu" 
                "armv7-unknown-linux-gnueabihf" 
                "riscv64gc-unknown-linux-gnu" 
              ])
              pkgs.cargo-cross
            ];
          };

          # 3. macOS Cross (ARM64)
          cross-macos = pkgs.mkShell {
            buildInputs = baseTools ++ [
              (mkRust [ "aarch64-apple-darwin" ])
              pkgs.zig
              pkgs.cargo-zigbuild
            ];
          };

          # 4. Windows Cross (x64, ARM64)
          cross-windows = pkgs.mkShell {
            buildInputs = baseTools ++ [
              (mkRust [ "x86_64-pc-windows-msvc" "aarch64-pc-windows-msvc" ])
              pkgs.zig
              pkgs.cargo-zigbuild
            ];
          };

          # 5. Fuzzing Shell (Nightly)
          fuzz = pkgs.mkShell {
            buildInputs = baseTools ++ [
              (pkgs.rust-bin.nightly.latest.default.override {
                extensions = [ "rust-src" "llvm-tools-preview" ];
              })
              pkgs.cargo-fuzz
            ];
          };

          # Default shell (includes everything for local dev)
          default = pkgs.mkShell {
            buildInputs = baseTools ++ [
              (mkRust [
                "x86_64-unknown-linux-gnu"
                "x86_64-unknown-linux-musl"
                "aarch64-unknown-linux-gnu"
                "aarch64-apple-darwin"
                "x86_64-pc-windows-msvc"
              ])
              pkgs.cargo-fuzz
              pkgs.cargo-criterion
              pkgs.cargo-tarpaulin
              pkgs.zig
              pkgs.cargo-zigbuild
              pkgs.cargo-cross
              pkgs.lefthook
            ];

            shellHook = ''
              # Auto-install lefthook git hooks (quiet for direnv compatibility)
              if [ -d .git ] && command -v lefthook &> /dev/null; then
                lefthook install > /dev/null 2>&1 || true
              fi
            '';
          };
        };
      }
    );
}
