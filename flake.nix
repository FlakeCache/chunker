{
  description = "Chunker development environment";

  nixConfig = {
    extra-substituters = [ "https://cache.centralcloud.com/default" ];
    extra-trusted-public-keys = [ "default:ESyvaQTiq681JA0iaH5tsQWS+R5qqJUVdVY1OXbi9to=" ];
  };

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
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
          erlang  # Required for NIF (Rustler)
        ];

        # Helper to create a Rust toolchain with specific targets
        mkRust = targets: pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
          inherit targets;
        };

        # The deployable single-node Nix binary-cache server (std-only; no NIF,
        # no openssl -- pure Rust deps only).
        nodeBin = pkgs.rustPlatform.buildRustPackage {
          pname = "flakecache-node-bin";
          version = "0.1.0";
          src = ./.;
          cargoLock.lockFile = ./Cargo.lock;
          cargoBuildFlags = [ "-p" "flakecache-node-bin" ];
          # Build only the node binary's closure; skip workspace tests (they need
          # the dev shell / bind sockets).
          doCheck = false;
          buildAndTestSubdir = null;
        };

        # A minimal OCI image wrapping the node binary. Listens on 0.0.0.0:8501;
        # data dir and signing key are provided at runtime (env / mounted secret).
        nodeImage = pkgs.dockerTools.buildLayeredImage {
          name = "flakecache-node";
          tag = "dev";
          contents = [ pkgs.cacert ];
          config = {
            Entrypoint = [ "${nodeBin}/bin/flakecache-node-bin" ];
            Env = [
              "FLAKECACHE_LISTEN=0.0.0.0:8501"
              "FLAKECACHE_DATA_DIR=/data"
            ];
            ExposedPorts = { "8501/tcp" = { }; };
            Volumes = { "/data" = { }; };
          };
        };

      in
      {
        packages = {
          flakecache-node-bin = nodeBin;
          flakecache-node-image = nodeImage;
        };

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
              pkgs.just
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
