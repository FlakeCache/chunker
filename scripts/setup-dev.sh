#!/usr/bin/env nix-shell
#!nix-shell -i bash -p bash git
# One-time dev machine setup - run this once per machine
# Configures global git hooks and direnv whitelist

set -e

echo "Setting up development environment..."

# 1. Whitelist flakecache repos for direnv (no manual approval needed)
mkdir -p ~/.config/direnv
cat >> ~/.config/direnv/direnv.toml << 'EOF'
[whitelist]
prefix = ["/home/nixos/code/flakecache", "~/code/flakecache"]
EOF
echo "✓ Direnv whitelist configured"

# 2. Set up global git template for lefthook
TEMPLATE_DIR="$HOME/.git-templates"
mkdir -p "$TEMPLATE_DIR/hooks"

# Create global pre-commit hook that delegates to lefthook if present
cat > "$TEMPLATE_DIR/hooks/pre-commit" << 'HOOK'
#!/usr/bin/env bash
if command -v lefthook &> /dev/null && [ -f lefthook.yml ]; then
  exec lefthook run pre-commit
elif command -v nix &> /dev/null && [ -f lefthook.yml ]; then
  exec nix develop -c lefthook run pre-commit
fi
HOOK
chmod +x "$TEMPLATE_DIR/hooks/pre-commit"

cat > "$TEMPLATE_DIR/hooks/pre-push" << 'HOOK'
#!/usr/bin/env bash
if command -v lefthook &> /dev/null && [ -f lefthook.yml ]; then
  exec lefthook run pre-push
elif command -v nix &> /dev/null && [ -f lefthook.yml ]; then
  exec nix develop -c lefthook run pre-push
fi
HOOK
chmod +x "$TEMPLATE_DIR/hooks/pre-push"

cat > "$TEMPLATE_DIR/hooks/commit-msg" << 'HOOK'
#!/usr/bin/env bash
if command -v lefthook &> /dev/null && [ -f lefthook.yml ]; then
  exec lefthook run commit-msg
elif command -v nix &> /dev/null && [ -f lefthook.yml ]; then
  exec nix develop -c lefthook run commit-msg
fi
HOOK
chmod +x "$TEMPLATE_DIR/hooks/commit-msg"

# Configure git to use this template for all new repos
git config --global init.templateDir "$TEMPLATE_DIR"
echo "✓ Global git template configured at $TEMPLATE_DIR"

# 3. Re-init current repo to apply template
if [ -d .git ]; then
  git init
  echo "✓ Applied hooks to current repo"
fi

echo ""
echo "Setup complete! All new git clones will auto-use lefthook."
echo "For existing repos, run: git init (in the repo dir)"
