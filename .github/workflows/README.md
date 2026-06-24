# Workflows moved to Forgejo

Canonical CI/CD workflows live in **[`.forgejo/workflows/`](../.forgejo/workflows/)**.

- **Forgejo** (primary): `https://git.infra.centralcloud.com/flakecache/<repo>/actions`
- **Git remotes**: `origin` fetch = Forgejo; push = Forgejo + GitHub mirror

GitHub Actions no longer runs from this directory. The GitHub mirror is for visibility and external clones only.
