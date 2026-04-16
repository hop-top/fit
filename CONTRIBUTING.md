# Contributing to fit

## Setup

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Install [Task](https://taskfile.dev): `brew install go-task`

## Development

```bash
task check        # lint + test all ports
task test:py      # python only
task test:go      # go only
```

## Commits

Conventional Commits:

```
feat(go): add remote advisor implementation
fix(ts): handle empty advice in session loop
docs(spec): clarify reward breakdown weights
chore: update taskfile with php lint
```

## PRs

- One logical change per PR
- All checks must pass (`task check`)
- New features need tests
- Spec changes need updates in all ports

## Code style

- Go: `gofmt`, `go vet`
- TypeScript: strict mode, ESM
- Python: ruff, type annotations
- Rust: clippy, rustfmt
- PHP: phpstan, PSR-12
