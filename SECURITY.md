# Security policy

## Reporting a vulnerability

Report security issues privately through GitHub's security advisory form on
<https://github.com/hallcyn/niverel-mamba/security/advisories/new>. Please do
not open a public issue for a vulnerability.

We aim to acknowledge a report within five working days.

## Supply-chain guarantees

These properties are enforced by tests in `tests/release/`, not merely
promised:

| Guarantee | Enforced by |
|---|---|
| Importing the package downloads nothing | `test_import_performs_no_network_or_subprocess` (audit hook) |
| Importing spawns no subprocess | same |
| Importing compiles nothing | no build step exists at import |
| Importing pulls in neither torch nor MLX | `test_import_pulls_in_neither_torch_nor_mlx` |
| The CLI's `--help` imports no framework | `test_cli_help_does_not_import_frameworks` |
| No large binary is committed | `test_no_large_binaries_are_tracked` |
| `.env` (which holds the HF token) is never tracked | `test_env_is_gitignored` |

## Binary artefacts

`niverel-mamba` itself is a pure-Python wheel. The CUDA extensions it can
install (`mamba-ssm`, `causal-conv1d`) are **never** bundled with it and are
**never** compiled on a user's machine.

`niverel-mamba install-backend cuda`:

1. prints a plan and exits — installation requires an explicit `--yes`;
2. fetches a build manifest (`niverel-mamba-binary-manifest-v1`) which records,
   for every artefact, its SHA-256, the source repository and commit it was
   built from, and the workflow that produced it;
3. verifies the downloaded wheel's SHA-256 against that manifest **before**
   installing, and aborts on any mismatch;
4. refuses, with an explanation, if no certified prebuilt wheel matches the
   environment. It does not fall back to building from source.

## Weight loading

Weights are validated against a versioned contract before use. A missing key,
an unexpected key, a differing shape, an incompatible configuration or an
unknown contract version is refused. `strict=False` is not offered.

Loading a checkpoint executes `torch.load`, which is not a safe operation on
untrusted input; prefer `safetensors` for anything you did not produce
yourself. The golden fixtures shipped with this project are safetensors.

## Supported versions

Only the latest minor release receives security fixes while the project is
pre-1.0.
