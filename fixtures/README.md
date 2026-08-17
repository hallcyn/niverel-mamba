# Golden fixtures

Three tiers, matching brief section 11.1.

| fixture | shape | purpose |
|---|---|---|
| `tiny` | B=1 L=8 D=16 N=4 headdim=8 | float64 mathematical ground truth |
| `segmented` | B=2 L=257, irregular `seq_idx` | strict reset and internal padding |
| `niverel` | one real Foundation V3 Mamba2 block | the checkpoint that matters |

## What is committed and what is not

**Committed:** `golden-manifests/*.json` — configuration, seeds, provenance and
a SHA-256 for every tensor.

**Not committed:** the tensor blobs themselves (`*.safetensors`). They are
regenerated deterministically:

```bash
python scripts/make_golden_fixture.py             # tiny + segmented
python scripts/make_golden_fixture.py --niverel   # + the real V3 block
```

The manifests are what reviewers actually need: a per-tensor digest change is
visible in a diff, whereas a binary change would not be.

## The Niverel fixture

`--niverel` downloads `ckpt.pt` (~1.7 GB) from the private repository
`thibaud-perrin/niverel-5b-v3-hnet-jepa-seed1337`, pinned to revision
`5da95e264026d80fd6d8debb50c4ca4c40277483`. It needs `HF_TOKEN` in `.env`.

Before the checkpoint is read at all, its SHA-256 is verified against the value
already recorded in the repository's own `bundle_manifest.json`
(`8212105328253ab727496ac0514baebf8d7bd55ca9e1be5fc0393d83cea3fa53`).

Only **one** Mamba2 block (~15 MB) is extracted. The full checkpoint is never
committed and never copied into the repository. The checkpoint contains 16
qualifying blocks: 4 encoder, 8 decoder, and 4 belonging to the frozen JEPA EMA
teacher.

Tests that need this fixture skip **by name** when it is absent:

```
SKIPPED - run: python scripts/make_golden_fixture.py --niverel  (needs HF_TOKEN)
```

They are never silently passed over.
