# Brief — `niverel-mamba`

## 1. Mission

Créer un runtime Mamba2 portable, vérifiable et multi-backend permettant de charger les **mêmes poids Mamba2** sur :

```text
Linux + NVIDIA CUDA
Linux CPU
macOS Apple Silicon + PyTorch MPS
macOS Apple Silicon + MLX
```

Le package doit résoudre le problème suivant :

> Un checkpoint Mamba2 ne devrait pas être prisonnier du backend CUDA avec lequel il a été entraîné.

Le package doit permettre à Niverel et à d’autres projets de conserver un seul contrat de poids tout en choisissant le backend adapté au matériel disponible.

Nom recommandé :

```text
Repository GitHub : Hallcyn/niverel-mamba
Package PyPI      : niverel-mamba
Import Python     : niverel_mamba
CLI               : niverel-mamba
```

Avant publication, réserver le nom sur TestPyPI et PyPI. Je n’ai pas trouvé de package existant portant précisément ce nom dans la recherche actuelle, mais cela doit être confirmé au moment de créer le Trusted Publisher.

---

## 2. Positionnement

`niverel-mamba` n’est pas un nouveau modèle et ne modifie pas les équations de Mamba2.

Il fournit :

```text
Mamba2 Weight Contract
        ↓
backend torch-reference
backend cuda-reference
backend mlx
        ↓
certification numérique inter-backends
```

Le backend CUDA utilisé par Foundation V3 reste la référence historique :

```text
mamba-ssm      2.3.2.post1
causal-conv1d  1.6.2.post1
PyTorch        2.11.0+cu128
Python         3.12
Linux x86_64
CUDA
```

Le projet amont `mamba-ssm` se présente toujours comme une implémentation hardware-aware destinée principalement à Linux avec GPU NVIDIA/CUDA, et demande souvent `--no-build-isolation` pour que sa compilation utilise le bon PyTorch CUDA. ([GitHub][3])

---

## 3. Décisions structurantes non négociables

### 3.1 Un seul contrat de poids

Les poids ne dépendent jamais du backend.

Le format canonique doit reprendre les noms et formes de tenseurs de la version Mamba2 réellement utilisée par Niverel :

```text
in_proj.weight
in_proj.bias                 si présent
conv1d.weight
conv1d.bias
dt_bias
A_log
D
norm.weight
out_proj.weight
out_proj.bias                si présent
init_states                  si activé
```

Le contrat exact ne doit pas être écrit à la main depuis cette liste indicative. Il doit être **extrait automatiquement** :

1. depuis un module `mamba_ssm.Mamba2` 2.3.2.post1 ;
2. depuis un vrai checkpoint Niverel Foundation V3 ;
3. puis scellé dans une fixture versionnée.

Le contrat doit inclure :

```json
{
  "schema_version": "niverel-mamba2-weights-v1",
  "upstream_package": "mamba-ssm",
  "upstream_version": "2.3.2.post1",
  "configuration": {
    "d_model": 768,
    "d_state": 128,
    "d_conv": 4,
    "expand": 2,
    "headdim": "...",
    "ngroups": "...",
    "chunk_size": "...",
    "bias": "...",
    "conv_bias": "...",
    "norm_epsilon": "..."
  },
  "tensors": {
    "in_proj.weight": {
      "shape": ["..."],
      "dtype": "float32"
    }
  }
}
```

Les valeurs marquées `...` doivent être extraites du runtime réel, pas supposées depuis la branche `main` actuelle de Mamba.

### 3.2 Chargement strict

Tous les backends doivent refuser :

```text
clé manquante
clé inattendue
forme différente
configuration incompatible
version de contrat inconnue
conversion implicite non documentée
```

Le chargement doit être équivalent à :

```python
load_state_dict(state_dict, strict=True)
```

Pour MLX, où les noms peuvent être transformés, la conversion doit être réversible :

```text
PyTorch upstream state_dict
        ↓
canonical weights
        ↓
MLX weights
        ↓
canonical weights
        ↓
PyTorch upstream state_dict

hashes et tenseurs identiques
```

### 3.3 Aucun fallback silencieux

Une demande explicite :

```python
backend="cuda"
```

doit échouer si la wheel compatible n’est pas installée.

Elle ne doit jamais se transformer silencieusement en backend CPU.

Le mode `auto` peut choisir un backend, mais doit retourner son identité et son statut :

```json
{
  "backend": "torch-reference",
  "framework": "torch",
  "device": "mps",
  "certification": "numerically-certified",
  "official_reference": false
}
```

### 3.4 Pas de promesse « bit-for-bit » entre matériels

On peut garantir :

```text
mêmes poids
mêmes équations
même ordre logique des opérations
équivalence numérique mesurée
```

On ne doit pas promettre :

```text
mêmes bits entre CUDA BF16, CPU FP32, MPS et MLX
```

Les réductions, fusions et arrondis diffèrent entre backends.

Les statuts officiels doivent être :

```text
reference
    backend exact utilisé pour produire ou certifier le résultat

numerically-certified
    backend comparé au reference sous tolérances scellées

experimental
    backend fonctionnel mais pas encore certifié

unsupported
    combinaison explicitement refusée
```

---

# 4. Architecture générale

```text
                         canonical Mamba2 config
                                   +
                         canonical weight contract
                                   |
             ┌─────────────────────┼─────────────────────┐
             |                     |                     |
             v                     v                     v
     torch-reference       cuda-reference             mlx
     CPU / CUDA / MPS      upstream kernels      Apple Silicon
             |                     |                     |
             └─────────────────────┼─────────────────────┘
                                   |
                         certification reports
                                   |
                       Niverel / Niverel Lab
```

## 4.1 Attention particulière à MLX

Le backend MLX ne peut pas être injecté directement dans le H-Net PyTorch actuel.

Faire :

```text
PyTorch H-Net
→ convertir vers MLX avant chaque bloc Mamba
→ calcul MLX
→ reconvertir vers PyTorch
```

serait beaucoup trop coûteux et fragile.

Deux chemins Mac doivent donc rester distincts :

```text
Chemin immédiat
Niverel PyTorch complet
+ Mamba2 torch-reference
+ CPU ou MPS

Chemin futur optimisé
Niverel entièrement porté vers MLX
+ Mamba2 MLX
+ routeur / Transformer / chunking / head également MLX
```

`niverel-mamba` fournit le bloc Mamba2 MLX et la conversion des poids, mais **le port complet de H-Net vers MLX est un projet séparé**.

---

# 5. Organisation du repository

```text
niverel-mamba/
├── pyproject.toml
├── README.md
├── LICENSE
├── NOTICE
├── THIRD_PARTY_NOTICES.md
├── CHANGELOG.md
├── SECURITY.md
├── CONTRIBUTING.md
├── uv.lock
│
├── src/
│   └── niverel_mamba/
│       ├── __init__.py
│       ├── version.py
│       ├── errors.py
│       ├── config.py
│       ├── capabilities.py
│       ├── weights.py
│       ├── schema.py
│       ├── runtime.py
│       ├── registry.py
│       │
│       ├── backends/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── torch_reference.py
│       │   ├── cuda_reference.py
│       │   └── mlx.py
│       │
│       ├── torch_ops/
│       │   ├── causal_conv.py
│       │   ├── ssd_sequential.py
│       │   ├── ssd_chunked.py
│       │   ├── gated_rmsnorm.py
│       │   ├── state.py
│       │   └── mamba2.py
│       │
│       ├── mlx_ops/
│       │   ├── causal_conv.py
│       │   ├── ssd.py
│       │   ├── gated_rmsnorm.py
│       │   ├── state.py
│       │   └── mamba2.py
│       │
│       ├── adapters/
│       │   ├── upstream.py
│       │   └── niverel.py
│       │
│       ├── certification/
│       │   ├── golden.py
│       │   ├── compare.py
│       │   ├── tolerances.py
│       │   └── report.py
│       │
│       └── cli/
│           ├── main.py
│           ├── doctor.py
│           ├── install_backend.py
│           ├── inspect.py
│           └── verify.py
│
├── schemas/
│   ├── mamba2-upstream-2.3.2.post1.json
│   └── backend-manifest-v1.schema.json
│
├── fixtures/
│   ├── tiny/
│   ├── segmented/
│   └── golden-manifests/
│
├── tests/
│   ├── unit/
│   ├── contract/
│   ├── torch_reference/
│   ├── cuda/
│   ├── mps/
│   ├── mlx/
│   ├── gradients/
│   ├── niverel/
│   └── release/
│
├── scripts/
│   ├── extract_weight_contract.py
│   ├── build_upstream_cuda_wheels.py
│   ├── make_golden_fixture.py
│   ├── verify_release_assets.py
│   └── generate_simple_index.py
│
├── docker/
│   ├── torch211-cu128.Dockerfile
│   ├── torch212-cu130.Dockerfile
│   └── torch213-cu130.Dockerfile
│
└── .github/
    └── workflows/
        ├── ci-core.yml
        ├── ci-torch-matrix.yml
        ├── ci-macos-mps.yml
        ├── ci-mlx.yml
        ├── build-cuda-wheels.yml
        ├── certify-cuda-sm80.yml
        ├── certify-cuda-sm90.yml
        ├── nightly-upstream.yml
        ├── release.yml
        └── publish-pypi.yml
```

---

# 6. API publique

## 6.1 Configuration

```python
from niverel_mamba import Mamba2Config

config = Mamba2Config(
    d_model=768,
    d_state=128,
    d_conv=4,
    expand=2,
    headdim=64,
    ngroups=1,
    chunk_size=256,
    bias=False,
    conv_bias=True,
)
```

Toutes les valeurs doivent être sérialisables.

```python
payload = config.to_dict()
config = Mamba2Config.from_dict(payload)
```

## 6.2 Backend PyTorch

```python
from niverel_mamba.torch import Mamba2

model = Mamba2(
    config,
    backend="reference",
    device="mps",
    dtype=torch.float32,
)

model.load_state_dict(weights, strict=True)

y = model(x, seq_idx=seq_idx)
```

Le module doit être un vrai `torch.nn.Module`, afin de pouvoir remplacer `mamba_ssm.Mamba2` dans le H-Net existant sans porter le reste de Niverel.

## 6.3 Backend CUDA

```python
from niverel_mamba.torch import Mamba2

model = Mamba2(
    config,
    backend="cuda-reference",
    device="cuda",
    dtype=torch.bfloat16,
)
```

Ce backend encapsule le `Mamba2` upstream certifié.

## 6.4 Backend MLX

```python
from niverel_mamba.mlx import Mamba2

model = Mamba2(config)
model.load_canonical_weights(weights)

y = model(x, seq_idx=seq_idx)
```

Il ne doit pas imiter artificiellement `torch.nn.Module`.

## 6.5 Diagnostic

```bash
niverel-mamba doctor
```

Exemple :

```text
niverel-mamba 0.1.0

Python          3.12.13
Platform        macOS arm64
Framework       torch 2.13.0
MPS             available
MLX             0.32.0
CUDA            unavailable

Available backends:
  torch-reference   yes   numerically-certified
  cuda-reference    no    CUDA backend not installed
  mlx               yes   experimental

Recommended backend:
  torch-reference / mps
```

## 6.6 Installation d’un backend CUDA

```bash
niverel-mamba install-backend cuda
```

Par défaut, la commande doit seulement afficher le plan :

```text
Detected:
  Python 3.12
  Torch 2.13.0
  Torch CUDA 13.0
  Linux x86_64
  GPU capability sm_90

Required artifacts:
  causal_conv1d ...
  mamba_ssm ...

Checksums verified.
Run again with --yes to install.
```

Puis :

```bash
niverel-mamba install-backend cuda --yes
```

Aucun téléchargement ou installation ne doit se produire à l’import du package.

---

# 7. Backend `torch-reference`

## Objectif

Fournir l’implémentation de référence portable, pure PyTorch :

```text
Linux CPU
Linux CUDA sans kernels Mamba
macOS CPU
macOS MPS
```

Ce backend privilégie la fidélité avant les performances.

## Implémentation mathématique

Le forward doit conserver l’ordre upstream :

```text
u
↓
in_proj
↓
split [z, x, B, C, dt]
↓
depthwise causal conv1d sur [x, B, C]
↓
SiLU
↓
dt = softplus(dt + dt_bias)
A  = -exp(A_log)
↓
SSD / recurrence
↓
D skip
↓
gated RMSNorm
↓
out_proj
```

L’upstream fournit une implémentation minimale SSD correspondant au papier, mais même ses fichiers « simple » importent encore des opérations Triton au niveau du module. Il faut donc reprendre uniquement les équations pures, sans importer les modules Triton. 

Deux implémentations doivent exister.

### Oracle séquentiel

```text
ssd_sequential.py
```

* boucle explicite sur les positions ;
* mémoire O(state) ;
* lent ;
* lisible ;
* source de vérité pour les tests.

### Implémentation chunkée

```text
ssd_chunked.py
```

* algorithme SSD chunké ;
* opérations PyTorch vectorisées ;
* pas de matrice globale L×L ;
* padding interne de la séquence au multiple du chunk ;
* suppression du padding à la sortie ;
* compatible autograd.

L’oracle séquentiel ne doit jamais être supprimé, même lorsque le backend chunké devient plus rapide.

## `seq_idx` et strict reset

C’est un invariant critique pour Niverel.

À chaque changement de `seq_idx`, le backend doit réinitialiser :

```text
état de convolution causale
état SSM
contexte inter-position
```

Tests requis :

```python
y_segmented = model(x, seq_idx=segments)

y_separate = concat(
    model(doc_1),
    model(doc_2),
    ...
)

assert_close(y_segmented, y_separate)
```

Le test doit couvrir :

```text
frontière à la première position
documents de longueur 1
frontières consécutives
plusieurs documents par batch
padding terminal
L8192
```

## API stateful

Le backend doit exposer :

```python
state = model.allocate_inference_state(batch_size=1)
y_t, state = model.step(x_t, state)
```

Et garantir :

```text
concat(step(x_t))
≈
forward(sequence)
```

Le state doit inclure :

```text
conv_state
ssm_state
position / segment identity
```

## Entraînement

Pour la version 0.1 :

```text
forward inference     supporté
autograd forward      probablement fonctionnel
training certifié     non
```

La capability doit être honnête :

```json
{
  "inference": true,
  "backward": "experimental",
  "training": false
}
```

Le statut `training=true` ne pourra être activé qu’après comparaison des gradients contre CUDA.

---

# 8. Backend `cuda-reference`

## Objectif

Fournir le backend le plus proche du runtime d’entraînement.

Version de référence Niverel :

```text
mamba-ssm      2.3.2.post1
causal-conv1d  1.6.2.post1
```

Le projet upstream utilise une implémentation hardware-aware, avec chemins fusionnés, causal convolution spécialisée et opérations Triton/CUDA. Son packaging calcule la wheel à partir de dimensions telles que la version de PyTorch, CUDA/HIP, Python, plateforme et ABI C++11. ([GitHub][4])

## Stratégie de fork

Ne pas commencer par maintenir un fork divergent de toute l’architecture.

Créer d’abord :

```text
Hallcyn/niverel-mamba
```

et utiliser une révision upstream pinée.

Seulement si des patchs de packaging sont nécessaires, créer :

```text
Hallcyn/mamba
```

avec les règles suivantes :

```text
origin-upstream = state-spaces/mamba
origin          = Hallcyn/mamba

branche upstream-sync
branche niverel-packaging
```

Les patchs doivent rester limités à :

```text
build
wheel naming
CI
compatibility guards
bug fixes nécessaires au chargement
```

Aucune modification mathématique ne doit être mélangée à un patch de packaging.

Le code amont Mamba est sous licence Apache 2.0. Les fichiers repris ou modifiés doivent conserver les notices, inclure la licence et signaler les modifications. 

## Wheels initiales

Matrice minimale recommandée :

| Runtime      | Python | CUDA | Architectures | Statut                |
| ------------ | -----: | ---: | ------------- | --------------------- |
| Torch 2.11.0 |   3.12 | 12.8 | sm80, sm90    | référence Niverel     |
| Torch 2.12.1 |   3.12 | 13.0 | sm80, sm90    | support stable        |
| Torch 2.13.0 |   3.12 | 13.0 | sm80, sm90    | support stable actuel |

PyTorch 2.12 propose officiellement des builds CUDA 12.6, 13.0 et 13.2 ; PyTorch 2.13 est la stable actuelle. CUDA 13.0 reste donc la cible raisonnable pour les builds modernes, tandis que 2.11/cu128 reste conservé spécifiquement pour reproduire Foundation V3. ([PyTorch][5])

Ne pas lancer immédiatement cette explosion :

```text
3 versions Python
× 3 versions Torch
× 3 CUDA
× 3 profils GPU
× 2 packages natifs
```

Commencer avec trois environnements certifiés.

Ensuite ajouter, selon la demande :

```text
Python 3.11
Python 3.13
CUDA 12.6
CUDA 13.2
sm86 / sm89
Blackwell
Linux aarch64
```

## Wheel manifest

Chaque asset doit être accompagné d’un manifest :

```json
{
  "schema_version": "niverel-mamba-binary-manifest-v1",
  "package": "mamba-ssm",
  "package_version": "2.3.2.post1",
  "torch_version": "2.13.0",
  "torch_cuda": "13.0",
  "python_tag": "cp312",
  "platform": "manylinux_2_28_x86_64",
  "cxx11_abi": true,
  "architectures": ["sm_80", "sm_90"],
  "sha256": "...",
  "source_repository": "...",
  "source_commit": "...",
  "build_workflow": "...",
  "created_at_utc": "..."
}
```

Le CLI télécharge d’abord ce manifest puis vérifie le SHA de la wheel.

---

# 9. Backend MLX

## Version 0.1 MLX

Implémentation en opérations MLX standards :

```text
Linear
depthwise causal conv
SSD chunké
gated RMSNorm
Linear
```

MLX est disponible sur PyPI pour Apple Silicon avec Python natif ≥3.10 et macOS ≥14. ([ML Explore][6])

Pin initial recommandé :

```toml
mlx = ">=0.32,<0.33"
```

Ne pas utiliser une dépendance sans borne supérieure tant que la certification n’est pas automatisée.

## Version 0.2 MLX

Optimisation via kernels Metal personnalisés :

```text
causal conv fusionnée
scan / SSD fusionné
gated RMSNorm fusionné
```

MLX permet d’écrire des kernels Metal personnalisés depuis ses API Python ou C++, avec compilation JIT de bibliothèques Metal. ([ML Explore][7])

Mais l’ordre est impératif :

```text
pure MLX correcte
        ↓
certification numérique
        ↓
profilage
        ↓
kernels Metal optimisés
```

Ne jamais optimiser avant d’avoir une référence fiable.

## Licence

MLX est sous licence MIT. Tout code repris doit conserver sa notice. 

---

# 10. Matrice de support publique

## Version 0.1

| Backend         | Torch        | Python    | OS              | Device    | Certification              |
| --------------- | ------------ | --------- | --------------- | --------- | -------------------------- |
| torch-reference | 2.11         | 3.10–3.13 | Linux           | CPU       | certifié                   |
| torch-reference | 2.12         | 3.10–3.13 | Linux           | CPU       | certifié                   |
| torch-reference | 2.13         | 3.10–3.13 | Linux           | CPU       | certifié                   |
| torch-reference | 2.11         | 3.10–3.13 | macOS arm64     | CPU/MPS   | certifié                   |
| torch-reference | 2.12         | 3.10–3.13 | macOS arm64     | CPU/MPS   | certifié                   |
| torch-reference | 2.13         | 3.10–3.13 | macOS arm64     | CPU/MPS   | certifié                   |
| cuda-reference  | 2.11.0+cu128 | 3.12      | Linux x86_64    | sm80/sm90 | référence                  |
| cuda-reference  | 2.12.1+cu130 | 3.12      | Linux x86_64    | sm80/sm90 | certifié                   |
| cuda-reference  | 2.13.0+cu130 | 3.12      | Linux x86_64    | sm80/sm90 | certifié                   |
| mlx             | N/A          | 3.10–3.13 | macOS arm64 ≥14 | Apple GPU | expérimental puis certifié |

PyTorch 2.13 peut supporter davantage de versions Python, mais la matrice native du projet doit rester volontairement plus petite au départ. L’objectif est une matrice réellement testée, pas une déclaration théorique.

## Non supporté en 0.1

```text
Windows CUDA
ROCm
Intel XPU
Jetson
musl / Alpine
PyTorch nightly
Python 3.14/3.15 pour les extensions CUDA
entraînement MLX
```

Ces plateformes peuvent être ajoutées ensuite, mais doivent être annoncées comme non supportées tant qu’elles ne sont pas certifiées.

---

# 11. Certification numérique

## 11.1 Trois niveaux de fixtures

### Fixture mathématique minuscule

```text
B=1
L=8
D=16
N=4
headdim=8
```

Exécutée en float64 lorsque possible.

Compare :

```text
oracle séquentiel
torch chunké
CUDA
MLX
```

### Fixture structurée

```text
B=2
L=257
plusieurs seq_idx
frontières irrégulières
```

Vérifie les resets et le padding interne.

### Fixture Niverel réelle

Extraire un bloc Mamba2 réel depuis le checkpoint primaire :

```text
niverel-5b-v3-hnet-jepa-seed1337
```

Ne pas versionner le checkpoint complet dans Git.

Versionner :

```text
config
hash du checkpoint source
poids du bloc extrait ou fixture réduite
entrée déterministe
sorties CUDA de référence
états intermédiaires
rapport de tolérance
```

Si la fixture est trop grosse pour Git, la publier dans une release GitHub ou un petit repo Hugging Face dédié avec SHA piné.

## 11.2 Tests obligatoires

```text
forward complet
step autoregressif
forward == concat(step)
causal conv
gated RMSNorm
SSD
seq_idx reset
batching
padding
dtypes
state_dict round-trip
conversion MLX round-trip
```

Si l’entraînement est annoncé :

```text
gradient entrée
gradient in_proj
gradient conv1d
gradient A_log
gradient D
gradient out_proj
```

## 11.3 Tolérances

Les tolérances doivent être observées puis scellées dans :

```text
certification/tolerances.yaml
```

Valeurs de départ possibles, à ne pas élargir automatiquement :

```yaml
cpu_float64:
  atol: 1.0e-10
  rtol: 1.0e-9

cpu_float32:
  atol: 2.0e-5
  rtol: 2.0e-4

cuda_bfloat16:
  atol: 2.0e-2
  rtol: 2.0e-2

mps_float32:
  atol: 1.0e-4
  rtol: 1.0e-3

mlx_float32:
  atol: 1.0e-4
  rtol: 1.0e-3
```

Ce sont des points de départ. Un agent ne doit pas simplement relever la tolérance jusqu’à ce que le test passe.

Chaque rapport doit inclure :

```json
{
  "reference_backend": "cuda-reference",
  "candidate_backend": "torch-reference-mps",
  "max_abs_error": 0.00007,
  "max_rel_error": 0.0009,
  "mean_abs_error": 0.000002,
  "cosine_similarity": 0.999999,
  "forward_passed": true,
  "step_passed": true,
  "segment_reset_passed": true
}
```

---

# 12. CI GitHub Actions

## `ci-core.yml`

Déclenchée à chaque PR.

```text
ruff
mypy/pyright
pytest sans framework optionnel
build sdist
build py3-none-any wheel
twine check
license check
```

## `ci-torch-matrix.yml`

Matrice Linux CPU :

```text
Python 3.10 / Torch 2.11
Python 3.11 / Torch 2.11
Python 3.12 / Torch 2.11

Python 3.10 / Torch 2.12
Python 3.11 / Torch 2.12
Python 3.12 / Torch 2.12

Python 3.10 / Torch 2.13
Python 3.11 / Torch 2.13
Python 3.12 / Torch 2.13
Python 3.13 / Torch 2.13
```

Les versions réellement impossibles à installer doivent être retirées explicitement, pas masquées avec `continue-on-error`.

## `ci-macos-mps.yml`

Utiliser :

```text
GitHub macOS arm64 M2 runner
ou
runner Mac auto-hébergé
```

GitHub propose des runners macOS arm64 M2, mais ce sont des larger runners facturés. ([GitHub Docs][8])

Tests :

```text
torch-reference CPU
torch-reference MPS
CPU ↔ MPS
forward
step
seq_idx
Niverel fixture
```

## `ci-mlx.yml`

Sur Apple Silicon :

```text
installation mlx
tests pure MLX
conversion des poids
MLX ↔ torch-reference CPU
```

## `build-cuda-wheels.yml`

Déclenchement :

```text
workflow_dispatch
tag release
modification du manifeste de build
```

Compilation dans des images de développement CUDA pinées.

Une GPU physique n’est pas nécessaire pour produire tout code CUDA si le toolkit et `nvcc` sont présents, mais les wheels ne doivent jamais être publiées sans test ultérieur sur un vrai GPU.

Optimisations de build :

```text
Ninja
MAX_JOBS explicite
ccache/sccache
cache des sources
Docker layer cache
TORCH_CUDA_ARCH_LIST limité
pas de build de toutes les architectures NVIDIA
```

## `certify-cuda-sm80.yml`

Runner :

```text
self-hosted
linux
x64
cuda
sm80
```

Teste A100 ou équivalent.

## `certify-cuda-sm90.yml`

Runner :

```text
self-hosted
linux
x64
cuda
sm90
```

Teste H100.

Les runners auto-hébergés peuvent être ciblés par labels OS et architecture, et rester hors ligne entre deux campagnes. ([GitHub Docs][9])

## `nightly-upstream.yml`

Non bloquant pour les PR.

Teste :

```text
dernière patch release Torch 2.11
dernière patch release Torch 2.12
stable Torch 2.13
nightly Torch suivante
dernière version mamba-ssm upstream
```

Un échec nightly crée automatiquement une issue mais ne casse pas la release stable existante.

## `release.yml`

Ordre :

```text
build core
test core
build CUDA assets
cold-install CUDA assets
certify GPU
build final manifest
create GitHub Release
publish PyPI
cold-install PyPI package
verify full install recipes
```

---

# 13. Publication PyPI

## Ce qui doit être publié sur PyPI

Package léger :

```text
niverel-mamba
```

Contenu :

```text
API
config
weight contract
torch-reference
MLX source
CUDA adapter
CLI
certification utilities
```

Pas les grosses extensions CUDA.

Installation de base :

```bash
pip install niverel-mamba
```

Extras :

```bash
pip install "niverel-mamba[torch]"
pip install "niverel-mamba[mlx]"
pip install "niverel-mamba[dev]"
```

`torch` ne doit pas être une dépendance obligatoire du package de base, sinon un utilisateur MLX téléchargerait inutilement PyTorch.

Exemple de `pyproject.toml` :

```toml
[project]
name = "niverel-mamba"
version = "0.1.0"
requires-python = ">=3.10"
license = { text = "Apache-2.0" }

dependencies = [
  "packaging>=24",
  "typing-extensions>=4.12",
]

[project.optional-dependencies]
torch = [
  "torch>=2.11,<2.14",
  "einops>=0.8",
]
mlx = [
  "mlx>=0.32,<0.33; platform_system == 'Darwin' and platform_machine == 'arm64'",
]
dev = [
  "pytest",
  "pytest-xdist",
  "ruff",
  "mypy",
  "build",
  "twine",
]
```

Pour les environnements CUDA certifiés, la documentation doit recommander d’installer PyTorch explicitement avant `niverel-mamba`, car les variantes CUDA de PyTorch utilisent des index et des distributions distinctes.

## Trusted Publishing

Utiliser le Trusted Publishing PyPI via GitHub Actions, sans token PyPI persistant.

Workflow :

```yaml
permissions:
  id-token: write

environment:
  name: pypi

steps:
  - uses: pypa/gh-action-pypi-publish@release/v1
```

PyPI documente cette méthode OIDC et recommande l’utilisation d’un environnement GitHub dédié. ([PyPI Docs][10])

Séquence :

```text
TestPyPI
    ↓
installation à froid
    ↓
tests smoke
    ↓
PyPI production
    ↓
installation à froid
```

## Où publier les wheels CUDA

Recommandation :

```text
GitHub Releases
        +
manifest JSON signé/hashé
        +
CLI d’installation
```

Éventuellement ajouter un index PEP 503 statique :

```text
https://.../simple/
```

Mais ne pas commencer par héberger toutes les wheels sur PyPI :

1. les fichiers dépassent le plafond par défaut de 100 Mo ;
2. l’ensemble de la matrice peut dépasser rapidement 10 Go ;
3. les tags wheels ne codent pas la version Torch/CUDA ;
4. pip pourrait sélectionner une wheel ABI-incompatible si plusieurs builds semblent équivalents. ([PyPI Docs][2])

Une augmentation de quota PyPI peut être demandée plus tard, mais elle ne résout pas le problème de sélection Torch/CUDA.

---

# 14. Versioning

## Package

```text
0.1.0
```

* torch-reference initial ;
* CUDA Niverel 2.11/cu128 ;
* contrat poids v1.

```text
0.2.0
```

* Torch 2.12 et 2.13 CUDA ;
* MPS certifié.

```text
0.3.0
```

* MLX fonctionnel et certifié.

```text
1.0.0
```

* API stable ;
* contrat de poids stable ;
* trois backends certifiés ;
* installation CUDA sans compilation locale.

## Sémantique

```text
MAJOR
  rupture API publique ou contrat de poids

MINOR
  nouveau backend
  nouvelle version PyTorch
  nouvelle plateforme
  nouvelle capability

PATCH
  correctif sans changement de contrat
  optimisation numériquement compatible
  documentation
```

Les wheels binaires doivent partager la version du package mais utiliser un build ID dans leur manifest.

---

# 15. Sécurité et supply chain

Obligatoire :

```text
aucun téléchargement à l’import
aucun subprocess implicite à l’import
aucune compilation implicite à l’import
SHA-256 de chaque asset
manifest de build
source commit piné
cold installation après publication
Trusted Publishing PyPI
permissions GitHub minimales
```

Recommandé :

```text
attestation GitHub des artefacts
SBOM CycloneDX
pip-audit
Dependabot
CodeQL
signatures de tags
branche main protégée
release uniquement depuis tag
```

Le repository doit contenir :

```text
LICENSE                   Apache-2.0
NOTICE
THIRD_PARTY_NOTICES.md
```

---

# 16. Intégration avec Niverel

L’intégration future ne doit pas modifier les checkpoints Foundation V3.

Dans Niverel :

```python
from niverel_mamba.torch import Mamba2
```

remplace conditionnellement :

```python
from mamba_ssm import Mamba2
```

avec une factory :

```python
def build_mamba2(config, backend):
    ...
```

Backends possibles :

```text
upstream-cuda
niverel-torch
```

Le `state_dict` doit rester identique.

Test d’intégration obligatoire :

```python
upstream = UpstreamMamba2(config)
portable = PortableMamba2(config)

portable.load_state_dict(upstream.state_dict(), strict=True)

assert portable.state_dict().keys() == upstream.state_dict().keys()
```

Pour Niverel Lab sur Mac :

```text
v1 :
  H-Net PyTorch complet
  + niverel-mamba torch-reference
  + MPS/CPU

v2 :
  Niverel H-Net complet porté MLX
  + niverel-mamba MLX
```

---

# 17. Séquencement recommandé

## Phase 0 — extraction du vrai contrat

Livrables :

```text
script d’extraction
schema poids 2.3.2.post1
fixture minuscule
fixture Niverel
rapport des clés/formes/dtypes
```

Gate :

```text
round-trip strict contre mamba-ssm 2.3.2.post1
```

## Phase 1 — oracle pure PyTorch

Livrables :

```text
causal conv
SSD séquentiel
gated RMSNorm
forward Mamba2
seq_idx
step
```

Gate :

```text
oracle cohérent avec équations
forward == step
segments == documents séparés
```

## Phase 2 — torch-reference chunké

Livrables :

```text
SSD chunké
pas de mémoire L² globale
CPU
MPS
```

Gate :

```text
oracle séquentiel == chunké
L8192 passe
```

## Phase 3 — backend CUDA historique

Livrables :

```text
wheels Torch 2.11/cu128/cp312/sm80-sm90
manifest
CLI install
```

Gate :

```text
checkpoint Niverel strict-load
parité CUDA
cold-install sans compiler
```

## Phase 4 — PyTorch modernes

Livrables :

```text
Torch 2.12/cu130
Torch 2.13/cu130
```

Gate :

```text
mêmes fixtures
mêmes rapports
cold-install
```

## Phase 5 — PyPI

Livrables :

```text
TestPyPI
Trusted Publisher
PyPI 0.1.0
documentation install
```

Gate :

```text
installation depuis un environnement vierge
```

## Phase 6 — MLX

Livrables :

```text
conversion poids
forward pure MLX
step MLX
parité
```

Gate :

```text
fixture Niverel certifiée
```

## Phase 7 — Metal optimisé

Livrables :

```text
kernels Metal
benchmarks
fallback pure MLX
```

Gate :

```text
aucune dérive numérique
gain mesuré
```

---

# 18. Critères d’acceptation finaux

Le projet ne peut pas annoncer `1.0` tant que les conditions suivantes ne sont pas toutes vraies :

```text
[ ] checkpoint Mamba2 Niverel charge strictement sur CUDA
[ ] même checkpoint charge strictement sur torch-reference CPU
[ ] même checkpoint charge strictement sur torch-reference MPS
[ ] même checkpoint se convertit et charge sur MLX
[ ] seq_idx strict-reset est certifié
[ ] forward et step sont cohérents
[ ] PyTorch 2.11 est certifié
[ ] PyTorch 2.12 est certifié
[ ] PyTorch 2.13 est certifié
[ ] aucune installation supportée ne compile localement
[ ] package core publié sur PyPI
[ ] PyPI utilise Trusted Publishing
[ ] wheels CUDA publiées avec SHA et manifests
[ ] chaque wheel est installée à froid
[ ] chaque backend publie son statut de certification
[ ] aucune bascule silencieuse vers un autre backend
[ ] Niverel Lab sait afficher le backend réellement utilisé
```

---

# 19. Instruction finale à l’agent

L’agent doit commencer par **Phase 0**, et non par la CI ou les kernels.

La première preuve attendue est :

```text
vrai Mamba2 upstream 2.3.2.post1
        ↓ state_dict
Mamba2 torch-reference
        ↓ strict load
mêmes clés, mêmes formes, sortie numériquement équivalente
```

Il ne doit pas :

```text
réécrire les noms de poids sans contrat
ignorer des clés du checkpoint
utiliser strict=False
présenter le stub Niverel comme backend
annoncer MLX avant la parité
annoncer toutes les versions de Torch comme supportées
publier une wheel non testée
compiler automatiquement sur la machine de l’utilisateur
```

La première release utile et réaliste est :

```text
niverel-mamba 0.1.0

torch-reference:
  Linux CPU
  macOS CPU/MPS
  Torch 2.11 / 2.12 / 2.13

cuda-reference:
  Torch 2.11.0 + CUDA 12.8
  Python 3.12
  sm80 / sm90

PyPI:
  package core

GitHub Release:
  wheels CUDA certifiées
```

Puis `0.2.0` étend les wheels CUDA à Torch 2.12 et 2.13.

**C’est un périmètre ambitieux mais réaliste.** La clé est de traiter la portabilité comme un problème de contrat et de certification, et non comme un simple problème de conversion de tenseurs.

[1]: https://pytorch.org/blog/pytorch-2-13-release-blog/?utm_source=chatgpt.com "PyTorch 2.13 Release Blog – PyTorch"
[2]: https://docs.pypi.org/project-management/storage-limits/?utm_source=chatgpt.com "Storage Limits - PyPI Docs"
[3]: https://github.com/state-spaces/mamba?utm_source=chatgpt.com "GitHub - state-spaces/mamba: Mamba SSM architecture · GitHub"
[4]: https://github.com/state-spaces/mamba/blob/main/setup.py?utm_source=chatgpt.com "mamba/setup.py at main · state-spaces/mamba · GitHub"
[5]: https://pytorch.org/get-started/previous-versions/?utm_source=chatgpt.com "Previous PyTorch Versions"
[6]: https://ml-explore.github.io/mlx/build/html/install.html?utm_source=chatgpt.com "Build and Install — MLX 0.32.0 documentation"
[7]: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html?utm_source=chatgpt.com "Custom Metal Kernels — MLX 0.32.0 documentation"
[8]: https://docs.github.com/en/billing/reference/actions-runner-pricing?utm_source=chatgpt.com "Actions runner pricing - GitHub Docs"
[9]: https://docs.github.com/en/actions/reference/runners/self-hosted-runners?utm_source=chatgpt.com "Self-hosted runners reference - GitHub Docs"
[10]: https://docs.pypi.org/trusted-publishers/using-a-publisher/?utm_source=chatgpt.com "Publishing with a Trusted Publisher - PyPI Docs"
