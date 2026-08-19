"""niverel-mamba -- a portable, verifiable, multi-backend Mamba2 runtime.

One weight contract, several backends, and an explicit certification status
for each. A Mamba2 checkpoint should not be a prisoner of the CUDA backend it
was trained with.

Importing this package downloads nothing, spawns no subprocess and compiles
nothing. ``torch`` and ``mlx`` are optional extras and are
only imported when a backend that needs them is actually built.

    >>> from niverel_mamba import Mamba2Config, describe
    >>> config = Mamba2Config(d_model=768, d_state=128, d_conv=4, expand=2)
    >>> config.d_inner, config.nheads
    (1536, 24)
"""

from __future__ import annotations

from .capabilities import Capability, Certification, Environment, detect_environment
from .config import Mamba2Config
from .errors import (
    AssetVerificationError,
    BackendError,
    BackendUnavailableError,
    CertificationError,
    ContractVersionError,
    DtypeMismatchError,
    FixtureError,
    InvalidSeqIdxError,
    MissingKeysError,
    NiverelMambaError,
    ShapeMismatchError,
    UnexpectedKeysError,
    UnknownBackendError,
    UnsupportedConfigError,
    UnsupportedDeviceError,
    UnsupportedDtypeError,
    WeightContractError,
)
from .registry import BACKENDS, backend_status, list_backends, resolve
from .runtime import Runtime, describe, load_mamba2
from .schema import SCHEMA_VERSION, WeightContract, default_contract, load_contract
from .version import __version__
from .weights import validate_state_dict

__all__ = [
    "BACKENDS",
    "SCHEMA_VERSION",
    "AssetVerificationError",
    "BackendError",
    "BackendUnavailableError",
    "Capability",
    "Certification",
    "CertificationError",
    "ContractVersionError",
    "DtypeMismatchError",
    "Environment",
    "FixtureError",
    "InvalidSeqIdxError",
    "Mamba2Config",
    "MissingKeysError",
    "NiverelMambaError",
    "Runtime",
    "ShapeMismatchError",
    "UnexpectedKeysError",
    "UnknownBackendError",
    "UnsupportedConfigError",
    "UnsupportedDeviceError",
    "UnsupportedDtypeError",
    "WeightContract",
    "WeightContractError",
    "__version__",
    "backend_status",
    "default_contract",
    "describe",
    "detect_environment",
    "list_backends",
    "load_contract",
    "load_mamba2",
    "resolve",
    "validate_state_dict",
]
