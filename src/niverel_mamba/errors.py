"""Error hierarchy for :mod:`niverel_mamba`.

Every failure mode the brief calls out has its own exception type. The point is
that a caller can distinguish "you asked for a backend that is not installed"
from "your weights do not match the contract" from "this configuration is not
supported yet" -- and that none of these ever degrade into a silent fallback.
"""

from __future__ import annotations

__all__ = [
    "AssetVerificationError",
    "BackendError",
    "BackendUnavailableError",
    "CertificationError",
    "ContractVersionError",
    "DtypeMismatchError",
    "FixtureError",
    "InvalidSeqIdxError",
    "MissingKeysError",
    "NiverelMambaError",
    "ShapeMismatchError",
    "UnexpectedKeysError",
    "UnknownBackendError",
    "UnsupportedConfigError",
    "UnsupportedDeviceError",
    "UnsupportedDtypeError",
    "WeightContractError",
]


class NiverelMambaError(Exception):
    """Base class for every error raised by this package."""


# --------------------------------------------------------------------------
# Backend resolution
# --------------------------------------------------------------------------


class BackendError(NiverelMambaError):
    """Base class for backend resolution failures."""


class BackendUnavailableError(BackendError):
    """An explicitly requested backend cannot be used on this machine.

    This is raised -- never silently downgraded -- when the caller asks for a
    specific backend (for example ``backend="cuda-reference"``) and the
    required wheel, framework or device is missing. Brief section 3.3.
    """


class UnknownBackendError(BackendError):
    """The requested backend name is not registered."""


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


class UnsupportedConfigError(NiverelMambaError):
    """The configuration is valid upstream but not supported by this backend."""


class UnsupportedDtypeError(NiverelMambaError):
    """The requested dtype cannot be honoured on this device.

    Notably: MPS has no float64. We refuse rather than silently downcast, so
    that a float64 certification run can never be quietly demoted to float32.
    """


class UnsupportedDeviceError(NiverelMambaError):
    """The requested device is unavailable or incompatible with the backend."""


# --------------------------------------------------------------------------
# Weight contract
# --------------------------------------------------------------------------


class WeightContractError(NiverelMambaError):
    """Base class for weight-contract violations.

    Loading is always equivalent to ``load_state_dict(..., strict=True)``:
    a missing key, an unexpected key, a differing shape, an incompatible
    configuration, an unknown contract version, or an undocumented implicit
    conversion are all refused. Brief section 3.2.
    """


class MissingKeysError(WeightContractError):
    """The state dict lacks keys required by the contract."""


class UnexpectedKeysError(WeightContractError):
    """The state dict carries keys the contract does not define."""


class ShapeMismatchError(WeightContractError):
    """A tensor has the right name but the wrong shape."""


class DtypeMismatchError(WeightContractError):
    """A tensor would require an undocumented implicit dtype conversion."""


class ContractVersionError(WeightContractError):
    """The contract's ``schema_version`` is not one this package understands."""


# --------------------------------------------------------------------------
# Runtime invariants
# --------------------------------------------------------------------------


class InvalidSeqIdxError(NiverelMambaError):
    """``seq_idx`` violates the invariant the reset masking relies upon.

    ``seq_idx`` must be non-decreasing along the sequence axis for every batch
    row. The whole chunked strict-reset derivation rests on the equivalence
    "``seq_idx[i] == seq_idx[j]`` (for ``i <= j``) iff no document boundary
    lies in ``(i, j]``", which only holds for non-decreasing ids.
    """


# --------------------------------------------------------------------------
# Certification and release
# --------------------------------------------------------------------------


class CertificationError(NiverelMambaError):
    """A numerical comparison exceeded its sealed tolerance."""


class FixtureError(NiverelMambaError):
    """A golden fixture is missing, malformed, or fails its integrity check."""


class AssetVerificationError(NiverelMambaError):
    """A downloaded asset does not match the SHA-256 recorded in its manifest."""
