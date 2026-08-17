"""Canonical, backend-independent Mamba2 configuration.

The field set mirrors the constructor of ``mamba_ssm.modules.mamba2.Mamba2``
at version 2.3.2.post1 exactly -- including the options Niverel does not use
(``d_ssm``, ``D_has_hdim``, ``norm_before_gate``, ``rmsnorm=False``) -- so that
the weight contract can describe any upstream checkpoint, not just ours.

Fields that only affect *initialisation* (``conv_init``, ``A_init_range``,
``dt_min``/``dt_max``/``dt_init_floor``) are kept because they belong to the
provenance of a checkpoint, but they never influence a forward pass.

Fields that only affect upstream *plumbing* (``process_group``,
``sequence_parallel``, ``use_mem_eff_path``, ``device``, ``dtype``) are
deliberately absent: they are runtime choices, not part of the weight
contract. ``layer_idx`` is likewise not a weight-contract property.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from .errors import UnsupportedConfigError

__all__ = ["Mamba2Config"]

# Upstream hard-codes this in ``Mamba2.__init__`` (it is *not* the 1e-6 default
# of ``rms_norm_ref``). Getting it wrong is a silent ~1e-3 error.
_UPSTREAM_NORM_EPS = 1e-5

_INFINITY_TOKENS = frozenset({"inf", "+inf", "infinity", "+infinity"})


def _parse_dt_limit(value: Any) -> tuple[float, float]:
    """Accept ``(lo, hi)`` in tuple, list or JSON-string form.

    ``float("inf")`` is not representable in JSON, so a serialised config may
    carry the string ``"inf"``. Round-tripping must be lossless.
    """
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise UnsupportedConfigError(
            f"dt_limit must be a pair (lo, hi), got {value!r}"
        )
    out = []
    for item in value:
        if isinstance(item, str):
            token = item.strip().lower()
            if token in _INFINITY_TOKENS:
                out.append(math.inf)
                continue
            if token in {"-inf", "-infinity"}:
                out.append(-math.inf)
                continue
            try:
                out.append(float(token))
            except ValueError as exc:  # pragma: no cover - defensive
                raise UnsupportedConfigError(f"dt_limit entry {item!r} is not a number") from exc
        else:
            out.append(float(item))
    lo, hi = out
    if not (lo < hi):
        raise UnsupportedConfigError(f"dt_limit must satisfy lo < hi, got ({lo}, {hi})")
    if lo < 0.0:
        raise UnsupportedConfigError(f"dt_limit lower bound must be >= 0, got {lo}")
    return (lo, hi)


def _encode_float(value: float) -> float | str:
    """JSON-safe encoding of a float that may be infinite."""
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


@dataclass(frozen=True)
class Mamba2Config:
    """A fully serialisable Mamba2 configuration.

    Example
    -------
    >>> config = Mamba2Config(d_model=768, d_state=128, d_conv=4, expand=2)
    >>> config.d_inner, config.nheads, config.conv_dim, config.d_in_proj
    (1536, 24, 1792, 3352)
    """

    d_model: int
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    chunk_size: int = 256
    bias: bool = False
    conv_bias: bool = True

    # Optional narrowing of the SSM path. ``None`` means "the whole d_inner".
    # When ``d_ssm < d_inner`` the remainder becomes a gated MLP branch.
    d_ssm: int | None = None

    D_has_hdim: bool = False
    rmsnorm: bool = True
    norm_before_gate: bool = False
    norm_epsilon: float = _UPSTREAM_NORM_EPS
    dt_limit: tuple[float, float] = (0.0, math.inf)

    # Initialisation-only provenance. Never used by a forward pass.
    conv_init: float | None = None
    A_init_range: tuple[float, float] = (1.0, 16.0)
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4

    # Free-form provenance carried alongside the config (upstream version the
    # checkpoint came from, originating run id, ...). Never affects numerics.
    metadata: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        object.__setattr__(self, "dt_limit", _parse_dt_limit(self.dt_limit))
        object.__setattr__(self, "A_init_range", tuple(float(v) for v in self.A_init_range))

        for name in ("d_model", "d_state", "d_conv", "expand", "headdim", "ngroups", "chunk_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise UnsupportedConfigError(f"{name} must be an int, got {value!r}")
            if value <= 0:
                raise UnsupportedConfigError(f"{name} must be positive, got {value}")

        d_inner = self.expand * self.d_model
        d_ssm = d_inner if self.d_ssm is None else self.d_ssm

        if d_ssm <= 0 or d_ssm > d_inner:
            raise UnsupportedConfigError(
                f"d_ssm must lie in (0, d_inner={d_inner}], got {d_ssm}"
            )
        if d_ssm % self.headdim != 0:
            raise UnsupportedConfigError(
                f"d_ssm ({d_ssm}) must be divisible by headdim ({self.headdim})"
            )
        nheads = d_ssm // self.headdim
        if self.ngroups > nheads:
            raise UnsupportedConfigError(
                f"ngroups ({self.ngroups}) cannot exceed nheads ({nheads})"
            )
        if nheads % self.ngroups != 0:
            raise UnsupportedConfigError(
                f"nheads ({nheads}) must be divisible by ngroups ({self.ngroups})"
            )
        if self.rmsnorm and d_ssm % self.ngroups != 0:
            raise UnsupportedConfigError(
                f"d_ssm ({d_ssm}) must be divisible by ngroups ({self.ngroups}) for the gated norm"
            )
        if self.norm_before_gate and not self.rmsnorm:
            raise UnsupportedConfigError("norm_before_gate=True requires rmsnorm=True")
        if self.norm_epsilon <= 0:
            raise UnsupportedConfigError(f"norm_epsilon must be positive, got {self.norm_epsilon}")

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def d_inner(self) -> int:
        """``expand * d_model`` -- the width of the gated inner path."""
        return self.expand * self.d_model

    @property
    def effective_d_ssm(self) -> int:
        """``d_ssm`` with the ``None`` default resolved to ``d_inner``."""
        return self.d_inner if self.d_ssm is None else self.d_ssm

    @property
    def nheads(self) -> int:
        """``d_ssm // headdim``."""
        return self.effective_d_ssm // self.headdim

    @property
    def d_mlp(self) -> int:
        """Width of each of the two gated-MLP splits (``z0`` and ``x0``).

        Zero whenever ``d_ssm == d_inner``, which is the Niverel case.

        Note this is ``d_inner - d_ssm``, *not* half of it. Upstream derives
        it as ``(d_in_proj - 2*d_ssm - 2*ngroups*d_state - nheads) // 2``, and
        since ``d_in_proj`` is built from ``d_inner`` (never ``d_ssm``) that
        expression simplifies to ``d_inner - d_ssm``. Halving it would make
        the five ``in_proj`` splits fail to sum to ``d_in_proj``.
        """
        return self.d_inner - self.effective_d_ssm

    @property
    def conv_dim(self) -> int:
        """Channel count of the depthwise causal conv: ``d_ssm + 2 * ngroups * d_state``."""
        return self.effective_d_ssm + 2 * self.ngroups * self.d_state

    @property
    def d_in_proj(self) -> int:
        """Output width of ``in_proj``: ``2 * d_inner + 2 * ngroups * d_state + nheads``."""
        return 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads

    @property
    def d_D(self) -> int:
        """Length of the ``D`` skip parameter: ``d_ssm`` if ``D_has_hdim`` else ``nheads``."""
        return self.effective_d_ssm if self.D_has_hdim else self.nheads

    @property
    def norm_group_size(self) -> int:
        """Group size of the gated RMSNorm: ``d_ssm // ngroups``."""
        return self.effective_d_ssm // self.ngroups

    @property
    def in_proj_split(self) -> tuple[int, int, int, int, int]:
        """The five ``in_proj`` output splits, in upstream order.

        ``[z0, x0, z, xBC, dt]`` -- exactly the ``torch.split`` sizes used by
        ``Mamba2.forward`` and ``Mamba2.step``.
        """
        return (
            self.d_mlp,
            self.d_mlp,
            self.effective_d_ssm,
            self.conv_dim,
            self.nheads,
        )

    @property
    def has_dt_limit(self) -> bool:
        """Whether ``dt_limit`` actually constrains anything."""
        return self.dt_limit != (0.0, math.inf)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable representation. Round-trips through :meth:`from_dict`."""
        payload = asdict(self)
        payload["dt_limit"] = [_encode_float(v) for v in self.dt_limit]
        payload["A_init_range"] = [_encode_float(v) for v in self.A_init_range]
        for key in ("norm_epsilon", "dt_min", "dt_max", "dt_init_floor"):
            payload[key] = _encode_float(payload[key])
        if not payload["metadata"]:
            payload.pop("metadata")
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Mamba2Config:
        """Rebuild a config, rejecting unknown keys rather than ignoring them."""
        known = {f.name for f in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise UnsupportedConfigError(
                f"unknown configuration keys: {sorted(unknown)}; known keys are {sorted(known)}"
            )
        return cls(**payload)

    def upstream_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for constructing the real upstream ``Mamba2``.

        Used by ``cuda-reference`` and by the contract-extraction script, so
        that a single config object drives both our module and upstream's.
        """
        return {
            "d_model": self.d_model,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "conv_init": self.conv_init,
            "expand": self.expand,
            "headdim": self.headdim,
            "d_ssm": self.d_ssm,
            "ngroups": self.ngroups,
            "A_init_range": tuple(self.A_init_range),
            "D_has_hdim": self.D_has_hdim,
            "rmsnorm": self.rmsnorm,
            "norm_before_gate": self.norm_before_gate,
            "dt_min": self.dt_min,
            "dt_max": self.dt_max,
            "dt_init_floor": self.dt_init_floor,
            "dt_limit": tuple(self.dt_limit),
            "bias": self.bias,
            "conv_bias": self.conv_bias,
            "chunk_size": self.chunk_size,
        }
