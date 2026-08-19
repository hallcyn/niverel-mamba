"""Strict loading: every refusal mode of brief section 3.2."""

from __future__ import annotations

import numpy as np
import pytest

from niverel_mamba import Mamba2Config, default_contract, validate_state_dict
from niverel_mamba.errors import (
    ContractVersionError,
    DtypeMismatchError,
    MissingKeysError,
    ShapeMismatchError,
    UnexpectedKeysError,
)
from niverel_mamba.schema import WeightContract


@pytest.fixture
def config():
    return Mamba2Config(d_model=64, d_state=16, expand=2, headdim=16)


@pytest.fixture
def good_state(config):
    return {
        key: np.zeros(shape, dtype=np.float32)
        for key, shape in default_contract().expected(config).items()
    }


def test_accepts_an_exact_match(config, good_state):
    checked = validate_state_dict(good_state, config)
    assert list(checked) == list(default_contract().expected(config))


def test_missing_key_is_refused(config, good_state):
    del good_state["A_log"]
    with pytest.raises(MissingKeysError, match="A_log"):
        validate_state_dict(good_state, config)


def test_unexpected_key_is_refused(config, good_state):
    good_state["mystery.weight"] = np.zeros((3,), dtype=np.float32)
    with pytest.raises(UnexpectedKeysError, match=r"mystery\.weight"):
        validate_state_dict(good_state, config)


def test_wrong_shape_is_refused(config, good_state):
    good_state["D"] = np.zeros((999,), dtype=np.float32)
    with pytest.raises(ShapeMismatchError, match="D"):
        validate_state_dict(good_state, config)


def test_incompatible_configuration_is_refused(good_state):
    """A checkpoint trained with bias=True cannot load as bias=False.

    It shows up as an unexpected key rather than a silent drop, which is the
    entire point of refusing rather than filtering.
    """
    biased = Mamba2Config(d_model=64, d_state=16, expand=2, headdim=16, bias=True)
    state = {
        key: np.zeros(shape, dtype=np.float32)
        for key, shape in default_contract().expected(biased).items()
    }
    unbiased = Mamba2Config(d_model=64, d_state=16, expand=2, headdim=16, bias=False)
    with pytest.raises(UnexpectedKeysError, match=r"in_proj\.bias"):
        validate_state_dict(state, unbiased)


def test_undocumented_dtype_conversion_can_be_refused(config, good_state):
    good_state["D"] = good_state["D"].astype(np.float64)
    validate_state_dict(good_state, config)  # allowed by default, documented
    with pytest.raises(DtypeMismatchError):
        validate_state_dict(good_state, config, allow_dtype_cast=False)


def test_unknown_contract_version_is_refused():
    with pytest.raises(ContractVersionError, match="unsupported contract schema_version"):
        WeightContract.from_dict({"schema_version": "from-the-future-v9", "tensors": {"a": {}}})


def test_contract_has_no_init_states():
    """Mamba2 2.3.2.post1 has no learnable initial state.

    The brief listed 'init_states, if enabled' as indicative; the extraction
    proved it does not exist. Assert that so a hand-edit cannot reintroduce it.
    """
    assert "init_states" not in {t.name for t in default_contract().tensors}
