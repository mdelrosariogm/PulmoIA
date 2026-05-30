"""Tests de la validación de datos ligera (Fase 3.2)."""

import numpy as np
import pandas as pd
import pytest

from pulmoia import config
from pulmoia.data.validation import (
    DataValidationError,
    assert_no_group_leakage,
    validate_features_frame,
)


def _frame(filenames, n_feats=5, seed=0):
    rng = np.random.RandomState(seed)
    data = {f"feat_{i}": rng.rand(len(filenames)) for i in range(n_feats)}
    data[config.GROUP_COL] = filenames
    for t in config.TARGETS:
        data[t] = rng.randint(0, 2, size=len(filenames))
    return pd.DataFrame(data)


def test_valida_frame_correcto():
    df = _frame([f"a{i}" for i in range(20)])
    rep = validate_features_frame(df, "ok")
    assert rep["n_rows"] == 20
    assert rep["n_features"] == 5


def test_falla_si_falta_grupo():
    df = _frame([f"a{i}" for i in range(5)]).drop(columns=[config.GROUP_COL])
    with pytest.raises(DataValidationError):
        validate_features_frame(df, "sin_grupo")


def test_falla_si_falta_target():
    df = _frame([f"a{i}" for i in range(5)]).drop(columns=[config.TARGETS[0]])
    with pytest.raises(DataValidationError):
        validate_features_frame(df, "sin_target")


def test_detecta_fuga_de_grupos():
    train = _frame(["a", "b", "c"])
    test = _frame(["c", "d"])  # 'c' está en ambos → fuga
    with pytest.raises(DataValidationError):
        assert_no_group_leakage(train, test)


def test_sin_fuga_ok():
    train = _frame(["a", "b", "c"])
    test = _frame(["d", "e"])
    assert assert_no_group_leakage(train, test) == 0
