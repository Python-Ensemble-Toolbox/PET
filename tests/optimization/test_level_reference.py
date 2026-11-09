"""Coarse gradient, fine objective: each level is compared with its own value at the mean state.

Froy's "mixed" EnOpt runs put every perturbation on a coarse model (``en_size = [ne, 0]``)
and evaluate the trial step on the fine model. The gradient subtracted the fine value at
the mean from the coarse members, so the coarse-fine offset entered every dF. The
perturbations are recentred and then clipped, so for a control at a bound the clipped
perturbations have a non-zero mean and the offset appeared as a gradient component on
every bound control. Normalised by its largest element, the gradient then moved the
bound controls into the bound and the free controls by a fraction of alpha.

Now ``function`` takes a ``level``; a single state goes to the finest level by default,
and the gradient evaluates the mean state on each coarser level with members and uses that
as the reference. Member counts per level come from the state, not from ``en_size``.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from ensemble.ensemble import BaseEnsemble
from popt.ensembles.ensemble_base import EnsembleOptimizationBase
from popt.ensembles.ensemble_gaussian import GaussianEnsemble

NX = 4
ML_NE = [6, 0]


def _multilevel(ens, ml_ne=ML_NE, weights=(1.0,)):
    ens.keys_en = {'multilevel': [['levels', len(ml_ne)], ['en_size', list(ml_ne)], ['cov_wgt', list(weights)]]}
    ens.multilevel = {'levels': list(range(len(ml_ne))), 'ml_ne': list(ml_ne),
                      'ml_weights': np.asarray(weights, dtype=float) / np.sum(weights)}
    ens.tot_level = len(ml_ne)
    return ens


# --------------------------------------------------------------------------
# Placing a state on a level
# --------------------------------------------------------------------------
def _base():
    return _multilevel(EnsembleOptimizationBase.__new__(EnsembleOptimizationBase))


def test_a_single_state_goes_to_the_finest_level_by_default():
    ens = _base()
    x = np.arange(NX, dtype=float)

    blocks = ens._reorganize_multilevel_ensemble(x)

    assert isinstance(blocks, list) and len(blocks) == 2
    assert blocks[0].shape == (NX, 0)
    np.testing.assert_array_equal(blocks[1], x[:, None])
    assert ens.levels_with_members(x) == [1]


@pytest.mark.parametrize('level, target', [(0, 0), (1, 1), (-1, 1), (-2, 0)])
def test_level_places_a_single_state_on_that_level(level, target):
    ens = _base()
    x = np.arange(NX, dtype=float)

    blocks = ens._reorganize_multilevel_ensemble(x, level=ens._resolve_level(level))

    for l, block in enumerate(blocks):
        assert block.shape == ((NX, 1) if l == target else (NX, 0))


def test_an_ensemble_is_split_by_en_size_unless_a_level_is_given():
    ens = _base()
    enX = np.random.default_rng(0).normal(size=(NX, ML_NE[0]))

    by_size = ens._reorganize_multilevel_ensemble(enX)
    np.testing.assert_array_equal(by_size[0], enX)
    assert by_size[1].shape == (NX, 0)

    on_fine = ens._reorganize_multilevel_ensemble(enX, level=1)
    assert on_fine[0].shape == (NX, 0)
    np.testing.assert_array_equal(on_fine[1], enX)

    np.testing.assert_array_equal(ens._reorganize_multilevel_ensemble(by_size), enX)   # and back


def test_level_is_rejected_without_multilevel_or_out_of_range():
    ens = _base()
    with pytest.raises(ValueError):
        ens._resolve_level(2)
    ens.keys_en = {}
    with pytest.raises(ValueError):
        ens._resolve_level(0)
    x = np.arange(NX, dtype=float)
    assert ens._reorganize_multilevel_ensemble(x) is x       # single level: untouched


# --------------------------------------------------------------------------
# `function` with a level
# --------------------------------------------------------------------------
def _host(value=3.0):
    """An EnsembleOptimizationBase with only what `function` reads; the forecast is recorded."""
    ens = _base()
    seen = []

    def calc_prediction(x, save_prediction=None):
        seen.append([np.array(b) for b in x])
        return True

    ens.ne = 1
    ens.num_models = 1
    ens.num_samples = ML_NE[0]
    ens.aux_input = None
    ens.idX = {'x': (0, NX)}
    ens.save_prediction = None
    ens.sim = SimpleNamespace(input_dict={}, true_order=None)
    ens._sim_data = None            # the `sim_data` property reads these
    ens.member_outputs = []
    ens.logger = SimpleNamespace(error=lambda m: None, info=lambda m: None)
    ens.calc_prediction = calc_prediction
    ens.obj_func = lambda *a, **k: np.array([value])
    ens.stateF = np.array([7.0])
    ens.enF = None
    return ens, seen


def test_the_mean_on_a_coarse_level_is_returned_but_does_not_become_stateF():
    ens, seen = _host(value=3.0)
    x = np.arange(NX, dtype=float)

    value = ens.function(x, level=0)

    np.testing.assert_array_equal(value, [3.0])
    np.testing.assert_array_equal(ens.stateF, [7.0])
    np.testing.assert_array_equal(seen[0][0], x[:, None])
    assert seen[0][1].shape == (NX, 0)


def test_a_single_state_without_level_runs_on_the_finest_level_and_sets_stateF():
    ens, seen = _host(value=3.0)
    x = np.arange(NX, dtype=float)

    ens.function(x)

    np.testing.assert_array_equal(ens.stateF, [3.0])
    assert seen[0][0].shape == (NX, 0)
    np.testing.assert_array_equal(seen[0][1], x[:, None])


# --------------------------------------------------------------------------
# The gradient compares each level with its own value at the mean
# --------------------------------------------------------------------------
A = np.array([0.5, -0.3, 0.8, 0.2])      # linear fine objective  F(x) = A @ x
SIGMA = 0.025
NE = 4000


def _gaussian(offset, seed=0, fail_reference=False):
    """Coarse level 0 is the fine objective plus `offset`; two controls sit at the lower bound."""
    ens = _multilevel(GaussianEnsemble.__new__(GaussianEnsemble), ml_ne=[NE, 0])
    ens.stateX = np.array([0.0, 0.4, 0.0, 0.7])
    ens.dimX = NX
    ens.covX = SIGMA ** 2 * np.eye(NX)
    ens.num_samples = NE
    ens.num_models = 1
    ens.lb = np.zeros(NX)
    ens.ub = np.ones(NX)
    ens.rng = np.random.default_rng(seed)
    ens.stateF = np.atleast_1d(A @ ens.stateX)        # the optimizer's fine value at the mean
    warnings = []
    ens.logger = SimpleNamespace(warning=warnings.append, info=lambda m: None, error=lambda m: None)
    calls = []

    def function(X, level=None, **kwargs):
        X = np.asarray(X)
        calls.append((X.shape, level))
        X2 = X if X.ndim == 2 else X[:, None]
        coarse = (X2.shape[1] > 1) or (level == 0)   # the ensemble sits on level 0 by en_size
        if coarse and X2.shape[1] == 1 and fail_reference:
            return np.array([np.inf])
        return A @ X2 + (offset if coarse else 0.0)

    ens.function = function
    return ens, calls, warnings


def test_the_coarse_offset_no_longer_enters_the_gradient_or_hessian():
    with_offset, calls, _ = _gaussian(offset=5.0)
    without, _, _ = _gaussian(offset=0.0)

    g_with = with_offset.gradient(with_offset.stateX, with_offset.covX)
    g_without = without.gradient(without.stateX, without.covX)
    np.testing.assert_allclose(g_with, g_without, rtol=0, atol=1e-12)
    np.testing.assert_allclose(with_offset.hessian(), without.hessian(), rtol=0, atol=1e-12)

    # the mean state was evaluated once on the coarse level, before the ensemble
    assert calls[0] == ((NX,), 0)
    assert calls[1] == ((NX, NE), None)
    np.testing.assert_array_equal(with_offset.stateF, np.atleast_1d(A @ with_offset.stateX))  # untouched


def test_the_old_reference_put_the_offset_on_every_bound_control():
    """What the fix removes: with the fine value as reference the bound controls get
    offset * mean(clipped perturbation) = offset * sigma / sqrt(2 pi) each."""
    ens, _, _ = _gaussian(offset=5.0)
    g_new = ens.gradient(ens.stateX, ens.covX)

    dF_old = ens.enF[0] - ens.stateF
    g_old = (ens.enX - ens.stateX[:, None]) @ dF_old / NE

    at_bound = ens.stateX <= 0.0
    leak = 5.0 * SIGMA / np.sqrt(2 * np.pi)
    np.testing.assert_allclose(g_old[at_bound] - g_new[at_bound], leak, rtol=0.1)
    np.testing.assert_allclose(g_old[~at_bound], g_new[~at_bound], atol=1e-12)
    # the leaked component dwarfs the real ones, which is why the normalised step collapsed
    assert np.abs(g_old[at_bound]).min() > 5 * np.abs(g_new).max()


def test_a_failed_reference_falls_back_to_the_level_mean_with_a_warning():
    ens, _, warnings = _gaussian(offset=5.0, fail_reference=True)

    g = ens.gradient(ens.stateX, ens.covX)

    assert np.all(np.isfinite(g))
    assert any('level 0' in w for w in warnings)
    assert ens.levelF[0] is None
    np.testing.assert_allclose(ens._reference(0), np.mean(ens.enF[0]))


# --------------------------------------------------------------------------
# The forecast counts members per level from the state
# --------------------------------------------------------------------------
def _forecaster():
    ens = _multilevel(BaseEnsemble.__new__(BaseEnsemble), ml_ne=[5, 0])
    levels_setup, runs = [], []
    ens.sim = SimpleNamespace(
        input_dict={'parallel': 1},
        redund_sim=None,
        setup_fwd_run=lambda **kw: levels_setup.append(kw['level']),
        run_fwd_sim=lambda state, member: runs.append(member) or [{'d': np.array([state['x'].sum()])}],
    )
    ens.idX = {'x': (0, NX)}
    ens.aux_input = None
    ens.logger = SimpleNamespace(info=lambda m: None)
    ens.rng = np.random.default_rng(0)
    ens.save = lambda: None
    ens.disable_tqdm = True
    return ens, levels_setup, runs


@pytest.mark.parametrize('level', [0, 1])
def test_a_single_state_runs_once_on_its_level(level):
    ens, levels_setup, runs = _forecaster()
    x = np.arange(NX, dtype=float)[:, None]
    enX = [x if l == level else np.empty((NX, 0)) for l in range(2)]

    assert ens.calc_prediction(enX)

    assert runs == [0]
    assert ens.run_levels == [level]
    assert len(ens.member_outputs) == 1
    assert levels_setup[-1] == 1                      # the simulator is set up per level


def test_members_are_counted_from_the_state_not_from_en_size():
    ens, _, runs = _forecaster()                      # en_size says 5 on level 0
    enX = [np.random.default_rng(0).normal(size=(NX, 3)), np.empty((NX, 0))]

    assert ens.calc_prediction(enX)

    assert runs == [0, 1, 2]
    assert ens.run_levels == [0]
