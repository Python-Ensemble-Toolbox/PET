'''
Tests for the analysis tools module.
'''
import pytest
import numpy as np
import pipt.misc_tools.analysis_tools as atools

def test_truncSVD_big_matrix():
    np.random.seed(10_08_1997)

    A = np.random.rand(1000, 1000)
    U, S, Vt = atools.truncSVD(A, energy=0.999)
    A_approx = U @ np.diag(S) @ Vt
    A_inv_approx = Vt.T @ np.diag(1/S) @ U.T

    np.testing.assert_allclose(A, A_approx, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(A @ A_inv_approx, np.eye(1000), rtol=1e-2, atol=1e-1)


def _reconstruct(U, S, Vt):
    return U @ np.diag(S) @ Vt

def _sorted_singular_values_desc(s):
    return np.sort(np.asarray(s))[::-1]

@pytest.mark.parametrize("shape,r", [((20, 15), 5), ((15, 20), 6)])
def test_truncSVD_matches_rank_r(shape, r):
    rng = np.random.default_rng(10081997)
    A = rng.standard_normal(shape)

    U, S, Vt = atools.truncSVD(A, r=r)
    U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)

    A_approx = _reconstruct(U, S, Vt)
    A_expected = _reconstruct(U_np[:, :r], S_np[:r], Vt_np[:r, :])

    np.testing.assert_allclose(A_approx, A_expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(S, S_np[:r], rtol=1e-12, atol=1e-12)


def test_truncSVD_matches_scipy_svds_rank_r():
    sp_linalg = pytest.importorskip("scipy.sparse.linalg")

    rng = np.random.default_rng(10081997)
    A = rng.standard_normal((30, 20))
    r = 7

    U_pet, S_pet, Vt_pet = atools.truncSVD(A, r=r)
    U_sp, S_sp, Vt_sp = sp_linalg.svds(A, k=r, which='LM')

    S_sp = _sorted_singular_values_desc(S_sp)
    S_pet_sorted = _sorted_singular_values_desc(S_pet)

    np.testing.assert_allclose(S_pet_sorted, S_sp, rtol=1e-6, atol=1e-6)

    A_pet = _reconstruct(U_pet, S_pet, Vt_pet)
    rel_err_pet = np.linalg.norm(A - A_pet, ord='fro') / np.linalg.norm(A, ord='fro')

    U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)
    A_best_rank_r = _reconstruct(U_np[:, :r], S_np[:r], Vt_np[:r, :])
    rel_err_best = np.linalg.norm(A - A_best_rank_r, ord='fro') / np.linalg.norm(A, ord='fro')

    np.testing.assert_allclose(rel_err_pet, rel_err_best, rtol=1e-8, atol=1e-10)


def test_truncSVD_matches_sklearn_truncatedsvd_rank_r():
    sklearn_decomp = pytest.importorskip("sklearn.decomposition")

    rng = np.random.default_rng(10081997)
    A = rng.standard_normal((25, 18))
    r = 6

    U_pet, S_pet, Vt_pet = atools.truncSVD(A, r=r)
    model = sklearn_decomp.TruncatedSVD(n_components=r, algorithm='randomized', random_state=0)
    A_proj = model.fit_transform(A)
    Vt_sk = model.components_
    S_sk = model.singular_values_

    S_pet_sorted = _sorted_singular_values_desc(S_pet)
    S_sk_sorted = _sorted_singular_values_desc(S_sk)
    np.testing.assert_allclose(S_pet_sorted, S_sk_sorted, rtol=1e-5, atol=1e-7)

    A_pet = _reconstruct(U_pet, S_pet, Vt_pet)
    A_sk = A_proj @ Vt_sk

    rel_err_pet = np.linalg.norm(A - A_pet, ord='fro') / np.linalg.norm(A, ord='fro')
    rel_err_sk = np.linalg.norm(A - A_sk, ord='fro') / np.linalg.norm(A, ord='fro')

    np.testing.assert_allclose(rel_err_pet, rel_err_sk, rtol=1e-4, atol=1e-6)
