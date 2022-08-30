from typing import List

import matplotlib.pyplot as plt
import numpy as np
from bmd_common.utils import cprofile
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.linear_model import LinearRegression

lin = LinearRegression()


def _poly(arr: np.ndarray, n_deg: int):
    X = np.vstack([arr ** k for k in range(n_deg + 1)]).T
    return np.linalg.qr(X)[0][:, 1:]


def mod_poly(
    arr: np.ndarray, *, n_deg: int = 2, max_iter: int = 100, gradient: float = 0.001
):
    criterion = np.inf

    _old = arr
    _orig = arr
    correction = None

    poly = _poly(np.arange(1, len(_orig) + 1), n_deg)
    it = 0

    while (criterion >= gradient) and (it <= max_iter):
        correction = lin.fit(poly, _old).predict(poly)
        _work = np.minimum(_orig, correction)
        criterion = np.sum(np.abs((_work - _old) / _old))
        _old = _work
        it += 1

    corrected = _orig - correction
    return corrected, correction


def imod_poly(
    arr: np.ndarray, *, n_deg: int = 2, max_iter: int = 100, gradient: float = 0.001
):
    _old = arr
    _orig = arr

    poly = _poly(np.arange(1, len(_orig) + 1), n_deg)
    _pred = lin.fit(poly, _old).predict(poly)
    _old_std = np.std(_orig - _pred)

    ind = _orig <= (_pred + _old_std)

    _old = _old[ind]
    poly_updated = poly[ind]
    _pred = _pred[ind]

    for i in range(1, max_iter):
        if i >= 2:
            _old_std = _std
        _pred = lin.fit(poly_updated, _old).predict(poly_updated)
        _std = np.std(_old - _pred)

        if np.abs((_std - _old_std) / _std) < gradient:
            break
        else:
            for j in range(len(_old)):
                if _old[j] >= _pred[j] + _std:
                    _old[j] = _pred[j] + _std

    correction = lin.predict(poly)
    corrected = _orig - correction
    return corrected, correction


def _whittaker_smooth(x, w, lambda_):
    X = np.matrix(x)
    m = X.size
    E = sparse.eye(m, format="csc")
    D = E[1:] - E[:-1]
    W = sparse.diags(w, 0, shape=(m, m))
    A = sparse.csc_matrix(W + (lambda_ * D.T * D))
    B = sparse.csc_matrix(W * X.T)
    background = spsolve(A, B)
    return background


def zhang(arr: np.ndarray, *, lambda_: float = 100, max_iter: int = 15):
    _orig = arr
    weights = np.ones_like(_orig)
    for i in range(max_iter):
        correction = _whittaker_smooth(_orig, weights, lambda_)
        d = _orig - correction
        dssn = np.abs(d[d < 0].sum())
        if (dssn < 0.001 * np.abs(_orig).sum()) or (i == max_iter - 1):
            if i == max_iter - 1:
                print("Max iterations reached")
            break
        weights[d >= 0] = 0
        weights[d < 0] = np.exp((i + 1) * np.abs(d[d < 0]) / dssn)
        weights[0] = np.exp((i + 1) * (d[d < 0]).max() / dssn)
        weights[-1] = weights[0]
    corrected = _orig - correction
    return corrected, correction


def baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return y - z, z


@cprofile()
def run(plot: bool = True):
    arr = np.load("spectrum.npy")

    mod_poly_corrected, mod_poly_correction = mod_poly(arr, n_deg=5)
    imod_poly_corrected, imod_poly_correction = imod_poly(arr, n_deg=5)
    zhang_poly_corrected, zhang_poly_correction = zhang(arr, lambda_=200)

    if plot:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 6))
        fig: Figure
        axes: np.ndarray
        axes: List[Axes] = axes.tolist()

        axes[0].plot(arr, linewidth=0.8)
        axes[0].yaxis.set_label_text("Raw spectrum")

        axes[1].plot(mod_poly_corrected, linewidth=0.8)
        axes[1].plot(imod_poly_corrected, linewidth=0.8)
        axes[1].plot(zhang_poly_corrected, linewidth=0.8)
        axes[1].yaxis.set_label_text("Corrected")

        axes[2].plot(mod_poly_correction, linewidth=0.8)
        axes[2].plot(imod_poly_correction, linewidth=0.8)
        axes[2].plot(zhang_poly_correction, linewidth=0.8)
        axes[2].yaxis.set_label_text("Correction")

        fig.tight_layout()
        fig.show()


if __name__ == "__main__":
    run()
