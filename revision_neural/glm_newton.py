"""Newton (IRLS) fit of the ridge-penalised Poisson GLM used by gated_glm2/3 (same objective as
gated_glm2.fit, about 20x faster). Returns (theta, nll) with theta = [coef..., intercept]."""
import numpy as np


def fit_newton(X, y, theta0=None, ridge=1e-3, tol=1e-12, maxit=100):
    n, p = X.shape
    Xa = np.concatenate([X, np.ones((n, 1), np.float32)], 1)
    y = y.astype(np.float64)
    if theta0 is None:
        theta0 = np.zeros(p + 1); theta0[p] = np.log(max(y.sum(), 1) / n)
    th = np.asarray(theta0, np.float64).copy()
    R = np.full(p + 1, ridge); R[p] = 0.0

    def obj(t):
        eta = (Xa @ t.astype(np.float32)).astype(np.float64)
        mu = np.exp(np.minimum(eta, 10.0))
        return mu.sum() - (y * eta).sum() + 0.5 * (R * t ** 2).sum(), eta, mu

    f, eta, mu = obj(th)
    for _ in range(maxit):
        g = (Xa.T @ (mu - y).astype(np.float32)).astype(np.float64) + R * th
        H = (Xa.T @ (Xa * mu.astype(np.float32)[:, None])).astype(np.float64) + np.diag(R + 1e-9)
        step = np.linalg.lstsq(H, g, rcond=None)[0]
        s = 1.0
        while True:
            fn, etan, mun = obj(th - s * step)
            if fn <= f or s < 1e-8:
                break
            s *= 0.5
        if fn > f:
            break
        th = th - s * step; dec = f - fn; f, eta, mu = fn, etan, mun
        if dec < tol * max(1.0, abs(f)):
            break
    return th, float(mu.sum() - (y * eta).sum())
