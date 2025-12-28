import inspect


def main() -> None:
    import numpy as np
    import cupy as cp
    from py_boost import GradientBoosting

    print('GradientBoosting module:', GradientBoosting.__module__)
    print('predict signature:', inspect.signature(GradientBoosting.predict))

    # Tiny fit
    n = 256
    d = 64
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d), dtype=np.float32)
    y = (rng.random(n) > 0.9).astype(np.float32)
    X_tr, X_va = X[:200], X[200:]
    y_tr, y_va = y[:200], y[200:]

    model = GradientBoosting(loss='logloss', ntrees=20, es=5, seed=1)
    model.fit(X_tr, y_tr, eval_sets=[{'X': X_va, 'y': y_va}])

    # Predict on numpy
    p_np = model.predict(X_va)
    print('predict numpy type:', type(p_np), 'shape:', getattr(p_np, 'shape', None))

    # Predict on cupy
    X_va_gpu = cp.asarray(X_va)
    p_cp = model.predict(X_va_gpu)
    print('predict cupy-in type:', type(p_cp), 'shape:', getattr(p_cp, 'shape', None))

    # If output is cupy, convert to numpy for finiteness check
    if hasattr(p_cp, 'get'):
        p_cp_host = p_cp.get()
    else:
        p_cp_host = np.asarray(p_cp)

    print('finite numpy:', np.isfinite(np.asarray(p_np)).all())
    print('finite cupy-in:', np.isfinite(np.asarray(p_cp_host)).all())


if __name__ == '__main__':
    main()
