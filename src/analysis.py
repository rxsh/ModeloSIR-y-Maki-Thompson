# Analisis y comparaciones
import numpy as np
import time
from solvers import euler_explicit, euler_improved, rk4, solve_rk45

def compare_methods(rhs, t_span, y0, params, dt):
    t_eval = np.linspace(t_span[0], t_span[1], 1001)

    t0 = time.time()
    tb, yb = solve_rk45(rhs, t_span, y0, t_eval, *params)
    t_bench = time.time() - t0

    results = []

    t0 = time.time()
    ts_e, ys_e = euler_explicit(rhs, t_span, y0, dt, *params)
    t_e = time.time() - t0

    err_e = np.linalg.norm(np.interp(ts_e, tb, yb[:,1]) - ys_e[:,1])

    results.append(("Euler", t_e, err_e))

    t0 = time.time()
    ts_em, ys_em = euler_improved(rhs, t_span, y0, dt, *params)
    t_em = time.time() - t0

    err_em = np.linalg.norm(np.interp(ts_em, tb, yb[:,1]) - ys_em[:,1])

    results.append(("Euler mejorado", t_em, err_em))

    t0 = time.time()
    ts_rk, ys_rk = rk4(rhs, t_span, y0, dt, *params)
    t_rk = time.time() - t0

    err_rk = np.linalg.norm(np.interp(ts_rk, tb, yb[:,1]) - ys_rk[:,1])

    results.append(("RK4", t_rk, err_rk))

    return results, (tb, yb)
