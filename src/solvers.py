# Metodos numericos (Euler, Euler Mejorado, Runge-Kutta)

import numpy as np
from scipy.integrate import solve_ivp

def euler_explicit(rhs, t_span, y0, dt, *args):
    t0, tf = t_span
    N = int(np.ceil((tf - t0) / dt)) + 1
    ts = np.linspace(t0, t0 + (N - 1) * dt, N)

    ys = np.zeros((N, len(y0)))
    ys[0] = y0

    for k in range(N - 1):
        ys[k+1] = ys[k] + dt * rhs(ts[k], ys[k], *args)

    return ts, ys


def euler_improved(rhs, t_span, y0, dt, *args):
    t0, tf = t_span
    N = int(np.ceil((tf - t0) / dt)) + 1
    ts = np.linspace(t0, t0 + (N - 1) * dt, N)

    ys = np.zeros((N, len(y0)))
    ys[0] = y0

    for k in range(N - 1):
        k1 = rhs(ts[k], ys[k], *args)
        k2 = rhs(ts[k] + dt/2, ys[k] + dt*k1/2, *args)
        ys[k+1] = ys[k] + dt * k2

    return ts, ys


def rk4(rhs, t_span, y0, dt, *args):
    t0, tf = t_span
    N = int(np.ceil((tf - t0)/dt)) + 1
    ts = np.linspace(t0, t0 + (N-1)*dt, N)

    ys = np.zeros((N, len(y0)))
    ys[0] = y0

    for k in range(N-1):
        t = ts[k]
        y = ys[k]

        k1 = rhs(t, y, *args)
        k2 = rhs(t + dt/2, y + dt*k1/2, *args)
        k3 = rhs(t + dt/2, y + dt*k2/2, *args)
        k4 = rhs(t + dt, y + dt*k3, *args)

        ys[k+1] = y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

    return ts, ys


def solve_rk45(rhs, t_span, y0, t_eval, *args):
    sol = solve_ivp(lambda t, y: rhs(t, y, *args), t_span, y0, t_eval=t_eval,
                    rtol=1e-8, atol=1e-10)
    return sol.t, sol.y.T
