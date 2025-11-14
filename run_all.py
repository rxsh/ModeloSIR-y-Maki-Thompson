import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import csv

sys.path.append(os.path.abspath("src"))
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

from models import (
    sir_rhs,
    sir_rhs_full,
    rumor_general_rhs,
    maki_thompson_rhs
)

from solvers import (
    solve_rk45,
    euler_explicit,
    euler_improved,
    rk4
)

from analysis import compare_methods


def run_sir():
    print("\n MODELO SIR ")

    beta = 0.5
    gamma = 0.1
    S0, I0 = 0.99, 0.01

    t_eval = np.linspace(0, 60, 800)

    t, y = solve_rk45(
        sir_rhs,
        (0, 60),
        [S0, I0],
        t_eval,
        beta,
        gamma
    )

    S, I = y[:, 0], y[:, 1]
    R = 1 - S - I

    plt.figure()
    plt.plot(t, S, label="S(t)")
    plt.plot(t, I, label="I(t)")
    plt.plot(t, R, label="R(t)")
    plt.title("Modelo SIR (β=0.5, γ=0.1)")
    plt.xlabel("t")
    plt.ylabel("Proporción")
    plt.grid(True)
    plt.legend()
    fname = os.path.join(OUTPUT_DIR, "sir_timeseries.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()


def run_retrato_fase():
    print("\n RETRATO DE FASE S-I ")

    beta = 0.5
    gamma = 0.1
    t_eval = np.linspace(0, 60, 800)

    condiciones = [
        [0.99, 0.01],
        [0.95, 0.05],
        [0.90, 0.10]
    ]

    plt.figure(figsize=(7,6))

    for ic in condiciones:
        t, y = solve_rk45(sir_rhs, (0,60), ic, t_eval, beta, gamma)
        S, I = y[:,0], y[:,1]
        plt.plot(S, I, label=f"S0={ic[0]}, I0={ic[1]}")

        idx = np.argmax(I)
        plt.scatter(S[idx], I[idx], color="red")

    Sg = np.linspace(0.01, 1.0, 20)
    Ig = np.linspace(0.0, 0.4, 20)
    Smesh, Imesh = np.meshgrid(Sg, Ig)
    dS = -beta * Smesh * Imesh
    dI = beta * Smesh * Imesh - gamma * Imesh
    plt.quiver(Smesh, Imesh, dS, dI)

    plt.xlabel("S")
    plt.ylabel("I")
    plt.title("Retrato de Fase S-I")
    plt.legend()
    plt.grid(True)
    fname = os.path.join(OUTPUT_DIR, "retrato_fase_SI.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()


def run_rumor_general():
    print("\n MODELO GENERAL DEL RUMOR ")

    lam = 0.6
    delta = 0.1
    alpha = 0.2

    X0, Y0 = 0.99, 0.01
    t_eval = np.linspace(0, 60, 800)

    t, y = solve_rk45(
        rumor_general_rhs,
        (0, 60),
        [X0, Y0],
        t_eval,
        lam, delta, alpha
    )

    X, Y = y[:, 0], y[:, 1]
    Z = 1 - X - Y

    plt.figure()
    plt.plot(t, X, label="Ignorantes X(t)")
    plt.plot(t, Y, label="Informantes Y(t)")
    plt.plot(t, Z, label="Neutros Z(t)")
    plt.title("Modelo General de Rumor")
    plt.xlabel("t")
    plt.ylabel("Proporción")
    plt.legend()
    plt.grid()
    fname = os.path.join(OUTPUT_DIR, "rumor_general_timeseries.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()


def run_maki():
    print("\n MODELO MAKI-THOMPSON ")

    lam = 0.6
    alpha = 0.2
    t_eval = np.linspace(0, 60, 800)

    condiciones = [
        [0.99, 0.01],
        [0.95, 0.05],
        [0.90, 0.10],
    ]

    for ic in condiciones:
        t, y = solve_rk45(
            maki_thompson_rhs, (0,60), ic, t_eval, lam, alpha
        )
        X, Y = y[:,0], y[:,1]

        plt.figure()
        plt.plot(t, X, label="Ignorantes X(t)")
        plt.plot(t, Y, label="Informantes Y(t)")
        plt.title(f"Maki-Thompson (IC={ic})")
        plt.xlabel("t")
        plt.grid(True)
        plt.legend()
        fname = os.path.join(OUTPUT_DIR, f"maki_IC_{int(ic[0]*100)}_{int(ic[1]*100)}.png")
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()

        # registrar proporcion final de informantes
        y_final = float(Y[-1])
        print(f"IC={ic} -> Y_final={y_final:.6f}")
        with open(os.path.join(OUTPUT_DIR, "maki_final_proportions.txt"), "a", encoding="utf-8") as f:
            f.write(f"IC={ic}, Y_final={y_final:.6f}\n")


def run_sir_sweep():
    """Barrido representativo de (beta,gamma) y guardado de sir_summary.csv"""
    print("\n BARRIDO SIR (guardando outputs/sir_summary.csv) ")
    pairs = [
        (0.5, 0.1),
        (0.3, 0.1),
        (0.2, 0.5),
        (0.15, 0.2)
    ]
    t_eval = np.linspace(0, 80, 1600)
    ic = [0.99, 0.01]
    rows = []
    for beta, gamma in pairs:
        t, y = solve_rk45(sir_rhs, (0, 80), ic, t_eval, beta, gamma)
        S = y[:,0]; I = y[:,1]
        imax = I.max()
        t_peak = t[I.argmax()]
        R0 = beta / gamma
        rows.append((beta, gamma, R0, float(imax), float(t_peak)))
    csv_path = os.path.join(OUTPUT_DIR, "sir_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["beta","gamma","R0","I_max","t_peak"])
        writer.writerows(rows)
    print("Sir summary saved to", csv_path)



def run_compare_sir_rumor():
    """Compara I(t) del SIR con Y(t) del modelo general de rumor en el mismo grafico.
    Guarda outputs/compare_sir_rumor.png
    """
    print("\n COMPARACION SIR vs RUMOR (I(t) vs Y(t))")

    # Parametros SIR
    beta = 0.5
    gamma = 0.1

    # Parametros modelo general de rumor
    lam = 0.6
    delta = 0.1
    alpha = 0.2

    S0, I0 = 0.99, 0.01
    X0, Y0 = 0.99, 0.01
    t_eval = np.linspace(0, 60, 800)

    t_s, y_s = solve_rk45(sir_rhs, (0, 60), [S0, I0], t_eval, beta, gamma)
    t_r, y_r = solve_rk45(rumor_general_rhs, (0, 60), [X0, Y0], t_eval, lam, delta, alpha)

    I = y_s[:, 1]
    Y = y_r[:, 1]

    plt.figure()
    plt.plot(t_s, I, label="I(t) - SIR")
    plt.plot(t_r, Y, label="Y(t) - Rumor general")
    plt.title(f"Comparacion SIR (\u03b2={beta},\u03b3={gamma}) vs Rumor (\u03bb={lam},\u03b4={delta},\u03b1={alpha})")
    plt.xlabel("t")
    plt.ylabel("Proporcion")
    plt.grid(True)
    plt.legend()
    fname = os.path.join(OUTPUT_DIR, "compare_sir_rumor.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {fname}")


def run_maki_phase_plot():
    """Genera retrato fase X vs Y para Maki-Thompson y lo guarda"""
    print("\n GENERANDO RETRATO FASE MAKI (X vs Y) ")
    lam = 0.6
    alpha = 0.2
    condiciones = [
        [0.99, 0.01],
        [0.95, 0.05],
        [0.90, 0.10],
    ]
    t_eval = np.linspace(0, 60, 800)
    plt.figure(figsize=(7,5))
    for ic in condiciones:
        t, y = solve_rk45(maki_thompson_rhs, (0,60), ic, t_eval, lam, alpha)
        X = y[:,0]; Y = y[:,1]
        plt.plot(X, Y, label=f"IC={ic}")
        plt.scatter(X[-1], Y[-1], s=20)  # punto final
    plt.xlabel("Ignorantes X")
    plt.ylabel("Informantes Y")
    plt.title("Retrato fase: Maki‑Thompson (X vs Y)")
    plt.legend()
    plt.grid(True)
    fname = os.path.join(OUTPUT_DIR, "maki_phase_XY.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved:", fname)

def run_comparacion_metodos():
    print("\n COMPARACION DE METODOS NUMERICOS ")

    beta = 0.5
    gamma = 0.1
    y0 = [0.99, 0.01]
    dt = 0.1

    results, (tb, yb) = compare_methods(
        sir_rhs,
        (0, 60),
        y0,
        params=[beta, gamma],
        dt=dt
    )

    print("\nMetodo            Tiempo (s)    Error relativo")
    print("-----------------------------------------------")
    for name, ttime, err in results:
        print(f"{name:18} {ttime:.5f}        {err:.6f}")

    plt.figure()
    plt.plot(tb, yb[:,1], label="RK45 Benchmark")

    from solvers import euler_explicit, euler_improved, rk4
    ts_e, ys_e = euler_explicit(sir_rhs, (0,60), y0, dt, beta, gamma)
    ts_em, ys_em = euler_improved(sir_rhs, (0,60), y0, dt, beta, gamma)
    ts_rk, ys_rk = rk4(sir_rhs, (0,60), y0, dt, beta, gamma)

    plt.plot(ts_e, ys_e[:,1], "--", label="Euler")
    plt.plot(ts_em, ys_em[:,1], "--", label="Euler Mejorado")
    plt.plot(ts_rk, ys_rk[:,1], "--", label="RK4")

    plt.title("Comparacion de Metodos Numericos")
    plt.xlabel("t")
    plt.ylabel("I(t)")
    plt.grid()
    plt.legend()
    fname = os.path.join(OUTPUT_DIR, "comparacion_metodos.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()

    # Barrido en dt para error/tiempo
    dt_values = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    methods = ["Euler", "Euler mejorado", "RK4"]
    errors = {m: [] for m in methods}
    times = {m: [] for m in methods}

    for dtv in dt_values:
        res, (tb, yb) = compare_methods(sir_rhs, (0,60), y0, params=[beta, gamma], dt=dtv)
        for name, ttime, err in res:
            errors[name].append(err)
            times[name].append(ttime)

    # Graficar error vs dt
    plt.figure()
    for name in methods:
        plt.loglog(dt_values, errors[name], marker='o', label=name)
    plt.gca().invert_xaxis()
    plt.xlabel('dt')
    plt.ylabel('Error (norm)')
    plt.title('Convergencia: Error vs dt')
    plt.grid(True, which='both')
    plt.legend()
    fname = os.path.join(OUTPUT_DIR, 'error_vs_dt.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()

    # Graficar tiempo vs dt
    plt.figure()
    for name in methods:
        plt.plot(dt_values, times[name], marker='o', label=name)
    plt.xlabel('dt')
    plt.ylabel('Tiempo (s)')
    plt.title('Tiempo de computo vs dt')
    plt.grid(True)
    plt.legend()
    fname = os.path.join(OUTPUT_DIR, 'time_vs_dt.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    run_sir()
    run_retrato_fase()
    run_rumor_general()
    run_maki()
    # funciones añadidas: barrido SIR y comparacion SIR vs rumor
    run_compare_sir_rumor()
    run_comparacion_metodos()
    try:
        run_sir_sweep()
    except Exception as e:
        print("run_sir_sweep failed:", e)
    try:
        run_maki_phase_plot()
    except Exception as e:
        print("run_maki_phase_plot failed:", e)


    print("\n EJECUCION COMPLETA ")
