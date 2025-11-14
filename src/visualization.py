# Funciones para graficas 

import matplotlib.pyplot as plt
import numpy as np


def plot_time_series(t, curves, labels, title, xlabel="t", ylabel="valor"):
    plt.figure(figsize=(8,5))
    for data, lbl in zip(curves, labels):
        plt.plot(t, data, label=lbl)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_phase(S, I, label, title="Retrato de fase S-I"):
    plt.figure(figsize=(6,6))
    plt.plot(S, I, label=label)
    plt.xlabel("S")
    plt.ylabel("I")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_vector_field(rhs, beta, gamma):
    S_vals = np.linspace(0.01, 1, 20)
    I_vals = np.linspace(0, 0.4, 20)
    S, I = np.meshgrid(S_vals, I_vals)

    dS, dI = rhs(0, np.array([S, I]), beta, gamma)
    plt.figure(figsize=(7,6))
    plt.quiver(S, I, dS, dI, angles='xy', scale_units='xy')
    plt.xlabel("S")
    plt.ylabel("I")
    plt.title("Campo vectorial del SIR")
    plt.grid(True)
    plt.show()
