# Definicion de los modelos (SIR, Maki-Thompson)

import numpy as np


def sir_rhs(t, y, beta, gamma):
    S, I = y
    dS = -beta * S * I
    dI = beta * S * I - gamma * I
    return np.array([dS, dI])

def sir_rhs_full(t, y, beta, gamma):
    S, I, R = y
    dS = -beta * S * I
    dI = beta * S * I - gamma * I
    dR = gamma * I
    return np.array([dS, dI, dR])

def rumor_general_rhs(t, y, lam, delta, alpha):
    X, Y = y
    dX = -lam * X * Y
    dY = lam * X * Y - delta * Y - alpha * Y * (1 - X)
    return np.array([dX, dY])

def maki_thompson_rhs(t, y, lam, alpha):
    X, Y = y
    dX = -lam * X * Y
    dY = lam * X * Y - alpha * Y * (1 - X)
    return np.array([dX, dY])
