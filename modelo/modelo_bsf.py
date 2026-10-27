"""Núcleo larvario de Eriksen y extensión alimento-lecho.

Unidades internas del núcleo: mg C por larva y mg C larva^-1 d^-1.
Unidades del lecho: g C por cámara y g C d^-1.
Este módulo no calibra parámetros.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class ParametrosEriksen:
    """Parámetros del núcleo larvario en equivalentes de carbono."""

    amax: float = 1.0       # d^-1
    alpha: float = 1.0
    beta: float = 1.0
    YB: float = 0.44
    YL: float = 0.42
    m: float = 0.08         # d^-1
    Bmax0: float = 90.0 * 0.49 * 0.9  # mg C; 90 mg MS en el Excel oficial
    tp_min: float = 13.0    # d desde neonato
    rho: float = 1.0        # mg C d^-1
    delta_I: float = 5.0
    delta_LB: float = 0.1
    delta_CB: float = 0.49
    delta_CL: float = 0.79  # valor usado en el Excel oficial de Eriksen
    delta_O: float = 0.9


@dataclass(frozen=True)
class ParametrosLecho:
    """Parámetros de la extensión alimento-lecho."""

    f_dieta: float
    K_D: float          # g C
    k_ref: float        # d^-1
    K_SM: float         # g C
    Y_X: float
    Y_Cmic: float
    Q10: float = 2.0
    T_ref_C: float = 25.0
    theta: float = 0.7
    K_theta: float = 0.15

    def __post_init__(self) -> None:
        if self.Y_X + self.Y_Cmic > 1.0 + 1e-12:
            raise ValueError("Y_X + Y_Cmic no puede ser mayor que 1")


@dataclass(frozen=True)
class TasasLarva:
    instar: int
    Bmax: float
    a_eff: float
    rA: float
    rCm: float
    muB: float
    rB: float
    rCB: float
    sL: float
    rL: float
    rCL: float
    rCO2_C: float
    dB: float
    dLtot: float


@dataclass(frozen=True)
class TasasLecho:
    phi_D: float
    k_eff: float
    rdeg: float
    rCmic: float
    dD_C: float
    dX: float


def Bmax_t(t: float, p: ParametrosEriksen) -> float:
    """Máximo dinámico de biomasa estructural, en mg C."""
    return max(p.Bmax0 - p.rho * max(t - p.tp_min, 0.0), 1e-12)


def instar(B: float, t: float, p: ParametrosEriksen) -> int:
    """Instar operacional usado en el Excel oficial de Eriksen."""
    if B < p.Bmax0 / p.delta_I:
        return 5
    return 7 if B >= Bmax_t(t, p) else 6


def tasas_larva(
    t: float,
    B: float,
    Ltot: float,
    p: ParametrosEriksen,
    factor_alimento: float = 1.0,
) -> TasasLarva:
    """Evalúa el núcleo de Eriksen para una larva.

    factor_alimento=1 reproduce el núcleo original. La extensión usa
    f_dieta*D_C/(K_D+D_C).
    """
    del Ltot  # Ltot no entra en las tasas del núcleo original.
    B = max(float(B), 0.0)
    bm = Bmax_t(t, p)
    I = instar(B, t, p)

    if I < 6:
        a = p.amax
    elif I == 6:
        a = p.amax * max(1.0 - (B / bm) ** p.alpha, 0.0)
    else:
        a = 0.0

    a_eff = a * factor_alimento
    rA = a_eff * B
    rCm = p.m * B
    G = max(1.0 - (B / bm) ** p.beta, 0.0)
    muB = ((a_eff - p.m) / (1.0 + p.YB)) * G
    rB = muB * B if I <= 6 else 0.0
    rCB = p.YB * rB
    sL = rA - (rB + rCm + rCB)
    rL = sL / (1.0 + p.YL) if sL >= 0.0 else sL
    rCL = p.YL * max(rL, 0.0)
    rCO2_C = rCm + rCB + rCL

    return TasasLarva(
        instar=I,
        Bmax=bm,
        a_eff=a_eff,
        rA=rA,
        rCm=rCm,
        muB=muB,
        rB=rB,
        rCB=rCB,
        sL=sL,
        rL=rL,
        rCL=rCL,
        rCO2_C=rCO2_C,
        dB=rB,
        dLtot=rL + p.delta_LB * rB,
    )


def humedad_desde_masas(agua_g: float, materia_seca_g: float) -> float:
    """Fracción húmeda del lecho usada como modificador microbiano."""
    total = agua_g + materia_seca_g
    if total <= 0.0:
        raise ValueError("agua_g + materia_seca_g debe ser positivo")
    return agua_g / total


def tasas_lecho(
    D_C: float,
    X: float,
    T_C: float,
    p: ParametrosLecho,
) -> TasasLecho:
    """Evalúa disponibilidad, degradación y CO₂ microbiano."""
    D_C = max(float(D_C), 0.0)
    X = max(float(X), 0.0)
    phi_D = D_C / (p.K_D + D_C) if D_C > 0.0 else 0.0
    factor_T = p.Q10 ** ((T_C - p.T_ref_C) / 10.0)
    factor_theta = p.theta / (p.K_theta + p.theta)
    k_eff = p.k_ref * factor_T * factor_theta
    rdeg = k_eff * X * D_C / (p.K_SM + D_C) if D_C > 0.0 else 0.0
    rCmic = p.Y_Cmic * rdeg
    return TasasLecho(
        phi_D=phi_D,
        k_eff=k_eff,
        rdeg=rdeg,
        rCmic=rCmic,
        dD_C=-rdeg,
        dX=p.Y_X * rdeg,
    )


def estado_inicial_desde_dw(
    XDW0_mg: float,
    D_C0: float,
    X0: float,
    p: ParametrosEriksen,
) -> np.ndarray:
    """Construye [B, Ltot, D_C, X, C_CO2, C_residuo].

    La conversión inicial de masa seca a carbono reproduce el Excel oficial:
    B0 = XDW0*delta_CB*delta_O.
    """
    B0 = XDW0_mg * p.delta_CB * p.delta_O
    Ltot0 = B0 * p.delta_LB * (p.delta_CL / (p.delta_CB * p.delta_O))
    return np.array([B0, Ltot0, D_C0, X0, 0.0, 0.0], dtype=float)


def rhs(
    t: float,
    y: np.ndarray,
    p_larva: ParametrosEriksen,
    p_lecho: ParametrosLecho,
    N: int,
    T_C: float,
) -> np.ndarray:
    """Sistema completo larva-alimento-microbiota.

    Estados: B y Ltot en mg C/larva; D_C, X, C_CO2 y C_residuo en g C/cámara.
    """
    B, Ltot, D_C, X, _, _ = [max(float(v), 0.0) for v in y]
    micro = tasas_lecho(D_C, X, T_C, p_lecho)
    factor_alimento = p_lecho.f_dieta * micro.phi_D
    larva = tasas_larva(t, B, Ltot, p_larva, factor_alimento)

    dD_C = -(N * larva.rA / 1000.0) - micro.rdeg
    dX = micro.dX
    dCO2 = N * larva.rCO2_C / 1000.0 + micro.rCmic
    dresiduo = (1.0 - p_lecho.Y_X - p_lecho.Y_Cmic) * micro.rdeg

    return np.array([larva.dB, larva.dLtot, dD_C, dX, dCO2, dresiduo])


def simular(
    t_eval: np.ndarray,
    y0: np.ndarray,
    p_larva: ParametrosEriksen,
    p_lecho: ParametrosLecho,
    N: int,
    T_C: float,
) -> object:
    """Integra el sistema con LSODA; no ajusta ningún parámetro."""
    t_eval = np.asarray(t_eval, dtype=float)
    if t_eval.ndim != 1 or len(t_eval) < 2:
        raise ValueError("t_eval debe ser un vector con al menos dos tiempos")
    sol = solve_ivp(
        rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        np.asarray(y0, dtype=float),
        args=(p_larva, p_lecho, N, T_C),
        method="LSODA",
        t_eval=t_eval,
        rtol=1e-7,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"falló la integración: {sol.message}")
    return sol


def componentes(
    sol: object,
    p_larva: ParametrosEriksen,
    p_lecho: ParametrosLecho,
    N: int,
    T_C: float,
) -> dict[str, np.ndarray]:
    """Recalcula flujos y fuentes para cada tiempo simulado."""
    salida: dict[str, list[float]] = {
        "a_eff": [], "rA": [], "rB": [], "rL": [], "rCO2_lar_C": [],
        "rdeg": [], "rCmic": [], "R_C_total": [],
    }
    for t, y in zip(sol.t, sol.y.T):
        B, Ltot, D_C, X, _, _ = [max(float(v), 0.0) for v in y]
        micro = tasas_lecho(D_C, X, T_C, p_lecho)
        larva = tasas_larva(t, B, Ltot, p_larva, p_lecho.f_dieta * micro.phi_D)
        salida["a_eff"].append(larva.a_eff)
        salida["rA"].append(larva.rA)
        salida["rB"].append(larva.rB)
        salida["rL"].append(larva.rL)
        salida["rCO2_lar_C"].append(larva.rCO2_C)
        salida["rdeg"].append(micro.rdeg)
        salida["rCmic"].append(micro.rCmic)
        salida["R_C_total"].append(N * larva.rCO2_C / 1000.0 + micro.rCmic)
    return {k: np.asarray(v) for k, v in salida.items()}


def carbono_a_ppm_min(
    R_C_g_d: np.ndarray | float,
    V_aire_L: float,
    T_K: float,
    P_Pa: float = 101325.0,
) -> np.ndarray | float:
    """Convierte g C d^-1 respirado a pendiente de CO₂ en ppm min^-1."""
    R_u = 8.314462618
    mol_d = np.asarray(R_C_g_d) / 12.0
    return mol_d / (V_aire_L / 1000.0) * (R_u * T_K / P_Pa) * 1e6 / 1440.0
