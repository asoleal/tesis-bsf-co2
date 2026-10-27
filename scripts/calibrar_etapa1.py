#!/usr/bin/env python3
"""Etapa 1 de calibracion: modulo microbiano contra controles de alimento.

Se calibran por dieta k_ref (escala logaritmica) y delta_CS (fraccion
degradable del carbono del alimento) contra los cierres de control
(alimento sin larvas, N=0), por minimos cuadrados con region de confianza
reflectante y multinicio.

Fijos por no identificabilidad o por literatura (ver datos/README.md):
  X0 = 1 g C (los controles informan el producto k*X, no cada factor),
  Y_X = 0.4, Y_Cmic = 0.4, K_SM = 50 g C, K_D = 50 g C,
  delta_C_alim = 0.47 g C/g MS, Q10 = 2, theta = 0.7, K_theta = 0.15,
  T = 300 K, V_aire = 13.08 L.

Reglas: si un optimo queda en la frontera de su cota, se reporta como tal
(la cota encierra el rango fisicamente plausible; tocarla es un resultado,
no un error). El ayuno y los tratamientos no entran en esta etapa.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from modelo import ParametrosLecho, carbono_a_ppm_min, tasas_lecho

# Condiciones del montaje
PULSOS = [(1.0, 250.0), (5.0, 250.0)]
MS_DIETA = {"D1": 0.863, "D4": 0.803}
T_K = 300.0
T_C = T_K - 273.15
V_AIRE_L = 13.08
DELTA_C_ALIM = 0.47
T_FIN = 14.0
DIA_MONTAJE_SEP = 3

# Fijos declarados
X0_G_C = 1.0
Y_X = 0.4
Y_CMIC = 0.4
K_SM = 50.0
K_D = 50.0

# Cotas fisicamente plausibles
BOUNDS = {
    "log10_k_ref": (-3.0, 0.0),   # k_ref en [1e-3, 1] d^-1
    "delta_CS": (0.05, 0.95),
}
ARRANQUES = [
    (-1.3, 0.6), (-0.3, 0.6), (-1.3, 0.2), (-0.3, 0.9),
]


def pulso_carbono(dieta, g_fresco, delta_CS):
    return g_fresco * MS_DIETA[dieta] * DELTA_C_ALIM * delta_CS


def simular_control(dieta, k_ref, delta_CS, t_eval):
    """Lecho sin larvas: solo los estados [D_C, X], con los dos pulsos."""
    p_lecho = ParametrosLecho(
        f_dieta=1.0, K_D=K_D, k_ref=k_ref, K_SM=K_SM, Y_X=Y_X, Y_Cmic=Y_CMIC,
    )

    def rhs_lecho(t, y):
        D_C, X = [max(float(v), 0.0) for v in y]
        micro = tasas_lecho(D_C, X, T_C, p_lecho)
        return [micro.dD_C, micro.dX]

    y = np.array([0.0, X0_G_C])
    cortes = [0.0] + [d for d, _ in PULSOS] + [T_FIN]
    pulsos = dict(PULSOS)
    ts, ys = [], []
    for a, b in zip(cortes[:-1], cortes[1:]):
        if a > 0.0:
            y[0] += pulso_carbono(dieta, pulsos[a], delta_CS)
        t_seg = t_eval[(t_eval >= a) & (t_eval <= b)]
        if len(t_seg) < 2:
            t_seg = np.linspace(a, b, 3)
        sol = solve_ivp(
            rhs_lecho, (a, b), y, method="LSODA", t_eval=t_seg,
            rtol=1e-7, atol=1e-10,
        )
        if not sol.success:
            raise RuntimeError(f"fallo la integracion en [{a}, {b}]: {sol.message}")
        ts.append(sol.t)
        ys.append(sol.y)
        y = sol.y[:, -1]
    t = np.concatenate(ts)
    Y = np.concatenate(ys, axis=1)
    R_C = np.array([
        tasas_lecho(max(float(y[0]), 0.0), max(float(y[1]), 0.0), T_C, p_lecho).rCmic
        for y in Y.T
    ])
    return t, carbono_a_ppm_min(R_C, V_AIRE_L, T_K)


def ajustar_dieta(dieta, obs):
    t_eval = np.arange(0.0, T_FIN + 0.1, 0.1)
    dias = obs["dia_larvario"].to_numpy(float)
    medidas = obs["tasa_obs"].to_numpy(float)

    def residuos(theta):
        log10_k, dcs = theta
        t, ppm = simular_control(dieta, 10.0 ** log10_k, dcs, t_eval)
        return np.interp(dias, t, ppm) - medidas

    lo = np.array([BOUNDS["log10_k_ref"][0], BOUNDS["delta_CS"][0]])
    hi = np.array([BOUNDS["log10_k_ref"][1], BOUNDS["delta_CS"][1]])
    mejor = None
    for arranque in ARRANQUES:
        x0 = np.clip(np.array(arranque, dtype=float), lo + 1e-9, hi - 1e-9)
        sol = least_squares(residuos, x0, bounds=(lo, hi), method="trf")
        if mejor is None or sol.cost < mejor.cost:
            mejor = sol
    log10_k, dcs = mejor.x
    en_frontera = []
    if abs(log10_k - lo[0]) < 1e-3 or abs(log10_k - hi[0]) < 1e-3:
        en_frontera.append("k_ref")
    if abs(dcs - lo[1]) < 1e-3 or abs(dcs - hi[1]) < 1e-3:
        en_frontera.append("delta_CS")
    return 10.0 ** log10_k, dcs, mejor, en_frontera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cierres", default="datos/procesados/cierres_etiquetados.csv")
    parser.add_argument("--salida", default="resultados/calibracion_etapa1.csv")
    parser.add_argument("--figura", default="figuras/calibracion_etapa1.png")
    args = parser.parse_args()

    cierres = pd.read_csv(args.cierres)
    ctrl = cierres[
        (~cierres["excluir_promedio"])
        & (cierres["Gas"] == "co2")
        & (cierres["condicion"] == "control")
    ].copy()
    if ctrl.empty:
        raise AssertionError("no hay cierres de control validos")
    ctrl["dia_larvario"] = ctrl["Dia_Experimento"] - DIA_MONTAJE_SEP
    ctrl = ctrl.rename(columns={"tasa_ppm_min_tiempo": "tasa_obs"})

    filas = []
    ajustes = {}
    for dieta, g in ctrl.groupby("dieta"):
        k_ref, dcs, sol, frontera = ajustar_dieta(dieta, g)
        ajustes[dieta] = (k_ref, dcs)
        t, ppm = simular_control(dieta, k_ref, dcs, np.arange(0.0, T_FIN + 0.05, 0.05))
        for _, c in g.iterrows():
            mod = float(np.interp(c["dia_larvario"], t, ppm))
            filas.append(dict(
                dieta=dieta, dia_larvario=c["dia_larvario"],
                tasa_obs=c["tasa_obs"], tasa_mod=mod, residuo=mod - c["tasa_obs"],
                k_ref=k_ref, delta_CS=dcs,
                en_frontera=";".join(frontera) if frontera else "",
            ))
        rmse = float(np.sqrt(np.mean((np.interp(g["dia_larvario"], t, ppm) - g["tasa_obs"]) ** 2)))
        marca = f" [frontera: {','.join(frontera)}]" if frontera else ""
        print(f"{dieta}: k_ref={k_ref:.4f} d^-1, delta_CS={dcs:.3f}, RMSE={rmse:.2f} ppm/min{marca}")

    comp = pd.DataFrame(filas)
    destino = Path(args.salida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    comp.to_csv(destino, index=False)
    print(f"\nArchivo: {destino}")
    print("\nTabla:")
    print(comp.round(3).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib no disponible; se omite la figura")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    colores = {"D1": "tab:blue", "D4": "tab:orange"}
    for ax, (dieta, g) in zip(axes, ctrl.groupby("dieta")):
        k_ref, dcs = ajustes[dieta]
        t, ppm = simular_control(dieta, k_ref, dcs, np.arange(0.0, T_FIN + 0.05, 0.05))
        ax.plot(t, ppm, color=colores[dieta], label=f"modelo (k_ref={k_ref:.3f})")
        ax.plot(g["dia_larvario"], g["tasa_obs"], "o", color=colores[dieta], label="control obs")
        ax.set_title(f"Control {dieta}")
        ax.set_xlabel("dia larvario")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("CO2 (ppm/min)")
    fig.suptitle("Etapa 1: modulo microbiano contra controles (alimento sin larvas)", fontsize=9)
    fig.tight_layout()
    ruta_fig = Path(args.figura)
    ruta_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_fig, dpi=200)
    print(f"\nFigura: {ruta_fig}")


if __name__ == "__main__":
    main()
