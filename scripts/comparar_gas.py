#!/usr/bin/env python3
"""Compara las tasas de CO2 del modelo con los cierres medidos, sin calibrar.

Tres validaciones separadas:
  1. Controles D1/D4 (alimento sin larvas, N=0): prueba el modulo microbiano.
  2. Larvas en ayuno (camara vacia): las larvas crecieron con alimento y
     solo ayunaron durante el cierre; se evalua el nucleo con a=0 sobre la
     trayectoria alimentada (mantenimiento a talla real).
  3. Tratamientos D1/D4 (N=700): sistema completo larvas + lecho.

Condiciones del montaje (documentadas en datos/README.md):
  - Pulsos de alimento: 250 g el dia larvario 1 y 250 g el dia 5.
  - MS de la dieta (bromatologia propia): D1 = 0.863, D4 = 0.803.
  - V_aire = 13.08 L (estimacion geometrica, sensibilidad +-0.5 L).
  - T = 300 K; N = 700 larvas por recipiente.

Parametros NO medidos (valores de referencia declarados, congelados en este
paso; ninguno se ajusta):
  - delta_C_alim = 0.47 g C/g MS (contenido de carbono de materia organica).
  - delta_CS = 0.6 (fraccion degradable del carbono del alimento).
  - X0 = 1.0 g C (biomasa microbiana efectiva inicial).
  - Lecho: f_dieta = 1 (hipotesis nula), K_D = 50, k_ref = 0.05,
    K_SM = 50, Y_X = 0.4, Y_Cmic = 0.4.

Los cierres con Saturado=True son cotas inferiores: se reportan pero se
excluyen de los RMSE y se marcan en la figura.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from modelo import (
    ParametrosEriksen,
    ParametrosLecho,
    carbono_a_ppm_min,
    estado_inicial_desde_dw,
    rhs,
    tasas_larva,
    tasas_lecho,
)

# Condiciones del montaje
# Primer pulso en t = 0: las larvas llegaron con contenido intestinal
# de la crianza (decision documentada); segundo pulso dia larvario 5.
PULSOS = [(0.0, 250.0), (5.0, 250.0)]  # (dia larvario, g en fresco)
MS_DIETA = {"D1": 0.863, "D4": 0.803}
N_LARVAS = 700
T_K = 300.0
T_C = T_K - 273.15
V_AIRE_L = 13.08
F_MS = 0.325

# Parametros de referencia declarados (no calibrados)
DELTA_C_ALIM = 0.47   # g C / g MS
DELTA_CS = 0.6        # fraccion degradable
X0_G_C = 1.0          # g C microbiano inicial
P_LECHO_REF = ParametrosLecho(
    f_dieta=1.0, K_D=50.0, k_ref=0.05, K_SM=50.0, Y_X=0.4, Y_Cmic=0.4,
)

T_FIN = 14.0
DIA_MONTAJE_SEP = 3  # dia larvario = dia de septiembre - 3 (montaje 3 sep)


def pulso_carbono(dieta: str, g_fresco: float) -> float:
    return g_fresco * MS_DIETA[dieta] * DELTA_C_ALIM * DELTA_CS


def simular_escenario(p_larva, p_lecho, N, XDW0, dieta, con_pulsos, t_eval):
    """Integra por tramos aplicando los pulsos de alimento en sus dias."""
    X0 = X0_G_C if con_pulsos else 0.0
    y = estado_inicial_desde_dw(XDW0, 0.0, X0, p_larva)
    cortes = sorted(set([0.0] + [d for d, _ in PULSOS] + [T_FIN]))
    pulsos_por_dia = dict(PULSOS)

    ts, ys = [], []
    for a, b in zip(cortes[:-1], cortes[1:]):
        if a in pulsos_por_dia and con_pulsos:
            y[2] += pulso_carbono(dieta, pulsos_por_dia[a])
        t_seg = t_eval[(t_eval >= a) & (t_eval <= b)]
        if len(t_seg) < 2:
            t_seg = np.linspace(a, b, 3)
        sol = solve_ivp(
            rhs, (a, b), y, args=(p_larva, p_lecho, N, T_C),
            method="LSODA", t_eval=t_seg, rtol=1e-7, atol=1e-10,
        )
        if not sol.success:
            raise RuntimeError(f"fallo la integracion en [{a}, {b}]: {sol.message}")
        if not np.all(np.isfinite(sol.y)):
            raise AssertionError("la simulacion produjo valores no finitos")
        ts.append(sol.t)
        ys.append(sol.y)
        y = sol.y[:, -1]

    t = np.concatenate(ts)
    Y = np.concatenate(ys, axis=1)
    if not np.all(np.diff(t) >= 0):
        raise AssertionError("tiempos no monotonos al concatenar tramos")
    return t, Y


def ppm_desde_estados(t, Y, p_larva, p_lecho, N):
    R_C = np.empty_like(t)
    for k, (tk, y) in enumerate(zip(t, Y.T)):
        B, Ltot, D_C, X, _, _ = [max(float(v), 0.0) for v in y]
        micro = tasas_lecho(D_C, X, T_C, p_lecho)
        larva = tasas_larva(tk, B, Ltot, p_larva, p_lecho.f_dieta * micro.phi_D)
        R_C[k] = N * larva.rCO2_C / 1000.0 + micro.rCmic
    return carbono_a_ppm_min(R_C, V_AIRE_L, T_K)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cierres", default="datos/procesados/cierres_etiquetados.csv")
    parser.add_argument("--crecimiento", default="datos/procesados/crecimiento_canonico.csv")
    parser.add_argument("--salida", default="resultados/comparacion_gas.csv")
    parser.add_argument("--figura", default="figuras/gas_modelo_vs_datos.png")
    args = parser.parse_args()

    cierres = pd.read_csv(args.cierres)
    crec = pd.read_csv(args.crecimiento)
    p_larva = ParametrosEriksen(tp_min=8.5)
    t_eval = np.arange(0.0, T_FIN + 0.05, 0.05)

    ph0 = {
        d: float(g.loc[g["dia_larvario"] == 0, "ph_mg"].iloc[0])
        for d, g in crec.groupby("dieta")
    }
    ph0_medio = float(np.mean(list(ph0.values())))

    escenarios = {}
    for dieta in ("D1", "D4"):
        for rol, N, con_pulsos in (("tratamiento", N_LARVAS, True), ("control", 0, True)):
            t, Y = simular_escenario(
                p_larva, P_LECHO_REF, N, ph0[dieta] * F_MS, dieta, con_pulsos, t_eval,
            )
            escenarios[(rol, dieta)] = (t, ppm_desde_estados(t, Y, p_larva, P_LECHO_REF, N))
    t_a, Y_a = simular_escenario(
        p_larva, P_LECHO_REF, N_LARVAS, ph0_medio * F_MS, "D1", True, t_eval,
    )
    R_a = np.empty_like(t_a)
    for k, (tk, y) in enumerate(zip(t_a, Y_a.T)):
        B = max(float(y[0]), 0.0)
        Ltot = max(float(y[1]), 0.0)
        larva = tasas_larva(tk, B, Ltot, p_larva, factor_alimento=0.0)
        R_a[k] = N_LARVAS * larva.rCO2_C / 1000.0
    escenarios[("larvas_solas", "ninguna")] = (
        t_a, carbono_a_ppm_min(R_a, V_AIRE_L, T_K),
    )

    validas = cierres[(~cierres["excluir_promedio"]) & (cierres["Gas"] == "co2")].copy()
    if validas.empty:
        raise AssertionError("no hay cierres de CO2 validos")
    validas["dia_larvario"] = validas["Dia_Experimento"] - DIA_MONTAJE_SEP

    filas = []
    for _, c in validas.iterrows():
        clave = (c["condicion"], c["dieta"] if c["condicion"] != "larvas_solas" else "ninguna")
        if clave not in escenarios:
            raise AssertionError(f"sin escenario para {clave}")
        t, ppm = escenarios[clave]
        mod = float(np.interp(c["dia_larvario"], t, ppm))
        obs = float(c["tasa_ppm_min_tiempo"])
        filas.append(dict(
            condicion=c["condicion"], dieta=c["dieta"], subtipo=c["subtipo"],
            dia_larvario=c["dia_larvario"], replica=int(c["replica"]),
            saturado=bool(c["Saturado"]),
            tasa_obs=obs, tasa_mod=mod,
            residuo=mod - obs, ratio=mod / obs if obs > 0 else np.nan,
        ))

    comp = pd.DataFrame(filas)
    destino = Path(args.salida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    comp.to_csv(destino, index=False)
    print(f"Archivo: {destino}")

    print("\nParametros de referencia (no calibrados):")
    print(f"  delta_C_alim={DELTA_C_ALIM}, delta_CS={DELTA_CS}, X0={X0_G_C} g C")
    print(f"  lecho: {P_LECHO_REF}")
    print(f"  V_aire={V_AIRE_L} L, T={T_K} K, N={N_LARVAS}")

    print("\nResumen por condicion/dieta (RMSE sin saturados):")
    for (cond, dieta), g in comp.groupby(["condicion", "dieta"]):
        ns = g[~g["saturado"]]
        rmse = float(np.sqrt(np.mean(ns["residuo"] ** 2))) if len(ns) else float("nan")
        print(
            f"  {cond:12s} {dieta:7s}: n={len(g)} (saturados={int(g['saturado'].sum())}), "
            f"RMSE={rmse:8.1f} ppm/min, ratio mediana={g['ratio'].median():5.2f}"
        )
    print("\nTabla de comparacion:")
    print(comp.round(2).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib no disponible; se omite la figura")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    colores = {"D1": "tab:blue", "D4": "tab:orange"}
    for ax, dieta in zip(axes[:2], ("D1", "D4")):
        for rol, estilo in (("tratamiento", "-"), ("control", "--")):
            t, ppm = escenarios[(rol, dieta)]
            ax.plot(t, ppm, estilo, color=colores[dieta], label=f"modelo {rol}")
        g = comp[(comp["dieta"] == dieta)]
        for rol in ("tratamiento", "control"):
            gg = g[g["condicion"] == rol]
            for sat, cara in ((False, colores[dieta]), (True, "none")):
                p = gg[gg["saturado"] == sat]
                if len(p):
                    ax.plot(
                        p["dia_larvario"], p["tasa_obs"],
                        "^" if sat else "o", color=colores[dieta],
                        markerfacecolor=cara, linestyle="none",
                    )
        ax.set_title(f"Dieta {dieta}")
        ax.set_xlabel("dia larvario")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("CO2 (ppm/min)")

    t, ppm = escenarios[("larvas_solas", "ninguna")]
    axes[2].plot(t, ppm, color="tab:green", label="modelo ayuno")
    g = comp[comp["condicion"] == "larvas_solas"]
    for sat, cara in ((False, "tab:green"), (True, "none")):
        p = g[g["saturado"] == sat]
        if len(p):
            axes[2].plot(p["dia_larvario"], p["tasa_obs"], "^" if sat else "o",
                         color="tab:green", markerfacecolor=cara, linestyle="none")
    axes[2].set_title("Larvas en ayuno")
    axes[2].set_xlabel("dia larvario")
    axes[2].legend(fontsize=7)

    fig.suptitle("Sin calibrar; triangulos = cierre saturado (cota inferior)", fontsize=9)
    fig.tight_layout()
    ruta_fig = Path(args.figura)
    ruta_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_fig, dpi=200)
    print(f"\nFigura: {ruta_fig}")


if __name__ == "__main__":
    main()
