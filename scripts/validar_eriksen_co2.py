#!/usr/bin/env python3
"""Verifica rCO2 de Eriksen contra las Figs. 5 y 6, sin recalibrar.

Las observaciones son digitalizaciones aproximadas de las figuras, no datos
exactos de S1 Table. AME y unidades reproducen la Tabla 2: mg CO2/d por larva.
"""
from pathlib import Path
import csv
import runpy
import sys

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from modelo import ParametrosEriksen, estado_inicial_desde_dw, tasas_larva

validador = runpy.run_path(str(RAIZ / "scripts" / "validar_eriksen.py"))
CASOS = {
    c[0]: c for c in validador["CASOS"]
    if c[0].startswith(("Fig5", "Fig6"))
}

AME_PUBLICADO = {
    "Fig5A": 2.5, "Fig5B": 1.7, "Fig5C": 2.1, "Fig5D": 1.8,
    "Fig6A": 3.2, "Fig6B": 1.7, "Fig6C": 3.1, "Fig6D": 3.7,
}

OBS = {
    "Fig5A": [(9, 6.13), (10, 10.53), (11, 14.17), (12, 13.56), (13, 16.88), (14, 19.68), (15, 18.35), (16, 19.47), (17, 18.61), (18, 12.22), (19, 12.34), (20, 10.67), (21, 7.96), (22, 16.14), (23, 4.58), (24, 4.15), (25, 7.04), (26, 6.87)],
    "Fig5B": [(9, 4.14), (10, 8.74), (11, 11.16), (12, 10.20), (13, 12.06), (14, 14.56), (15, 13.06), (16, 16.12), (17, 15.90), (18, 13.70), (19, 13.47), (20, 12.59), (21, 9.20), (22, 8.97), (23, 9.44), (24, 7.33), (25, 4.91), (26, 8.16), (27, 6.52), (28, 9.05)],
    "Fig5C": [(9, 5.84), (10, 9.35), (11, 9.41), (12, 12.63), (13, 12.40), (14, 13.53), (15, 12.23), (16, 14.16), (17, 13.99), (18, 13.98), (19, 8.48), (20, 12.45), (21, 7.84), (22, 9.58), (23, 9.81), (24, 7.53), (25, 9.18), (26, 7.76), (27, 8.29), (28, 7.29)],
    "Fig5D": [(8, 1.40), (9, 3.96), (10, 9.16), (11, 10.94), (12, 11.76), (13, 12.21), (14, 15.88), (15, 11.61), (16, 16.17), (17, 16.33), (18, 14.35), (19, 15.50), (20, 17.25), (21, 14.34), (22, 14.34), (23, 13.75), (24, 14.22), (25, 11.19), (26, 12.12), (27, 8.90), (28, 9.36)],
    "Fig6A": [(8, 2.25), (9, 2.81), (10, 7.01), (11, 7.31), (12, 10.34), (13, 15.89), (14, 9.72), (15, 7.32), (16, 10.23), (17, 8.49), (18, 5.81), (19, 8.28), (20, 7.77), (21, 7.88), (22, 3.69), (23, 5.50), (24, 4.31)],
    "Fig6B": [(8, 3.33), (9, 3.88), (10, 8.36), (11, 6.98), (12, 11.45), (13, 10.85), (14, 14.70), (15, 11.60), (16, 11.92), (17, 11.92), (18, 8.69), (19, 10.24), (20, 7.65), (21, 9.38), (22, 7.42), (23, 6.48)],
    "Fig6C": [(8, 3.06), (9, 3.61), (10, 5.83), (11, 10.51), (12, 8.28), (13, 8.33), (14, 13.62), (15, 13.95), (16, 16.29), (17, 16.72), (18, 13.65), (19, 15.41), (20, 7.80), (21, 8.90), (22, 6.78), (23, 7.00), (24, 7.50)],
    "Fig6D": [(9, 3.01), (10, 4.38), (11, 6.30), (12, 7.26), (13, 11.46), (14, 7.26), (15, 11.10), (16, 12.00), (17, 15.96), (18, 15.96), (19, 12.86), (20, 12.28), (21, 12.00), (22, 8.67)],
}


def rhs(t, y, p):
    r = tasas_larva(t, y[0], y[1], p, factor_alimento=1.0)
    return [r.dB, r.dLtot]


def rco2_mg_dia(t, y, p):
    r = tasas_larva(t, y[0], y[1], p, factor_alimento=1.0)
    return r.rCO2_C * (44.0 / 12.0)  # Conversión del Excel oficial.


def simular(caso):
    nombre, xdw0, edad0, bmax_dw, amax, yb, m, alpha, beta, *_ = caso
    p = ParametrosEriksen(
        amax=amax, alpha=alpha, beta=beta, YB=yb, m=m,
        Bmax0=bmax_dw * 0.49 * 0.9,
    )
    y0 = estado_inicial_desde_dw(xdw0, 0.0, 0.0, p)[:2]
    dias = np.array([d for d, _ in OBS[nombre]], dtype=float)
    obs = np.array([v for _, v in OBS[nombre]], dtype=float)

    sol = solve_ivp(
        lambda t, y: rhs(t, y, p),
        (float(edad0), float(dias.max())), y0,
        dense_output=True, rtol=1e-9, atol=1e-11, max_step=0.02,
    )
    assert sol.success, f"{nombre}: {sol.message}"
    estados = sol.sol(dias)
    mod = np.array([
        rco2_mg_dia(t, estados[:, i], p) for i, t in enumerate(dias)
    ])
    assert np.all(np.isfinite(mod)) and np.all(mod >= 0.0)
    return p, sol, dias, obs, mod


def main():
    assert set(CASOS) == set(OBS) == set(AME_PUBLICADO)
    salida = RAIZ / "resultados" / "revision_co2"
    figuras = RAIZ / "figuras" / "revision_co2"
    salida.mkdir(parents=True, exist_ok=True)
    figuras.mkdir(parents=True, exist_ok=True)

    metricas, puntos = [], []
    fig, axes = plt.subplots(4, 2, figsize=(12, 15), sharex=False)

    for ax, nombre in zip(axes.flat, OBS):
        p, sol, dias, obs, mod = simular(CASOS[nombre])
        residuo = mod - obs
        ame = float(np.mean(np.abs(residuo)))
        rmse = float(np.sqrt(np.mean(residuo**2)))
        sesgo = float(np.mean(residuo))

        malla = np.linspace(dias.min(), dias.max(), 1200)
        estados = sol.sol(malla)
        curva = np.array([rco2_mg_dia(t, estados[:, i], p)
                          for i, t in enumerate(malla)])

        metricas.append({
            "caso": nombre, "n": len(dias),
            "AME_publicado_mg_d": AME_PUBLICADO[nombre],
            "AME_calculado_mg_d": ame,
            "diferencia_AME_mg_d": ame - AME_PUBLICADO[nombre],
            "RMSE_mg_d": rmse, "sesgo_mg_d": sesgo,
            "dia_pico_obs": float(dias[np.argmax(obs)]),
            "dia_pico_mod": float(malla[np.argmax(curva)]),
        })
        puntos.extend({
            "caso": nombre, "dia": float(d), "obs_mg_d": float(o),
            "mod_mg_d": float(m), "residuo_mod_menos_obs": float(m - o),
        } for d, o, m in zip(dias, obs, mod))

        ax.plot(malla, curva, "k-", lw=1.4, label="Modelo")
        ax.plot(dias, obs, "ko", ms=4, label="Digitalizado")
        ax.set_title(f"{nombre}: AME={ame:.2f}; publicado={AME_PUBLICADO[nombre]:.1f}")
        ax.set_xlabel("Edad larvaria (d)")
        ax.set_ylabel("rCO2 (mg CO2/d/larva)")
        ax.grid(alpha=0.2)

    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(figuras / "eriksen_co2.png", dpi=180)
    plt.close(fig)

    for archivo, filas in [
        ("eriksen_co2_metricas.csv", metricas),
        ("eriksen_co2_puntos.csv", puntos),
    ]:
        with (salida / archivo).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0]))
            w.writeheader()
            w.writerows(filas)

    print("AME en mg CO2/d por larva; diferencia = calculado - publicado")
    print("caso    n  AME_pub  AME_calc  dif_AME   RMSE   sesgo  pico_obs pico_mod")
    for r in metricas:
        print(f"{r['caso']:6s} {r['n']:3d} {r['AME_publicado_mg_d']:8.2f} "
              f"{r['AME_calculado_mg_d']:9.3f} {r['diferencia_AME_mg_d']:9.3f} "
              f"{r['RMSE_mg_d']:7.3f} {r['sesgo_mg_d']:7.3f} "
              f"{r['dia_pico_obs']:8.1f} {r['dia_pico_mod']:8.2f}")
    print(f"Archivos escritos en: {salida}")
    print(f"Figura: {figuras / 'eriksen_co2.png'}")


if __name__ == "__main__":
    main()
