#!/usr/bin/env python3
"""Compara el nucleo larvario con el crecimiento observado, sin calibracion.

Hipotesis nula: ambas dietas se simulan con los parametros publicados de
Eriksen y factor_alimento = 1 (sin efecto de dieta ni limitacion por D_C).
Los residuos cuantifican cuanto falta por explicar y motivan la extension.

Conversion modelo -> peso humedo (decision metodologica A):
  m_DW = B/(delta_CB*delta_O) + Ltot/delta_CL
  PH_modelo = m_DW / f_MS, con banda de sensibilidad f_MS en [0.30, 0.35].

Supuestos declarados:
  - f_MS central 0.325 (literatura BSF, larva viva; no medido en el ensayo).
  - Estado inicial desde PH del dia 0: XDW0 = PH0 * f_MS central.
  - tp_min = 13 d de Eriksen cuenta desde neonato; nuestras larvas ya
    tenian ~4 mg PH al montaje, asi que un desfase temporal es esperable
    y se cuantifica en los residuos (no se ajusta aqui).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from modelo import ParametrosEriksen, estado_inicial_desde_dw, tasas_larva

F_MS_CENTRAL = 0.325
F_MS_BANDA = (0.30, 0.35)


def simular_nucleo(p: ParametrosEriksen, XDW0: float, t_eval: np.ndarray):
    """Integra solo B y Ltot con factor_alimento = 1."""

    def rhs_nucleo(t, y):
        B, Ltot = y
        tasas = tasas_larva(t, B, Ltot, p, factor_alimento=1.0)
        return [tasas.dB, tasas.dLtot]

    y0 = estado_inicial_desde_dw(XDW0, 0.0, 0.0, p)[:2]
    sol = solve_ivp(
        rhs_nucleo,
        (float(t_eval[0]), float(t_eval[-1])),
        y0,
        method="LSODA",
        t_eval=t_eval,
        rtol=1e-7,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"fallo la integracion: {sol.message}")
    if not np.all(np.isfinite(sol.y)):
        raise AssertionError("la simulacion produjo valores no finitos")
    return sol


def ph_desde_estados(B, Ltot, p: ParametrosEriksen, f_ms: float) -> np.ndarray:
    m_dw = np.asarray(B) / (p.delta_CB * p.delta_O) + np.asarray(Ltot) / p.delta_CL
    return m_dw / f_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crecimiento", default="datos/procesados/crecimiento_canonico.csv")
    parser.add_argument("--salida", default="resultados/comparacion_crecimiento.csv")
    parser.add_argument("--figura", default="figuras/crecimiento_modelo_vs_datos.png")
    args = parser.parse_args()

    obs = pd.read_csv(args.crecimiento)
    p = ParametrosEriksen()
    t_eval = np.arange(0.0, 14.0 + 0.05, 0.1)

    filas = []
    curvas = {}
    for dieta, g in obs.groupby("dieta"):
        ph0 = float(g.loc[g["dia_larvario"] == 0, "ph_mg"].iloc[0])
        sol = simular_nucleo(p, ph0 * F_MS_CENTRAL, t_eval)
        B, Ltot = sol.y
        curvas[dieta] = {
            "t": sol.t,
            "ph_central": ph_desde_estados(B, Ltot, p, F_MS_CENTRAL),
            "ph_bajo": ph_desde_estados(B, Ltot, p, F_MS_BANDA[1]),
            "ph_alto": ph_desde_estados(B, Ltot, p, F_MS_BANDA[0]),
        }
        for _, fila in g.iterrows():
            dia = float(fila["dia_larvario"])
            k = int(np.argmin(np.abs(sol.t - dia)))
            ph_mod = float(curvas[dieta]["ph_central"][k])
            filas.append(dict(
                dieta=dieta,
                dia_larvario=dia,
                ph_obs=float(fila["ph_mg"]),
                de_obs=float(fila["de_mg"]),
                ph_modelo=ph_mod,
                residuo=ph_mod - float(fila["ph_mg"]),
                residuo_relativo=(ph_mod - float(fila["ph_mg"])) / float(fila["ph_mg"]),
            ))

    comp = pd.DataFrame(filas)
    destino = Path(args.salida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    comp.to_csv(destino, index=False)
    print(f"Archivo: {destino}")

    print("\nResiduos por dieta (f_MS central = %.3f):" % F_MS_CENTRAL)
    for dieta, g in comp.groupby("dieta"):
        rmse = float(np.sqrt(np.mean(g["residuo"] ** 2)))
        sesgo = float(np.mean(g["residuo"]))
        print(f"  {dieta}: RMSE = {rmse:7.1f} mg PH; sesgo medio = {sesgo:+7.1f} mg PH")
    print("\nTabla de comparacion:")
    print(comp.round(2).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib no disponible; se omite la figura")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colores = {"D1": "tab:blue", "D4": "tab:orange"}
    for dieta, g in obs.groupby("dieta"):
        c = colores.get(dieta, "tab:gray")
        curva = curvas[dieta]
        ax.fill_between(curva["t"], curva["ph_bajo"], curva["ph_alto"], color=c, alpha=0.15)
        ax.plot(curva["t"], curva["ph_central"], color=c, label=f"modelo {dieta}")
        ax.errorbar(
            g["dia_larvario"], g["ph_mg"], yerr=g["de_mg"],
            fmt="o", color=c, capsize=3, label=f"observado {dieta}",
        )
    ax.set_xlabel("dia larvario")
    ax.set_ylabel("PH (mg/larva)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    ruta_fig = Path(args.figura)
    ruta_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_fig, dpi=200)
    print(f"\nFigura: {ruta_fig}")


if __name__ == "__main__":
    main()
