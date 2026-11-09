#!/usr/bin/env python3
"""Comparacion E0/M con datos propios, sin recalibrar parametros."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import calibrar_etapa2 as e2
from modelo import (
    ParametrosEriksen, carbono_a_ppm_min,
    estado_inicial_desde_dw, tasas_larva,
)

OUT = Path("resultados/revision_co2")
OUT.mkdir(parents=True, exist_ok=True)

pars = pd.read_csv(
    "resultados/dcs_fija/parametros_etapa2.csv"
).set_index("parametro")["valor"]
amax = float(pars["a_max"])
rho = float(pars["rho"])

base = pd.read_csv("resultados/dcs_fija/calibracion_etapa2.csv")
crec = pd.read_csv("datos/procesados/crecimiento_canonico.csv")
e2.PH0 = {
    d: float(g.loc[g["dia_larvario"] == 0, "ph_mg"].iloc[0])
    for d, g in crec.groupby("dieta")
}

curvas = {}
for dieta in ("D1", "D4"):
    p = ParametrosEriksen(amax=amax, rho=rho, tp_min=e2.TP_MIN)
    inicial = estado_inicial_desde_dw(
        e2.PH0[dieta] * e2.F_MS, 0.0, e2.X0_G_C, p
    )

    def rhs_e0(t, y, p):
        r = tasas_larva(t, y[0], y[1], p, factor_alimento=1.0)
        return [r.dB, r.dLtot]

    t_eval = np.arange(0.0, e2.T_FIN + 0.05, 0.05)
    sol = solve_ivp(
        rhs_e0, (0.0, e2.T_FIN), inicial[:2], args=(p,),
        method="LSODA", t_eval=t_eval, rtol=1e-7, atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"{dieta}: {sol.message}")

    ppm = np.empty_like(sol.t)
    for k, (t, y) in enumerate(zip(sol.t, sol.y.T)):
        lar = tasas_larva(
            t, max(y[0], 0.0), max(y[1], 0.0), p,
            factor_alimento=1.0,
        )
        ppm[k] = carbono_a_ppm_min(
            e2.N_LARVAS * lar.rCO2_C / 1000.0,
            e2.V_AIRE_L, e2.T_K,
        )

    B = np.maximum(sol.y[0], 0.0)
    Ltot = np.maximum(sol.y[1], 0.0)
    ph = (
        (1.0 - p.delta_LB) * B / (p.delta_CB * p.delta_O)
        + Ltot / p.delta_CL
    ) / e2.F_MS
    curvas[dieta] = (sol.t, ppm, ph)

filas = []
for _, r in base.iterrows():
    dieta = str(r["dieta"])
    dia = float(r["dia_larvario"])
    t, ppm, ph = curvas[dieta]
    es_gas = r["bloque"] == "gas"
    mod_e0 = np.interp(dia, t, ppm if es_gas else ph)
    filas.append({
        "bloque": r["bloque"],
        "dieta": dieta,
        "dia_larvario": dia,
        "obs": float(r["obs"]),
        "mod_E0": mod_e0,
        "mod_M": float(r["mod"]),
        "residuo_E0": mod_e0 - float(r["obs"]),
        "residuo_M": float(r["mod"]) - float(r["obs"]),
        "saturado": bool(r["saturado"]),
        "en_ajuste": bool(r["en_ajuste"]),
    })

comp = pd.DataFrame(filas)
comp.to_csv(OUT / "comparacion_propios_E0_M.csv", index=False)

resumen = []
for bloque in ("gas", "crecimiento"):
    z0 = comp[comp["bloque"] == bloque]
    for dieta, z in z0.groupby("dieta"):
        if bloque == "gas":
            z = z[~z["saturado"]]
        else:
            z = z[z["en_ajuste"]]
        rmse_e0 = float(np.sqrt(np.mean(z["residuo_E0"] ** 2)))
        rmse_m = float(np.sqrt(np.mean(z["residuo_M"] ** 2)))
        resumen.append({
            "bloque": bloque,
            "dieta": dieta,
            "n": len(z),
            "rmse_E0": rmse_e0,
            "rmse_M": rmse_m,
            "mejora_M_pct": 100 * (rmse_e0 - rmse_m) / rmse_e0,
        })

resumen = pd.DataFrame(resumen)
resumen.to_csv(OUT / "resumen_propios_E0_M.csv", index=False)

print("\nCO2: RMSE en ppm/min; mejora positiva favorece a M")
print(
    resumen[resumen["bloque"] == "gas"]
    .round(3).to_string(index=False)
)
print("\nBiomasa auxiliar: RMSE en mg PH")
print(
    resumen[resumen["bloque"] == "crecimiento"]
    .round(3).to_string(index=False)
)
print(f"\nArchivos escritos en: {OUT}")
