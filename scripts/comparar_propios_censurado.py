#!/usr/bin/env python3
"""Evaluacion E0/M en todos los cierres, con saturados como cotas."""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("resultados/revision_co2")
comp = pd.read_csv(OUT / "comparacion_propios_E0_M.csv")
comp = comp[comp["bloque"] == "gas"].copy()

for modelo in ("E0", "M"):
    r = comp[f"residuo_{modelo}"].to_numpy(float)
    sat = comp["saturado"].to_numpy(bool)
    comp[f"residuo_cens_{modelo}"] = np.where(
        sat, np.minimum(r, 0.0), r
    )
    comp[f"cumple_cota_{modelo}"] = np.where(
        sat, comp[f"mod_{modelo}"] >= comp["obs"], np.nan
    )

filas = []
grupos = list(comp.groupby("dieta")) + [("TOTAL", comp)]

for dieta, g in grupos:
    sat = g["saturado"].to_numpy(bool)
    rmse_e0 = float(np.sqrt(np.mean(g["residuo_cens_E0"] ** 2)))
    rmse_m = float(np.sqrt(np.mean(g["residuo_cens_M"] ** 2)))
    cob_e0 = (
        float(g.loc[sat, "cumple_cota_E0"].mean())
        if sat.any() else np.nan
    )
    cob_m = (
        float(g.loc[sat, "cumple_cota_M"].mean())
        if sat.any() else np.nan
    )
    filas.append({
        "dieta": dieta,
        "n_total": len(g),
        "n_no_saturados": int((~sat).sum()),
        "n_saturados": int(sat.sum()),
        "rmse_cens_E0": rmse_e0,
        "rmse_cens_M": rmse_m,
        "mejora_M_pct": 100 * (rmse_e0 - rmse_m) / rmse_e0,
        "cobertura_cotas_E0": cob_e0,
        "cobertura_cotas_M": cob_m,
    })

resumen = pd.DataFrame(filas)
comp.to_csv(OUT / "comparacion_propios_censurado.csv", index=False)
resumen.to_csv(OUT / "resumen_propios_censurado.csv", index=False)

assert len(comp) == 8
assert int((~comp["saturado"]).sum()) == 4

print("RMSE censurado en ppm/min; cobertura entre 0 y 1")
print(resumen.round(3).to_string(index=False))
print("\nDetalle por cierre:")
print(comp[[
    "dieta", "dia_larvario", "obs", "saturado",
    "mod_E0", "mod_M", "residuo_cens_E0", "residuo_cens_M",
]].round(3).to_string(index=False))
print(f"\nArchivos escritos en: {OUT}")
