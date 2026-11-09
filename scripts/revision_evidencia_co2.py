#!/usr/bin/env python3
"""Consolida evidencia disponible y metricas de CO2 sin recalibrar."""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("resultados/revision_co2")
OUT.mkdir(parents=True, exist_ok=True)

matriz = pd.DataFrame([
    {
        "fuente": "Eriksen",
        "rol": "Verificacion estructural",
        "gas_directo": "No extraido",
        "series_tasa": "No extraidas",
        "acumulado": "No extraido",
        "alimento": "15 escenarios",
        "control_sin_larvas": "No aplica",
        "condiciones_iniciales": "Si",
        "estado": "Nucleo verificado",
    },
    {
        "fuente": "Datos propios",
        "rol": "Calibracion y evaluacion interna",
        "gas_directo": "Si",
        "series_tasa": "8 cierres",
        "acumulado": "No observado directamente",
        "alimento": "D1/D4; caracterizacion propia",
        "control_sin_larvas": "Si",
        "condiciones_iniciales": "Si",
        "estado": "Falta evaluacion interna por bloques",
    },
    {
        "fuente": "Ermolaev",
        "rol": "Transferencia externa",
        "gas_directo": "Si",
        "series_tasa": "3 tasas",
        "acumulado": "Si",
        "alimento": "Food waste; pulsos documentados",
        "control_sin_larvas": "No usado en L",
        "condiciones_iniciales": "Si",
        "estado": "Nivel A parcial",
    },
    {
        "fuente": "Schon",
        "rol": "Transferencia entre dietas",
        "gas_directo": "Si",
        "series_tasa": "Curvas y totales",
        "acumulado": "Si",
        "alimento": "4 dietas P:C",
        "control_sin_larvas": "No usado",
        "condiciones_iniciales": "Si",
        "estado": "A no diferencia las dietas",
    },
    {
        "fuente": "Rossi",
        "rol": "Candidato a nueva transferencia",
        "gas_directo": "Si",
        "series_tasa": "Cada 12 h",
        "acumulado": "Si",
        "alimento": "Chicken feed caracterizado",
        "control_sin_larvas": "Si",
        "condiciones_iniciales": "Si",
        "estado": "Pendiente de incorporar",
    },
])
matriz.to_csv(OUT / "matriz_evidencia.csv", index=False)

filas = []

# Datos propios: son ajuste, no validacion interna independiente.
prop = pd.read_csv("resultados/dcs_fija/calibracion_etapa2.csv")
gas = prop[(prop["bloque"] == "gas") & ~prop["saturado"]]
for dieta, g in gas.groupby("dieta"):
    filas.append({
        "fuente": "Propios",
        "dieta": dieta,
        "evaluacion": "Ajuste en muestra",
        "rmse_tasa": np.sqrt(np.mean(g["residuo"] ** 2)),
    })

# Ermolaev A.
erm = pd.read_csv("resultados/dcs_fija/ermolaev_A.csv")
obs = erm.dropna(subset=["obs_mg_min"]).copy()
f7 = float(obs.loc[obs["dia"] == 7.0, "mod_A_mg_min"].iloc[0])
f11 = float(obs.loc[obs["dia"] == 11.0, "mod_A_mg_min"].iloc[0])
ult = erm.dropna(subset=["acum_obs_g_kg"]).iloc[-1]
filas.append({
    "fuente": "Ermolaev",
    "dieta": "L",
    "evaluacion": "Transferencia A",
    "rmse_tasa": np.sqrt(np.mean(
        (obs["mod_A_mg_min"] - obs["obs_mg_min"]) ** 2
    )),
    "error_total_pct": 100 * (
        float(ult["acum_A_g_kg"]) / float(ult["acum_obs_g_kg"]) - 1
    ),
    "pico_obs_dia": 7.0,
    "pico_mod_dia": float(obs.loc[obs["mod_A_mg_min"].idxmax(), "dia"]),
    "declive_obs_pct": 31.0,
    "declive_mod_pct": 100 * (f7 - f11) / f7,
})

# Schon A.
sch = pd.read_csv("resultados/dcs_fija/schon_A.csv")
for _, r in sch.iterrows():
    filas.append({
        "fuente": "Schon",
        "dieta": r["dieta"],
        "evaluacion": "Transferencia A",
        "error_total_pct": 100 * (r["total_A_L"] / r["total_obs_L"] - 1),
        "pico_obs_dia": r["pico_obs_dia"],
        "pico_mod_dia": r["pico_A_dia"],
    })

metricas = pd.DataFrame(filas)
columnas = [
    "fuente", "dieta", "evaluacion", "rmse_tasa", "error_total_pct",
    "pico_obs_dia", "pico_mod_dia", "declive_obs_pct", "declive_mod_pct",
]
metricas = metricas.reindex(columns=columnas)
metricas["error_pico_dia"] = (
    metricas["pico_mod_dia"] - metricas["pico_obs_dia"]
)
metricas.to_csv(OUT / "metricas_co2.csv", index=False)

print("\nMATRIZ DE EVIDENCIA")
print(matriz.to_string(index=False))
print("\nMETRICAS DE CO2")
print(metricas.round(3).to_string(index=False))
print(f"\nArchivos escritos en: {OUT}")
