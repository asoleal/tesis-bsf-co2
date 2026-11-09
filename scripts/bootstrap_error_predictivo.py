#!/usr/bin/env python3
"""Bootstrap exploratorio por dieta y dia de metricas fuera de muestra."""

from pathlib import Path

import numpy as np
import pandas as pd

N_BOOT = 5000
SEMILLA = 20261109


def main():
    p = pd.read_csv("resultados/validacion_co2/predicciones.csv")

    grupos = {
        d: [g.copy() for _, g in gd.groupby("dia_larvario")]
        for d, gd in p.groupby("dieta")
    }

    rng = np.random.default_rng(SEMILLA)
    filas = []

    for b in range(N_BOOT):
        partes = []

        for d in ("D1", "D4"):
            disponibles = grupos[d]
            elegidos = rng.integers(
                0, len(disponibles), len(disponibles)
            )
            partes.extend(disponibles[i] for i in elegidos)

        muestra = pd.concat(partes, ignore_index=True)
        rmse = {}

        for modo, g in muestra.groupby("modo"):
            rmse[modo] = float(
                np.sqrt(np.mean(g["residuo_censurado"] ** 2))
            )

        for modo, g in muestra.groupby("modo"):
            ns = g[~g["saturado"]]
            sat = g[g["saturado"]]

            filas.append(dict(
                repeticion=b,
                modo=modo,
                rmse_censurado=rmse[modo],
                rmse_exactos=(
                    float(np.sqrt(np.mean(ns["residuo"] ** 2)))
                    if len(ns) else np.nan
                ),
                cobertura_saturados=(
                    float(np.mean(
                        sat["mod_ppm_min"] >= sat["obs_ppm_min"]
                    ))
                    if len(sat) else np.nan
                ),
                reduccion_vs_E0_pct=(
                    100.0 * (rmse["E0"] - rmse[modo]) / rmse["E0"]
                ),
                n_exactos=len(ns),
            ))

    boot = pd.DataFrame(filas)

    resumen = boot.groupby("modo").agg(
        rmse_cens_p2_5=(
            "rmse_censurado", lambda x: x.quantile(0.025)
        ),
        rmse_cens_mediana=("rmse_censurado", "median"),
        rmse_cens_p97_5=(
            "rmse_censurado", lambda x: x.quantile(0.975)
        ),
        rmse_exactos_p2_5=(
            "rmse_exactos", lambda x: x.quantile(0.025)
        ),
        rmse_exactos_mediana=("rmse_exactos", "median"),
        rmse_exactos_p97_5=(
            "rmse_exactos", lambda x: x.quantile(0.975)
        ),
        reduccion_p2_5=(
            "reduccion_vs_E0_pct", lambda x: x.quantile(0.025)
        ),
        reduccion_mediana=("reduccion_vs_E0_pct", "median"),
        reduccion_p97_5=(
            "reduccion_vs_E0_pct", lambda x: x.quantile(0.975)
        ),
        fraccion_mejora_positiva=(
            "reduccion_vs_E0_pct", lambda x: np.mean(x > 0.0)
        ),
        cobertura_mediana=("cobertura_saturados", "median"),
    )

    destino = Path("resultados/validacion_co2")
    boot.to_csv(destino / "bootstrap_metricas.csv", index=False)
    resumen.to_csv(destino / "resumen_bootstrap.csv")

    print("=== Bootstrap por dieta y dia; percentiles exploratorios ===")
    print(resumen.round(3).to_string())
    print(
        "\nLa fraccion de mejora positiva no es una "
        "probabilidad de superioridad."
    )
    print(
        "Los intervalos describen remuestreo de siete dias, "
        "no validacion independiente."
    )
    print(f"Semilla={SEMILLA}; repeticiones={N_BOOT}")
    print(f"Archivos: {destino}")
    print("BOOTSTRAP EXPLORATORIO OK")


if __name__ == "__main__":
    main()
