#!/usr/bin/env python3
"""Sensibilidad local del modelo C alrededor de la calibracion canonica."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validacion_cruzada_co2 as v

R_U = 8.314462618
P_PA = 101325.0


def co2_acumulado_g(t, ppm_min):
    """Integra la pendiente de camara y la convierte a g CO2."""
    ppm_acumulado = np.trapezoid(ppm_min, t) * 1440.0
    n_aire = P_PA * (v.e2.V_AIRE_L / 1000.0) / (R_U * v.e2.T_K)
    return ppm_acumulado * 1e-6 * n_aire * 44.0


def main():
    crec, ctrl, gas = v.cargar_datos()
    del crec, ctrl

    micro_df = pd.read_csv("resultados/calibracion_etapa1.csv")
    micro0 = {
        d: (float(g["k_ref"].iloc[0]), float(g["delta_CS"].iloc[0]))
        for d, g in micro_df.groupby("dieta")
    }

    p = pd.read_csv(
        "resultados/parametros_etapa2.csv"
    ).set_index("parametro")

    base = {
        "a_max": float(p.loc["a_max", "valor"]),
        "rho": float(p.loc["rho", "valor"]),
        "f_D4": float(p.loc["f_D4", "valor"]),
        "k_ref_D1": micro0["D1"][0],
        "delta_CS_D1": micro0["D1"][1],
        "k_ref_D4": micro0["D4"][0],
        "delta_CS_D4": micro0["D4"][1],
    }

    limites = {
        "a_max": (1.0, 1.5),
        "rho": (0.5, 15.0),
        "f_D4": (0.4, 1.0),
        "k_ref_D1": (0.001, 1.0),
        "delta_CS_D1": (0.05, 0.95),
        "k_ref_D4": (0.001, 1.0),
        "delta_CS_D4": (0.05, 0.95),
    }

    escenarios = [("canonico", "canonico", 0.0, np.nan)]

    for nombre, valor in base.items():
        lo, hi = limites[nombre]
        if abs(valor - hi) < 1e-3:
            factores = [-0.10]
        elif abs(valor - lo) < 1e-3:
            factores = [0.10]
        else:
            factores = [-0.10, 0.10]

        for factor in factores:
            nuevo = valor * (1.0 + factor)
            assert lo <= nuevo <= hi, (nombre, nuevo)
            escenarios.append((nombre, f"{factor:+.0%}", factor, nuevo))

    def evaluar(parametro, nuevo_valor):
        amax = base["a_max"]
        rho = base["rho"]
        f_d4 = base["f_D4"]
        micro = dict(micro0)

        if parametro == "a_max":
            amax = nuevo_valor
        elif parametro == "rho":
            rho = nuevo_valor
        elif parametro == "f_D4":
            f_d4 = nuevo_valor
        elif parametro.startswith("k_ref_"):
            d = parametro[-2:]
            micro[d] = (nuevo_valor, micro[d][1])
        elif parametro.startswith("delta_CS_"):
            d = parametro[-2:]
            micro[d] = (micro[d][0], nuevo_valor)

        salida, detalle = {}, []
        crudos, censurados = [], []

        for d in ("D1", "D4"):
            t, ppm, _ = v.simular(d, amax, rho, f_d4, micro, "C")
            salida[f"co2_{d}_g"] = co2_acumulado_g(t, ppm)
            salida[f"pico_{d}_ppm_min"] = float(np.max(ppm))
            salida[f"dia_pico_{d}"] = float(t[np.argmax(ppm)])

            for r in gas[gas["dieta"] == d].itertuples(index=False):
                mod = float(np.interp(r.dia_larvario, t, ppm))
                obs = float(r.tasa_ppm_min_tiempo)
                residuo = mod - obs
                crudos.append(residuo)
                censurados.append(
                    min(residuo, 0.0) if r.Saturado else residuo
                )
                detalle.append(dict(
                    dieta=d,
                    dia_larvario=float(r.dia_larvario),
                    Muestra_ID=r.Muestra_ID,
                    obs_ppm_min=obs,
                    mod_ppm_min=mod,
                    saturado=bool(r.Saturado),
                ))

        salida["co2_total_g"] = (
            salida["co2_D1_g"] + salida["co2_D4_g"]
        )
        salida["rmse_crudo"] = float(
            np.sqrt(np.mean(np.asarray(crudos) ** 2))
        )
        salida["rmse_censurado"] = float(
            np.sqrt(np.mean(np.asarray(censurados) ** 2))
        )
        return salida, detalle

    filas, predicciones = [], []

    for parametro, etiqueta, factor, nuevo_valor in escenarios:
        es_canonico = parametro == "canonico"
        valores, detalle = evaluar(
            parametro,
            base["a_max"] if es_canonico else nuevo_valor,
        )
        filas.append(dict(
            parametro=parametro,
            perturbacion=etiqueta,
            valor=(base["a_max"] if es_canonico else nuevo_valor),
            **valores,
        ))

        for r in detalle:
            predicciones.append(dict(
                parametro=parametro,
                perturbacion=etiqueta,
                **r,
            ))

    resultado = pd.DataFrame(filas)
    canonico = resultado[
        resultado["parametro"] == "canonico"
    ].iloc[0]

    assert abs(canonico["rmse_censurado"] - 85.1604) < 0.05
    assert abs(canonico["rmse_crudo"] - 98.2363) < 0.05

    resultado["cambio_co2_D1_pct"] = 100.0 * (
        resultado["co2_D1_g"] / canonico["co2_D1_g"] - 1.0
    )
    resultado["cambio_co2_D4_pct"] = 100.0 * (
        resultado["co2_D4_g"] / canonico["co2_D4_g"] - 1.0
    )
    resultado["cambio_co2_total_pct"] = 100.0 * (
        resultado["co2_total_g"] / canonico["co2_total_g"] - 1.0
    )
    resultado["delta_rmse_censurado"] = (
        resultado["rmse_censurado"] - canonico["rmse_censurado"]
    )

    esc = resultado[resultado["parametro"] != "canonico"].copy()
    resumen = esc.groupby("parametro").agg(
        max_cambio_D1_pct=(
            "cambio_co2_D1_pct", lambda x: x.abs().max()
        ),
        max_cambio_D4_pct=(
            "cambio_co2_D4_pct", lambda x: x.abs().max()
        ),
        max_cambio_total_pct=(
            "cambio_co2_total_pct", lambda x: x.abs().max()
        ),
        max_delta_rmse=(
            "delta_rmse_censurado", lambda x: x.abs().max()
        ),
    ).sort_values("max_cambio_total_pct", ascending=False)

    destino = Path("resultados/sensibilidad_co2")
    destino.mkdir(parents=True, exist_ok=True)

    resultado.to_csv(destino / "escenarios.csv", index=False)
    resumen.to_csv(destino / "resumen_parametros.csv")
    pd.DataFrame(predicciones).to_csv(
        destino / "predicciones_observadas.csv", index=False
    )

    print("=== Escenario canonico ===")
    print(canonico[[
        "co2_D1_g", "co2_D4_g", "co2_total_g",
        "pico_D1_ppm_min", "pico_D4_ppm_min",
        "rmse_crudo", "rmse_censurado",
    ]].astype(float).round(4).to_string())

    print("\n=== Escenarios; cambios respecto al canonico ===")
    print(esc[[
        "parametro", "perturbacion", "valor",
        "cambio_co2_D1_pct", "cambio_co2_D4_pct",
        "cambio_co2_total_pct", "delta_rmse_censurado",
    ]].round(3).to_string(index=False))

    print("\n=== Maximo cambio absoluto por parametro ===")
    print(resumen.round(3).to_string())
    print(f"\nResultados: {destino}")
    print("SENSIBILIDAD LOCAL OK")


if __name__ == "__main__":
    main()
