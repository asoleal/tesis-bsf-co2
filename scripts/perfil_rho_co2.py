#!/usr/bin/env python3
"""Perfil de costo de rho; reoptimiza a_max y f_D4 en cada punto."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validacion_cruzada_co2 as v
from sensibilidad_co2 import co2_acumulado_g


def main():
    crec, ctrl, gas = v.cargar_datos()
    del ctrl

    micro_df = pd.read_csv("resultados/calibracion_etapa1.csv")
    micro = {
        d: (float(g["k_ref"].iloc[0]), float(g["delta_CS"].iloc[0]))
        for d, g in micro_df.groupby("dieta")
    }

    p = pd.read_csv(
        "resultados/parametros_etapa2.csv"
    ).set_index("parametro")
    rho_canonico = float(p.loc["rho", "valor"])

    obs_crec = [
        (
            r.dieta, float(r.dia_larvario), float(r.ph_mg),
            max(float(r.de_mg), 0.1 * float(r.ph_mg))
        )
        for r in crec.itertuples(index=False)
        if r.dia_larvario <= v.e2.DIA_AJUSTE_MAX
    ]

    obs_gas = [
        (
            r.dieta, float(r.dia_larvario),
            float(r.tasa_ppm_min_tiempo), bool(r.Saturado)
        )
        for r in gas.itertuples(index=False)
    ]

    lo = np.array([v.e2.LO[0], v.e2.LO[2]])
    hi = np.array([v.e2.HI[0], v.e2.HI[2]])
    arranques = [
        np.array([x[0], x[2]]) for x in v.e2.ARRANQUES
    ]

    def ajustar(rho):
        def residuos(theta):
            amax, f_d4 = 10.0 ** theta[0], theta[1]
            curvas = {
                d: v.simular(d, amax, rho, f_d4, micro, "C")
                for d in ("D1", "D4")
            }
            res = []

            for d, dia, obs, w in obs_crec:
                t, _, ph = curvas[d]
                res.append((np.interp(dia, t, ph) - obs) / w)

            for d, dia, obs, sat in obs_gas:
                t, ppm, _ = curvas[d]
                r = np.interp(dia, t, ppm) - obs
                res.append(
                    (min(r, 0.0) if sat else r) / v.e2.PESO_GAS
                )

            return np.asarray(res)

        mejor = None
        for x in arranques:
            x0 = np.clip(x, lo + 1e-9, hi - 1e-9)
            sol = least_squares(
                residuos, x0, bounds=(lo, hi), method="trf"
            )
            if mejor is None or sol.cost < mejor.cost:
                mejor = sol

        return (
            10.0 ** mejor.x[0],
            float(mejor.x[1]),
            float(mejor.cost),
        )

    valores = sorted(set([
        0.5, 0.75, 1.0, 2.0, 3.0, rho_canonico,
        6.0, 8.0, 10.0, 12.0, 15.0,
    ]))

    filas = []

    for rho in valores:
        amax, f_d4, costo = ajustar(rho)
        salida = {
            "rho": rho,
            "a_max": amax,
            "f_D4": f_d4,
            "costo": costo,
        }
        crudos, censurados = [], []

        for d in ("D1", "D4"):
            t, ppm, _ = v.simular(d, amax, rho, f_d4, micro, "C")
            salida[f"co2_{d}_g"] = co2_acumulado_g(t, ppm)

            for r in gas[gas["dieta"] == d].itertuples(index=False):
                residuo = float(
                    np.interp(r.dia_larvario, t, ppm)
                    - r.tasa_ppm_min_tiempo
                )
                crudos.append(residuo)
                censurados.append(
                    min(residuo, 0.0) if r.Saturado else residuo
                )

        salida["co2_total_g"] = (
            salida["co2_D1_g"] + salida["co2_D4_g"]
        )
        salida["rmse_crudo"] = np.sqrt(
            np.mean(np.asarray(crudos) ** 2)
        )
        salida["rmse_censurado"] = np.sqrt(
            np.mean(np.asarray(censurados) ** 2)
        )
        filas.append(salida)

        print(
            f"rho={rho:.4f}, a_max={amax:.4f}, "
            f"f_D4={f_d4:.4f}, costo={costo:.4f}, "
            f"CO2_total={salida['co2_total_g']:.3f} g",
            flush=True,
        )

    perfil = pd.DataFrame(filas)
    costo_min = perfil["costo"].min()

    perfil["delta_costo_pct"] = 100.0 * (
        perfil["costo"] / costo_min - 1.0
    )

    co2_canonico = perfil.loc[
        np.isclose(perfil["rho"], rho_canonico), "co2_total_g"
    ].iloc[0]

    perfil["cambio_co2_total_vs_canonico_pct"] = 100.0 * (
        perfil["co2_total_g"] / co2_canonico - 1.0
    )

    cercanos = perfil[perfil["delta_costo_pct"] <= 1.0]

    destino = Path("resultados/sensibilidad_co2")
    destino.mkdir(parents=True, exist_ok=True)
    perfil.to_csv(destino / "perfil_rho.csv", index=False)

    print("\n=== Perfil de rho ===")
    print(perfil[[
        "rho", "a_max", "f_D4", "costo", "delta_costo_pct",
        "co2_D1_g", "co2_D4_g", "co2_total_g",
        "cambio_co2_total_vs_canonico_pct", "rmse_censurado",
    ]].round(4).to_string(index=False))

    print("\n=== Region exploratoria con aumento de costo <= 1 % ===")
    print(
        f"rho: {cercanos['rho'].min():.4f} a "
        f"{cercanos['rho'].max():.4f}; "
        f"CO2 total: {cercanos['co2_total_g'].min():.3f} a "
        f"{cercanos['co2_total_g'].max():.3f} g"
    )
    print("No es un intervalo de confianza estadistico.")
    print(f"Archivo: {destino / 'perfil_rho.csv'}")
    print("PERFIL DE RHO OK")


if __name__ == "__main__":
    main()
