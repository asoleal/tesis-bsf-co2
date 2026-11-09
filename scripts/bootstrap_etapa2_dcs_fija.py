#!/usr/bin/env python3
"""Bootstrap condicional: propaga etapa 1 y reajusta solo etapa 2."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import calibrar_etapa2 as e2

SEMILLA = 20261109


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--bootstrap1",
                    default="resultados/dcs_fija/bootstrap_etapa1.csv")
    ap.add_argument("--calibracion2",
                    default="resultados/dcs_fija/calibracion_etapa2.csv")
    ap.add_argument("--parametros2",
                    default="resultados/dcs_fija/parametros_etapa2.csv")
    ap.add_argument("--crecimiento",
                    default="datos/procesados/crecimiento_canonico.csv")
    ap.add_argument("--salida",
                    default="resultados/dcs_fija/bootstrap_etapa2.csv")
    args = ap.parse_args()

    b1 = pd.read_csv(args.bootstrap1)
    base = pd.read_csv(args.calibracion2)
    pars = pd.read_csv(args.parametros2).set_index("parametro")["valor"]
    crec = pd.read_csv(args.crecimiento)

    a0 = float(pars["a_max"])
    r0 = float(pars["rho"])
    f0 = float(pars["f_D4"])
    x0 = np.clip(
        np.array([np.log10(a0), np.log10(r0), f0]),
        e2.LO + 1e-9, e2.HI - 1e-9,
    )

    e2.PH0 = {
        d: float(g.loc[g["dia_larvario"] == 0, "ph_mg"].iloc[0])
        for d, g in crec.groupby("dieta")
    }
    pesos = {
        (str(r["dieta"]), float(r["dia_larvario"])):
            max(float(r["de_mg"]), 0.1 * float(r["ph_mg"]))
        for _, r in crec.iterrows()
    }

    gc = base[
        (base["bloque"] == "crecimiento") & base["en_ajuste"]
    ].copy()
    gc["w"] = [
        pesos[(str(r["dieta"]), float(r["dia_larvario"]))]
        for _, r in gc.iterrows()
    ]
    gg = base[base["bloque"] == "gas"].copy()
    gn = gg[~gg["saturado"]]

    assert len(gc) > 0 and len(gn) > 0
    pool_crec = (gc["residuo"] / gc["w"]).to_numpy(float)
    pool_gas = (gn["residuo"] / e2.PESO_GAS).to_numpy(float)
    t_eval = np.arange(0.0, e2.T_FIN + 0.1, 0.1)
    rng = np.random.default_rng(SEMILLA)
    filas = []

    for b in range(args.n):
        micro = {}
        for dieta in ("D1", "D4"):
            z = b1[(b1["dieta"] == dieta) & (b1["replica"] == b)]
            assert len(z) == 1, (dieta, b, len(z))
            micro[dieta] = (
                float(z["k_ref"].iloc[0]),
                float(z["delta_CS"].iloc[0]),
            )

        crec_b = [
            (
                str(r["dieta"]), float(r["dia_larvario"]),
                float(r["mod"]) + rng.choice(pool_crec) * float(r["w"]),
                float(r["w"]),
            )
            for _, r in gc.iterrows()
        ]
        gas_b = [
            (
                str(r["dieta"]), float(r["dia_larvario"]),
                float(r["obs"]) if bool(r["saturado"]) else
                float(r["mod"]) + rng.choice(pool_gas) * e2.PESO_GAS,
                bool(r["saturado"]),
            )
            for _, r in gg.iterrows()
        ]

        def residuos(theta):
            amax, rho, f_d4 = 10 ** theta[0], 10 ** theta[1], theta[2]
            curvas = {}
            for dieta in ("D1", "D4"):
                t, ppm, ph, _ = e2.simular_dieta(
                    dieta, amax, rho, f_d4, micro, t_eval
                )
                curvas[dieta] = (t, ppm, ph)

            res = []
            for dieta, dia, obs, w in crec_b:
                t, _, ph = curvas[dieta]
                res.append((np.interp(dia, t, ph) - obs) / w)
            for dieta, dia, obs, sat in gas_b:
                t, ppm, _ = curvas[dieta]
                r = np.interp(dia, t, ppm) - obs
                if sat:
                    r = min(r, 0.0)
                res.append(r / e2.PESO_GAS)
            return np.asarray(res)

        sol = least_squares(
            residuos, x0, bounds=(e2.LO, e2.HI),
            method="trf", max_nfev=60,
        )
        if not sol.success:
            raise RuntimeError(f"Replica {b}: {sol.message}")

        nombres = ("a_max", "rho", "f_D4")
        frontera = [
            nombre for nombre, val, lo, hi in zip(
                nombres, sol.x, e2.LO, e2.HI
            )
            if abs(val - lo) < 1e-3 or abs(val - hi) < 1e-3
        ]
        filas.append({
            "replica": b,
            "a_max": 10 ** sol.x[0],
            "rho": 10 ** sol.x[1],
            "f_D4": sol.x[2],
            "k_ref_D1": micro["D1"][0],
            "k_ref_D4": micro["D4"][0],
            "frontera": ";".join(frontera),
            "nfev": sol.nfev,
        })
        if (b + 1) % 25 == 0:
            print(f"Replicas completadas: {b + 1}/{args.n}", flush=True)

    salida = pd.DataFrame(filas)
    destino = Path(args.salida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(destino, index=False)

    print("\nResumen: percentiles 2.5, 50 y 97.5 %")
    for nombre in ("a_max", "rho", "f_D4"):
        q = np.percentile(salida[nombre], [2.5, 50, 97.5])
        f = salida["frontera"].str.contains(nombre, regex=False).mean()
        print(
            f"{nombre}: {q[0]:.4f} | {q[1]:.4f} | {q[2]:.4f}; "
            f"frontera={f:.2f}"
        )

    print("\nCorrelaciones de Spearman")
    print(
        salida[["k_ref_D1", "k_ref_D4", "a_max", "rho", "f_D4"]]
        .corr(method="spearman").round(2).to_string()
    )
    print(f"\nArchivo: {destino}")


if __name__ == "__main__":
    main()
