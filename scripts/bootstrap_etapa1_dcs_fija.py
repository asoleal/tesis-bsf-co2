#!/usr/bin/env python3
"""Bootstrap residual de etapa 1 con delta_CS fijo y k_ref libre."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibrar_etapa1 import BOUNDS, T_FIN, simular_control

SEMILLA = 20261108


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument(
        "--entrada",
        default="resultados/dcs_fija/calibracion_etapa1.csv",
    )
    parser.add_argument(
        "--salida",
        default="resultados/dcs_fija/bootstrap_etapa1.csv",
    )
    args = parser.parse_args()
    assert args.n > 0

    base = pd.read_csv(args.entrada)
    requeridas = {
        "dieta", "dia_larvario", "tasa_obs", "tasa_mod",
        "k_ref", "delta_CS",
    }
    assert requeridas.issubset(base.columns), (
        f"Faltan columnas: {sorted(requeridas - set(base.columns))}"
    )

    rng = np.random.default_rng(SEMILLA)
    filas = []

    for dieta, g0 in base.groupby("dieta", sort=True):
        g = g0.sort_values("dia_larvario")
        dias = g["dia_larvario"].to_numpy(float)
        ajuste = g["tasa_mod"].to_numpy(float)
        residuos = (g["tasa_mod"] - g["tasa_obs"]).to_numpy(float)
        k0 = float(g["k_ref"].iloc[0])
        dcs = float(g["delta_CS"].iloc[0])

        assert np.allclose(g["k_ref"], k0)
        assert np.allclose(g["delta_CS"], dcs)
        assert len(g) >= 2

        print(f"{dieta}: k0={k0:.6f}, delta_CS fija={dcs:.6f}")
        t_eval = np.arange(0.0, T_FIN + 0.1, 0.1)
        lo, hi = BOUNDS["log10_k_ref"]

        for replica in range(args.n):
            obs_b = ajuste + rng.choice(residuos, size=len(g), replace=True)

            def fun(theta):
                t, ppm = simular_control(
                    dieta, 10.0 ** theta[0], dcs, t_eval
                )
                return np.interp(dias, t, ppm) - obs_b

            x0 = np.array([
                np.clip(np.log10(k0), lo + 1e-9, hi - 1e-9)
            ])
            sol = least_squares(
                fun, x0, bounds=([lo], [hi]),
                method="trf", max_nfev=60,
            )
            if not sol.success:
                raise RuntimeError(
                    f"Fallo {dieta}, replica {replica}: {sol.message}"
                )

            logk = float(sol.x[0])
            frontera = ""
            if abs(logk - lo) < 1e-3:
                frontera = "inferior"
            elif abs(logk - hi) < 1e-3:
                frontera = "superior"

            filas.append({
                "dieta": dieta,
                "replica": replica,
                "k_ref": 10.0 ** logk,
                "delta_CS": dcs,
                "rmse": float(np.sqrt(np.mean(sol.fun ** 2))),
                "en_frontera": frontera,
                "nfev": sol.nfev,
            })

    salida = pd.DataFrame(filas)
    destino = Path(args.salida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(destino, index=False)

    print("\nResumen: percentiles 2.5, 50 y 97.5 %")
    for dieta, g in salida.groupby("dieta", sort=True):
        q = np.percentile(g["k_ref"], [2.5, 50, 97.5])
        f = float(np.mean(g["en_frontera"] != ""))
        print(
            f"{dieta}: k_ref={q[0]:.4f} | {q[1]:.4f} | {q[2]:.4f}; "
            f"frontera={f:.2f}"
        )
    print(f"\nArchivo: {destino}")


if __name__ == "__main__":
    main()
