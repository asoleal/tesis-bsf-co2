#!/usr/bin/env python3
"""Bootstrap residual de etapas 1 y 2, sin modificar las calibraciones canónicas."""
from pathlib import Path
import argparse
import sys
import time
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from scripts import calibrar_etapa1 as e1
from scripts import calibrar_etapa2 as e2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--semilla", type=int, default=20261108)
    ap.add_argument("--salida", default="resultados/bootstrap_parametros.csv")
    args = ap.parse_args()
    rng = np.random.default_rng(args.semilla)
    inicio = time.time()

    micro0 = e2.cargar_microbiano(RAIZ / "resultados/calibracion_etapa1.csv")
    par = pd.read_csv(RAIZ / "resultados/parametros_etapa2.csv").set_index("parametro")["valor"]
    theta0 = np.array([np.log10(par["a_max"]), np.log10(par["rho"]), par["f_D4"]])
    t_eval = np.arange(0.0, e2.T_FIN + 0.1, 0.1)

    cierres = pd.read_csv(RAIZ / "datos/procesados/cierres_etiquetados.csv")
    crec = pd.read_csv(RAIZ / "datos/procesados/crecimiento_canonico.csv")
    e2.PH0 = {
        d: float(g.loc[g["dia_larvario"] == 0, "ph_mg"].iloc[0])
        for d, g in crec.groupby("dieta")
    }

    ctrl = cierres[
        (~cierres["excluir_promedio"])
        & (cierres["Gas"] == "co2")
        & (cierres["condicion"] == "control")
    ].copy()
    ctrl["dia_larvario"] = ctrl["Dia_Experimento"] - e1.DIA_MONTAJE_SEP
    ctrl = ctrl.rename(columns={"tasa_ppm_min_tiempo": "tasa_obs"})

    gas = cierres[
        (~cierres["excluir_promedio"])
        & (cierres["Gas"] == "co2")
        & (cierres["condicion"] == "tratamiento")
    ].copy()
    gas["dia_larvario"] = gas["Dia_Experimento"] - e2.DIA_MONTAJE_SEP

    obs_crec = [
        (d, float(r["dia_larvario"]), float(r["ph_mg"]),
         max(float(r["de_mg"]), .1 * float(r["ph_mg"])))
        for d, g in crec.groupby("dieta") for _, r in g.iterrows()
        if r["dia_larvario"] <= e2.DIA_AJUSTE_MAX
    ]
    obs_gas = [
        (d, float(r["dia_larvario"]), float(r["tasa_ppm_min_tiempo"]), bool(r["Saturado"]))
        for d, g in gas.groupby("dieta") for _, r in g.iterrows()
    ]

    # Predicciones y residuos canónicos de etapa 1.
    base1 = {}
    for dieta, g in ctrl.groupby("dieta"):
        dias = g["dia_larvario"].to_numpy(float)
        obs = g["tasa_obs"].to_numpy(float)
        k, dcs = micro0[dieta]
        t, ppm = e1.simular_control(dieta, k, dcs, t_eval)
        mod = np.interp(dias, t, ppm)
        base1[dieta] = (dias, obs, mod, mod - obs)

    # Predicciones y residuos estandarizados canónicos de etapa 2.
    curvas0 = {
        d: e2.simular_dieta(d, par["a_max"], par["rho"], par["f_D4"], micro0, t_eval)
        for d in ("D1", "D4")
    }
    crec0, gas0 = [], []
    for d, dia, obs, w in obs_crec:
        t, _, ph, _ = curvas0[d]
        mod = float(np.interp(dia, t, ph))
        crec0.append((d, dia, obs, w, mod, (mod - obs) / w))
    for d, dia, obs, sat in obs_gas:
        t, ppm, _, _ = curvas0[d]
        mod = float(np.interp(dia, t, ppm))
        gas0.append((d, dia, obs, sat, mod, (mod - obs) / e2.PESO_GAS))
    res_crec = np.array([r[-1] for r in crec0])
    res_gas = np.array([r[-1] for r in gas0 if not r[3]])

    def ajustar1(dieta, dias, obs):
        k, dcs = micro0[dieta]
        x0 = np.array([np.log10(k), dcs])
        lo = np.array([e1.BOUNDS["log10_k_ref"][0], e1.BOUNDS["delta_CS"][0]])
        hi = np.array([e1.BOUNDS["log10_k_ref"][1], e1.BOUNDS["delta_CS"][1]])

        def residuos(theta):
            t, ppm = e1.simular_control(dieta, 10 ** theta[0], theta[1], t_eval)
            return np.interp(dias, t, ppm) - obs

        sol = least_squares(
            residuos, np.clip(x0, lo + 1e-9, hi - 1e-9),
            bounds=(lo, hi), max_nfev=60,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        return 10 ** sol.x[0], sol.x[1]

    def residuos2(theta, micro, crec_syn, gas_syn):
        amax, rho, f_d4 = 10 ** theta[0], 10 ** theta[1], theta[2]
        curvas = {
            d: e2.simular_dieta(d, amax, rho, f_d4, micro, t_eval)
            for d in ("D1", "D4")
        }
        res = []
        for d, dia, obs, w in crec_syn:
            t, _, ph, _ = curvas[d]
            res.append((np.interp(dia, t, ph) - obs) / w)
        for d, dia, obs, sat in gas_syn:
            t, ppm, _, _ = curvas[d]
            r = np.interp(dia, t, ppm) - obs
            if sat:
                r = min(r, 0.0)
            res.append(r / e2.PESO_GAS)
        return np.asarray(res)

    filas = []
    destino = Path(args.salida)
    destino.parent.mkdir(parents=True, exist_ok=True)

    for rep in range(1, args.n + 1):
        try:
            micro = {}
            fila = {"replica": rep}
            for dieta, (dias, obs, mod, res) in base1.items():
                obs_syn = mod - rng.choice(res, size=len(res), replace=True)
                k, dcs = ajustar1(dieta, dias, obs_syn)
                micro[dieta] = (k, dcs)
                fila[f"k_ref_{dieta}"] = k
                fila[f"delta_CS_{dieta}"] = dcs

            crec_syn = [
                (d, dia, mod - w * rng.choice(res_crec), w)
                for d, dia, obs, w, mod, _ in crec0
            ]
            gas_syn = []
            for d, dia, obs, sat, mod, _ in gas0:
                obs_syn = obs if sat else mod - e2.PESO_GAS * rng.choice(res_gas)
                gas_syn.append((d, dia, obs_syn, sat))

            sol = least_squares(
                residuos2, np.clip(theta0, e2.LO + 1e-9, e2.HI - 1e-9),
                args=(micro, crec_syn, gas_syn), bounds=(e2.LO, e2.HI),
                max_nfev=60,
            )
            if not sol.success:
                raise RuntimeError(sol.message)
            fila.update(a_max=10 ** sol.x[0], rho=10 ** sol.x[1], f_D4=sol.x[2])
            filas.append(fila)
            pd.DataFrame(filas).to_csv(destino, index=False)
            print(f"replica {rep}/{args.n} OK; {time.time() - inicio:.0f} s")
        except Exception as exc:
            print(f"replica {rep}/{args.n} FALLO: {exc}")

    if not filas:
        raise RuntimeError("ninguna replica termino")
    df = pd.DataFrame(filas)
    limites = {
        "k_ref_D1": (.001, 1), "k_ref_D4": (.001, 1),
        "delta_CS_D1": (.05, .95), "delta_CS_D4": (.05, .95),
        "a_max": (1, 1.5), "rho": (.5, 15), "f_D4": (.4, 1),
    }
    print(f"\nReplicas exitosas: {len(df)}/{args.n}")
    print("Percentiles 2.5, 50 y 97.5 %; frontera = frecuencia en cualquier cota")
    for nombre, (lo, hi) in limites.items():
        q = np.percentile(df[nombre], [2.5, 50, 97.5])
        tol = 1e-3 * (hi - lo)
        frontera = np.mean((df[nombre] <= lo + tol) | (df[nombre] >= hi - tol))
        print(f"{nombre:12s} {q[0]:.4f} | {q[1]:.4f} | {q[2]:.4f}; frontera={frontera:.2f}")
    print(f"CSV: {destino}")


if __name__ == "__main__":
    main()
