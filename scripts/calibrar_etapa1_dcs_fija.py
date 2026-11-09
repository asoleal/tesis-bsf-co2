#!/usr/bin/env python3
"""Etapa 1 condicional: delta_CS fijo al valor canónico; solo k_ref se ajusta."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from scripts import calibrar_etapa1 as e1


def main():
    canon = pd.read_csv(RAIZ / "resultados/calibracion_etapa1.csv")
    fijos = {
        d: (float(g["k_ref"].iloc[0]), float(g["delta_CS"].iloc[0]))
        for d, g in canon.groupby("dieta")
    }
    cierres = pd.read_csv(RAIZ / "datos/procesados/cierres_etiquetados.csv")
    ctrl = cierres[
        (~cierres["excluir_promedio"])
        & (cierres["Gas"] == "co2")
        & (cierres["condicion"] == "control")
    ].copy()
    ctrl["dia_larvario"] = ctrl["Dia_Experimento"] - e1.DIA_MONTAJE_SEP
    ctrl = ctrl.rename(columns={"tasa_ppm_min_tiempo": "tasa_obs"})

    t_eval = np.arange(0.0, e1.T_FIN + 0.1, 0.1)
    filas = []
    for dieta, g in ctrl.groupby("dieta"):
        k0, dcs = fijos[dieta]
        dias = g["dia_larvario"].to_numpy(float)
        obs = g["tasa_obs"].to_numpy(float)

        def residuos(theta):
            t, ppm = e1.simular_control(dieta, 10 ** theta[0], dcs, t_eval)
            return np.interp(dias, t, ppm) - obs

        lo = np.array([e1.BOUNDS["log10_k_ref"][0]])
        hi = np.array([e1.BOUNDS["log10_k_ref"][1]])
        mejor = None
        for x0 in ([np.log10(k0)], [-1.3], [-0.3]):
            sol = least_squares(
                residuos, np.clip(x0, lo + 1e-9, hi - 1e-9),
                bounds=(lo, hi), max_nfev=80,
            )
            if mejor is None or sol.cost < mejor.cost:
                mejor = sol
        if not mejor.success:
            raise RuntimeError(mejor.message)

        k = 10 ** mejor.x[0]
        frontera = abs(mejor.x[0] - lo[0]) < 1e-3 or abs(mejor.x[0] - hi[0]) < 1e-3
        t, ppm = e1.simular_control(dieta, k, dcs, t_eval)
        mod = np.interp(dias, t, ppm)
        rmse = float(np.sqrt(np.mean((mod - obs) ** 2)))
        print(f"{dieta}: delta_CS={dcs:.4f} fijo; k_ref={k:.4f}; "
              f"RMSE={rmse:.2f}; frontera={frontera}")

        for dia, ob, mo in zip(dias, obs, mod):
            filas.append(dict(
                dieta=dieta, dia_larvario=dia, tasa_obs=ob, tasa_mod=mo,
                residuo=mo - ob, k_ref=k, delta_CS=dcs,
                en_frontera="k_ref" if frontera else "",
            ))

    salida = RAIZ / "resultados/dcs_fija/calibracion_etapa1.csv"
    salida.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(salida, index=False)
    print(f"Archivo: {salida}")


if __name__ == "__main__":
    main()
