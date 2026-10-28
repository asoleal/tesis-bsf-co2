#!/usr/bin/env python3
"""Etapa 3 (opcional): modulo de prepupa contra fase tardia y ayuno.

Calibrados (2 grados de libertad, multinicio):
  f_mant_prepupa en [0.05, 1.0]   (factor del mantenimiento en instar 7)
  k_cat_prepupa  en [0.0, 0.3] d^-1 (catabolismo estructural en instar 7)

Congelados: todo lo de las etapas 1 y 2 (se leen calibracion_etapa1.csv
y parametros_etapa2.csv). Los datos usados son exactamente los que la
etapa 2 excluyo del ajuste (opcion A):
  - crecimiento dias 10 y 14 (4 puntos, peso = max(DE, 10 %)),
  - gas tardio no saturado (3 cierres, peso = 25 ppm/min),
  - ayuno del dia 14 (1 punto, peso = 25 ppm/min).
El nucleo original es el caso f_mant=1, k_cat=0; se reporta la mejora
y se verifica que la ventana de ajuste de la etapa 2 (dias <= 8) no
cambia con el modulo activo.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from modelo import (
    ParametrosEriksen,
    ParametrosLecho,
    carbono_a_ppm_min,
    estado_inicial_desde_dw,
    rhs,
    tasas_larva,
    tasas_lecho,
)
from calibrar_etapa2 import (
    PULSOS, MS_DIETA, N_LARVAS, T_K, T_C, V_AIRE_L, DELTA_C_ALIM,
    F_MS, T_FIN, DIA_MONTAJE_SEP, TP_MIN, DIA_AJUSTE_MAX,
    X0_G_C, Y_X, Y_CMIC, K_SM, K_D, PESO_GAS,
    cargar_microbiano, AYUNO_OBS, AYUNO_DIA,
)

LO = np.array([0.05, 0.0])
HI = np.array([1.0, 0.3])
ARRANQUES = [(0.3, 0.05), (0.15, 0.1)]


def simular_dieta_prep(dieta, amax, rho, f_d4, f_mant, k_cat, micro, ph0, t_eval):
    """simular_dieta de la etapa 2 con los parametros de prepupa."""
    k_ref, dcs = micro[dieta]
    p_larva = ParametrosEriksen(amax=amax, rho=rho, tp_min=TP_MIN,
                                f_mant_prepupa=f_mant, k_cat_prepupa=k_cat)
    p_lecho = ParametrosLecho(
        f_dieta=(1.0 if dieta == "D1" else f_d4),
        K_D=K_D, k_ref=k_ref, K_SM=K_SM, Y_X=Y_X, Y_Cmic=Y_CMIC,
    )
    y = estado_inicial_desde_dw(ph0 * F_MS, 0.0, X0_G_C, p_larva)
    cortes = sorted(set([0.0] + [d for d, _ in PULSOS] + [T_FIN]))
    pulsos = dict(PULSOS)
    ts, ys = [], []
    for a, b in zip(cortes[:-1], cortes[1:]):
        if a in pulsos:
            y[2] += pulsos[a] * MS_DIETA[dieta] * DELTA_C_ALIM * dcs
        t_seg = t_eval[(t_eval >= a) & (t_eval <= b)]
        if len(t_seg) < 2:
            t_seg = np.linspace(a, b, 3)
        sol = solve_ivp(
            rhs, (a, b), y, args=(p_larva, p_lecho, N_LARVAS, T_C),
            method="LSODA", t_eval=t_seg, rtol=1e-6, atol=1e-9,
        )
        if not sol.success:
            raise RuntimeError(f"fallo la integracion en [{a}, {b}]: {sol.message}")
        ts.append(sol.t)
        ys.append(sol.y)
        y = sol.y[:, -1]
    t = np.concatenate(ts)
    Y = np.concatenate(ys, axis=1)
    R_C = np.empty_like(t)
    for k, (tk, yk) in enumerate(zip(t, Y.T)):
        B, Ltot, D_C, X, _, _ = [max(float(v), 0.0) for v in yk]
        mic = tasas_lecho(D_C, X, T_C, p_lecho)
        lar = tasas_larva(tk, B, Ltot, p_larva, p_lecho.f_dieta * mic.phi_D)
        R_C[k] = N_LARVAS * lar.rCO2_C / 1000.0 + mic.rCmic
    ppm = carbono_a_ppm_min(R_C, V_AIRE_L, T_K)
    B = np.maximum(Y[0], 0.0)
    Ltot = np.maximum(Y[1], 0.0)
    ph = ((1.0 - p_larva.delta_LB) * B / (p_larva.delta_CB * p_larva.delta_O)
          + Ltot / p_larva.delta_CL) / F_MS
    return t, ppm, ph, Y, p_larva


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cierres", default="datos/procesados/cierres_etiquetados.csv")
    parser.add_argument("--crecimiento", default="datos/procesados/crecimiento_canonico.csv")
    parser.add_argument("--etapa1", default="resultados/calibracion_etapa1.csv")
    parser.add_argument("--etapa2", default="resultados/parametros_etapa2.csv")
    parser.add_argument("--salida", default="resultados/calibracion_prepupa.csv")
    parser.add_argument("--figura", default="figuras/calibracion_prepupa.png")
    args = parser.parse_args()

    micro = cargar_microbiano(args.etapa1)
    par2 = pd.read_csv(args.etapa2)
    par2 = dict(zip(par2["parametro"], par2["valor"]))
    for k in ("a_max", "rho", "f_D4"):
        assert k in par2, f"parametros_etapa2.csv sin {k}"
    amax, rho, f_d4 = float(par2["a_max"]), float(par2["rho"]), float(par2["f_D4"])
    print(f"Congelados de etapas 1 y 2: a_max={amax:.4f}, rho={rho:.3f}, f_D4={f_d4:.4f}")
    print(f"Microbiano: {micro}")

    crec = pd.read_csv(args.crecimiento)
    cierres = pd.read_csv(args.cierres)
    gas = cierres[
        (~cierres["excluir_promedio"])
        & (cierres["Gas"] == "co2")
        & (cierres["condicion"] == "tratamiento")
    ].copy()
    gas["dia_larvario"] = gas["Dia_Experimento"] - DIA_MONTAJE_SEP
    ph0 = {
        d: float(g.loc[g["dia_larvario"] == 0, "ph_mg"].iloc[0])
        for d, g in crec.groupby("dieta")
    }

    t_eval = np.arange(0.0, T_FIN + 0.1, 0.1)
    obs_crec_tarde = [
        (d, float(r["dia_larvario"]), float(r["ph_mg"]), max(float(r["de_mg"]), 0.1 * float(r["ph_mg"])))
        for d, g in crec.groupby("dieta") for _, r in g.iterrows()
        if float(r["dia_larvario"]) > DIA_AJUSTE_MAX
    ]
    obs_gas_tarde = [
        (d, float(r["dia_larvario"]), float(r["tasa_ppm_min_tiempo"]))
        for d, g in gas.groupby("dieta") for _, r in g.iterrows()
        if float(r["dia_larvario"]) > DIA_AJUSTE_MAX and not bool(r["Saturado"])
    ]
    print(f"Puntos de ajuste: {len(obs_crec_tarde)} crecimiento tardio, "
          f"{len(obs_gas_tarde)} gas tardio, 1 ayuno")

    def curvas(f_mant, k_cat):
        return {
            d: simular_dieta_prep(d, amax, rho, f_d4, f_mant, k_cat, micro, ph0[d], t_eval)
            for d in ("D1", "D4")
        }

    def ayuno_ppm(cur):
        t, ppm, ph, Y, p_larva = cur["D1"]
        i = int(np.argmin(np.abs(t - AYUNO_DIA)))
        B14 = max(float(Y[0, i]), 0.0)
        L14 = max(float(Y[1, i]), 0.0)
        lar = tasas_larva(AYUNO_DIA, B14, L14, p_larva, factor_alimento=0.0)
        return float(carbono_a_ppm_min(N_LARVAS * lar.rCO2_C / 1000.0, V_AIRE_L, T_K))

    def residuos(theta):
        f_mant, k_cat = float(theta[0]), float(theta[1])
        cur = curvas(f_mant, k_cat)
        res = []
        for dieta, dia, obs, w in obs_crec_tarde:
            t, _, ph, _, _ = cur[dieta]
            res.append((np.interp(dia, t, ph) - obs) / w)
        for dieta, dia, obs in obs_gas_tarde:
            t, ppm, _, _, _ = cur[dieta]
            res.append((np.interp(dia, t, ppm) - obs) / PESO_GAS)
        res.append((ayuno_ppm(cur) - AYUNO_OBS) / PESO_GAS)
        return np.array(res)

    # Chequeo interno: modulo apagado reproduce el ayuno canonico
    ay0 = ayuno_ppm(curvas(1.0, 0.0))
    print(f"Chequeo modulo apagado: ayuno={ay0:.2f} ppm/min (canonico 186.35)")

    mejor = None
    for arranque in ARRANQUES:
        x0 = np.clip(np.array(arranque, dtype=float), LO + 1e-9, HI - 1e-9)
        sol = least_squares(residuos, x0, bounds=(LO, HI), method="trf", max_nfev=60)
        print(f"  arranque {arranque}: costo={sol.cost:.4f}, nfev={sol.nfev}", flush=True)
        if mejor is None or sol.cost < mejor.cost:
            mejor = sol

    f_mant, k_cat = float(mejor.x[0]), float(mejor.x[1])
    frontera = [n for n, v, lo, hi in (("f_mant", mejor.x[0], LO[0], HI[0]),
                                       ("k_cat", mejor.x[1], LO[1], HI[1]))
                if abs(v - lo) < 1e-3 or abs(v - hi) < 1e-3]
    marca = f" [frontera: {','.join(frontera)}]" if frontera else ""
    print(f"\nCalibrados: f_mant_prepupa={f_mant:.4f}, k_cat_prepupa={k_cat:.4f} d^-1{marca}")

    cur0 = curvas(1.0, 0.0)
    cur1 = curvas(f_mant, k_cat)
    filas = []
    for dieta, dia, obs, w in obs_crec_tarde:
        m0 = float(np.interp(dia, cur0[dieta][0], cur0[dieta][2]))
        m1 = float(np.interp(dia, cur1[dieta][0], cur1[dieta][2]))
        filas.append(dict(bloque="crecimiento", dieta=dieta, dia_larvario=dia,
                          obs=obs, mod_original=m0, mod_prepupa=m1))
    for dieta, dia, obs in obs_gas_tarde:
        m0 = float(np.interp(dia, cur0[dieta][0], cur0[dieta][1]))
        m1 = float(np.interp(dia, cur1[dieta][0], cur1[dieta][1]))
        filas.append(dict(bloque="gas", dieta=dieta, dia_larvario=dia,
                          obs=obs, mod_original=m0, mod_prepupa=m1))
    filas.append(dict(bloque="ayuno", dieta="ninguna", dia_larvario=AYUNO_DIA,
                      obs=AYUNO_OBS, mod_original=ayuno_ppm(cur0),
                      mod_prepupa=ayuno_ppm(cur1)))
    comp = pd.DataFrame(filas)
    comp["residuo_original"] = comp["mod_original"] - comp["obs"]
    comp["residuo_prepupa"] = comp["mod_prepupa"] - comp["obs"]
    destino = Path(args.salida)
    comp.to_csv(destino, index=False)
    print(f"Archivo: {destino}")
    print(comp.round(2).to_string(index=False))

    rmse0 = float(np.sqrt(np.mean(comp["residuo_original"] ** 2)))
    rmse1 = float(np.sqrt(np.mean(comp["residuo_prepupa"] ** 2)))
    print(f"\nRMSE fase tardia + ayuno: original={rmse0:.2f}, prepupa={rmse1:.2f}")

    # Verificacion: la ventana de ajuste de la etapa 2 no debe cambiar
    cambio = 0.0
    for d, g in crec.groupby("dieta"):
        for _, r in g.iterrows():
            dia = float(r["dia_larvario"])
            if dia <= DIA_AJUSTE_MAX:
                m0 = float(np.interp(dia, cur0[d][0], cur0[d][2]))
                m1 = float(np.interp(dia, cur1[d][0], cur1[d][2]))
                cambio = max(cambio, abs(m0 - m1))
    print(f"Max cambio en ventana <= {DIA_AJUSTE_MAX:.0f} d con el modulo activo: {cambio:.4f} mg PH")

    pd.DataFrame([
        dict(parametro="f_mant_prepupa", valor=f_mant, frontera=("f_mant" in frontera)),
        dict(parametro="k_cat_prepupa", valor=k_cat, frontera=("k_cat" in frontera)),
    ]).to_csv(Path(args.salida).with_name("parametros_prepupa.csv"), index=False)
    print(f"Parametros: {Path(args.salida).with_name('parametros_prepupa.csv')}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colores = {"D1": "tab:blue", "D4": "tab:orange"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for j, dieta in enumerate(("D1", "D4")):
        t0, ppm0, ph0m0, _, _ = cur0[dieta]
        t1, ppm1, ph1m1, _, _ = cur1[dieta]
        gc = crec[crec["dieta"] == dieta]
        axes[0][j].plot(t0, ph0m0, "--", color=colores[dieta], label="original")
        axes[0][j].plot(t1, ph1m1, "-", color=colores[dieta], label="prepupa")
        axes[0][j].errorbar(gc["dia_larvario"], gc["ph_mg"], yerr=gc["de_mg"],
                            fmt="o", color=colores[dieta], capsize=3)
        axes[0][j].axvspan(DIA_AJUSTE_MAX, T_FIN, color="gray", alpha=0.15)
        axes[0][j].set_title(f"Crecimiento {dieta}")
        axes[0][j].legend(fontsize=8)
        gg = gas[gas["dieta"] == dieta]
        axes[1][j].plot(t0, ppm0, "--", color=colores[dieta])
        axes[1][j].plot(t1, ppm1, "-", color=colores[dieta])
        axes[1][j].plot(gg["dia_larvario"], gg["tasa_ppm_min_tiempo"], "o",
                        color=colores[dieta])
        axes[1][j].axvspan(DIA_AJUSTE_MAX, T_FIN, color="gray", alpha=0.15)
        axes[1][j].set_title(f"Gas tratamiento {dieta}")
        axes[1][j].set_xlabel("dia larvario")
    axes[1][0].plot([AYUNO_DIA], [AYUNO_OBS], "k*", markersize=12, label="ayuno obs")
    axes[1][0].legend(fontsize=8)
    axes[0][0].set_ylabel("PH (mg/larva)")
    axes[1][0].set_ylabel("CO2 (ppm/min)")
    fig.suptitle(f"Etapa 3 prepupa: f_mant={f_mant:.3f}, k_cat={k_cat:.3f} "
                 f"(linea = modulo, raya = original)", fontsize=9)
    fig.tight_layout()
    ruta_fig = Path(args.figura)
    fig.savefig(ruta_fig, dpi=200)
    print(f"Figura: {ruta_fig}")


if __name__ == "__main__":
    main()
