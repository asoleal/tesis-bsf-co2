#!/usr/bin/env python3
"""Validacion cruzada interna agrupada por dieta y dia."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import calibrar_etapa1 as e1
import calibrar_etapa2 as e2
from modelo import (
    ParametrosEriksen,
    ParametrosLecho,
    carbono_a_ppm_min,
    estado_inicial_desde_dw,
    tasas_larva,
    tasas_lecho,
)

MODOS = ("E0", "R", "A", "C")
T_EVAL = np.arange(0.0, e2.T_FIN + 0.1, 0.1)
PH0 = {}


def simular(dieta, amax, rho, f_d4, micro, modo, t_eval=T_EVAL):
    """E0/R: sin lecho; A: lecho aditivo; C: acople alimento-lecho."""
    p_larva = ParametrosEriksen(amax=amax, rho=rho, tp_min=e2.TP_MIN)
    y0 = estado_inicial_desde_dw(
        PH0[dieta] * e2.F_MS, 0.0, e2.X0_G_C, p_larva
    )

    if modo in ("E0", "R"):
        def fun(tt, yy):
            r = tasas_larva(
                tt, max(yy[0], 0.0), max(yy[1], 0.0), p_larva, 1.0
            )
            return [r.dB, r.dLtot]

        sol = solve_ivp(
            fun, (0.0, e2.T_FIN), y0[:2], method="LSODA",
            t_eval=t_eval, rtol=1e-7, atol=1e-10,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        t = sol.t
        Y = sol.y
        ppm = np.array([
            carbono_a_ppm_min(
                e2.N_LARVAS * tasas_larva(
                    tt, max(yy[0], 0.0), max(yy[1], 0.0),
                    p_larva, 1.0
                ).rCO2_C / 1000.0,
                e2.V_AIRE_L, e2.T_K,
            )
            for tt, yy in zip(t, Y.T)
        ])
    else:
        k_ref, dcs = micro[dieta]
        p_lecho = ParametrosLecho(
            f_dieta=(1.0 if dieta == "D1" else f_d4),
            K_D=e2.K_D, k_ref=k_ref, K_SM=e2.K_SM,
            Y_X=e2.Y_X, Y_Cmic=e2.Y_CMIC,
        )

        def fun(tt, yy):
            B, Ltot, D_C, X = [max(float(v), 0.0) for v in yy[:4]]
            mic = tasas_lecho(D_C, X, e2.T_C, p_lecho)
            factor = p_lecho.f_dieta * mic.phi_D if modo == "C" else 1.0
            lar = tasas_larva(tt, B, Ltot, p_larva, factor)
            return [
                lar.dB,
                lar.dLtot,
                -(e2.N_LARVAS * lar.rA / 1000.0) - mic.rdeg,
                mic.dX,
                e2.N_LARVAS * lar.rCO2_C / 1000.0 + mic.rCmic,
                (1.0 - p_lecho.Y_X - p_lecho.Y_Cmic) * mic.rdeg,
            ]

        y = y0.copy()
        cortes = sorted(set(
            [0.0] + [d for d, _ in e2.PULSOS] + [e2.T_FIN]
        ))
        pulsos = dict(e2.PULSOS)
        ts, ys = [], []

        for a, b in zip(cortes[:-1], cortes[1:]):
            if a in pulsos:
                y[2] += (
                    pulsos[a] * e2.MS_DIETA[dieta]
                    * e2.DELTA_C_ALIM * dcs
                )
            t_seg = t_eval[(t_eval >= a) & (t_eval <= b)]
            if len(t_seg) < 2:
                t_seg = np.linspace(a, b, 3)
            sol = solve_ivp(
                fun, (a, b), y, method="LSODA", t_eval=t_seg,
                rtol=1e-6, atol=1e-9,
            )
            if not sol.success:
                raise RuntimeError(sol.message)
            ts.append(sol.t)
            ys.append(sol.y)
            y = sol.y[:, -1]

        t = np.concatenate(ts)
        Y = np.concatenate(ys, axis=1)
        ppm = np.empty_like(t)

        for k, (tt, yy) in enumerate(zip(t, Y.T)):
            B, Ltot, D_C, X = [max(float(v), 0.0) for v in yy[:4]]
            mic = tasas_lecho(D_C, X, e2.T_C, p_lecho)
            factor = p_lecho.f_dieta * mic.phi_D if modo == "C" else 1.0
            lar = tasas_larva(tt, B, Ltot, p_larva, factor)
            ppm[k] = carbono_a_ppm_min(
                e2.N_LARVAS * lar.rCO2_C / 1000.0 + mic.rCmic,
                e2.V_AIRE_L, e2.T_K,
            )

    B = np.maximum(Y[0], 0.0)
    Ltot = np.maximum(Y[1], 0.0)
    ph = (
        (1.0 - p_larva.delta_LB) * B
        / (p_larva.delta_CB * p_larva.delta_O)
        + Ltot / p_larva.delta_CL
    ) / e2.F_MS
    return t, ppm, ph


def ajustar_micro(ctrl):
    salida = {}
    for dieta in ("D1", "D4"):
        obs = ctrl[ctrl["dieta"] == dieta]
        if len(obs) < 2:
            raise AssertionError(
                f"Menos de dos controles de entrenamiento en {dieta}"
            )
        k_ref, dcs, _, frontera = e1.ajustar_dieta(dieta, obs)
        salida[dieta] = (k_ref, dcs, frontera)
    return salida


def ajustar_modo(modo, crec, gas, micro):
    if modo == "E0":
        return 1.0, 1.0, 1.0, []

    obs_crec = [
        (
            r.dieta, float(r.dia_larvario), float(r.ph_mg),
            max(float(r.de_mg), 0.1 * float(r.ph_mg))
        )
        for r in crec.itertuples(index=False)
        if r.dia_larvario <= e2.DIA_AJUSTE_MAX
    ]
    obs_gas = [
        (
            r.dieta, float(r.dia_larvario),
            float(r.tasa_ppm_min_tiempo), bool(r.Saturado)
        )
        for r in gas.itertuples(index=False)
    ]

    npar = 3 if modo == "C" else 2
    lo, hi = e2.LO[:npar], e2.HI[:npar]

    def residuos(theta):
        amax, rho = 10.0 ** theta[0], 10.0 ** theta[1]
        f_d4 = theta[2] if modo == "C" else 1.0
        curvas = {
            d: simular(d, amax, rho, f_d4, micro, modo)
            for d in ("D1", "D4")
        }
        res = []

        for d, dia, obs, w in obs_crec:
            t, _, ph = curvas[d]
            res.append((np.interp(dia, t, ph) - obs) / w)

        for d, dia, obs, sat in obs_gas:
            t, ppm, _ = curvas[d]
            r = np.interp(dia, t, ppm) - obs
            if sat:
                r = min(r, 0.0)
            res.append(r / e2.PESO_GAS)

        return np.asarray(res)

    mejor = None
    for arranque in e2.ARRANQUES:
        x0 = np.clip(
            np.asarray(arranque[:npar], float),
            lo + 1e-9, hi - 1e-9
        )
        sol = least_squares(
            residuos, x0, bounds=(lo, hi), method="trf"
        )
        if mejor is None or sol.cost < mejor.cost:
            mejor = sol

    amax, rho = 10.0 ** mejor.x[0], 10.0 ** mejor.x[1]
    f_d4 = mejor.x[2] if modo == "C" else 1.0
    frontera = []
    nombres = ("a_max", "rho", "f_D4")[:npar]

    for nombre, val, vmin, vmax in zip(nombres, mejor.x, lo, hi):
        if abs(val - vmin) < 1e-3 or abs(val - vmax) < 1e-3:
            frontera.append(nombre)

    return amax, rho, f_d4, frontera


def cargar_datos():
    global PH0

    c = pd.read_csv("datos/procesados/cierres_etiquetados.csv")
    c = c[
        (~c["excluir_promedio"]) & (c["Gas"] == "co2")
    ].copy()
    c["dia_larvario"] = (
        c["Dia_Experimento"] - e2.DIA_MONTAJE_SEP
    )

    ctrl = c[c["condicion"] == "control"].copy()
    ctrl = ctrl.rename(
        columns={"tasa_ppm_min_tiempo": "tasa_obs"}
    )
    gas = c[c["condicion"] == "tratamiento"].copy()
    crec = pd.read_csv("datos/procesados/crecimiento_canonico.csv")

    PH0 = {
        d: float(g.loc[g["dia_larvario"] == 0, "ph_mg"].iloc[0])
        for d, g in crec.groupby("dieta")
    }

    assert len(gas) == 8 and len(ctrl) == 7
    return crec, ctrl, gas


def verificar_canonico(crec, ctrl, gas):
    micro_archivo_df = pd.read_csv("resultados/calibracion_etapa1.csv")
    micro = {
        d: (float(g["k_ref"].iloc[0]), float(g["delta_CS"].iloc[0]))
        for d, g in micro_archivo_df.groupby("dieta")
    }

    micro_reajustado = ajustar_micro(ctrl)
    for d in ("D1", "D4"):
        assert np.allclose(micro[d], micro_reajustado[d][:2])

    p = pd.read_csv(
        "resultados/parametros_etapa2.csv"
    ).set_index("parametro")
    amax = float(p.loc["a_max", "valor"])
    rho = float(p.loc["rho", "valor"])
    f_d4 = float(p.loc["f_D4", "valor"])

    e2.PH0 = PH0
    for d in ("D1", "D4"):
        t0, ppm0, ph0, _ = e2.simular_dieta(
            d, amax, rho, f_d4, micro, T_EVAL
        )
        t1, ppm1, ph1 = simular(d, amax, rho, f_d4, micro, "C")
        assert np.allclose(t0, t1)
        assert np.allclose(ppm0, ppm1, rtol=1e-8, atol=1e-6)
        assert np.allclose(ph0, ph1, rtol=1e-8, atol=1e-6)

    crudos, censurados = [], []
    for d in ("D1", "D4"):
        t, ppm, _ = simular(d, amax, rho, f_d4, micro, "C")
        for r in gas[gas["dieta"] == d].itertuples(index=False):
            residuo = float(
                np.interp(r.dia_larvario, t, ppm)
                - r.tasa_ppm_min_tiempo
            )
            crudos.append(residuo)
            censurados.append(
                min(residuo, 0.0) if r.Saturado else residuo
            )

    rmse_crudo = float(np.sqrt(np.mean(np.asarray(crudos) ** 2)))
    rmse_censurado = float(
        np.sqrt(np.mean(np.asarray(censurados) ** 2))
    )

    print(
        f"Canonicos: a_max={amax:.6f}, rho={rho:.6f}, "
        f"f_D4={f_d4:.6f}"
    )
    print(f"RMSE crudo C={rmse_crudo:.4f} ppm/min")
    print(f"RMSE censurado C={rmse_censurado:.4f} ppm/min")

    assert abs(rmse_crudo - 98.2363) < 0.05, rmse_crudo
    assert abs(rmse_censurado - 85.1604) < 0.05, rmse_censurado
    print("VERIFICACION CANONICA OK")


def ejecutar(crec, ctrl, gas):
    pliegues = gas[["dieta", "dia_larvario"]].drop_duplicates()
    pliegues = pliegues.sort_values(["dieta", "dia_larvario"])
    assert len(pliegues) == 7

    predicciones, parametros = [], []

    for p in pliegues.itertuples(index=False):
        d, dia = p.dieta, float(p.dia_larvario)
        mascara_gas = (
            (gas["dieta"] == d) & (gas["dia_larvario"] == dia)
        )
        mascara_ctrl = (
            (ctrl["dieta"] == d) & (ctrl["dia_larvario"] == dia)
        )
        mascara_crec = (
            (crec["dieta"] == d) & (crec["dia_larvario"] == dia)
        )

        gas_prueba = gas[mascara_gas].copy()
        gas_ent = gas[~mascara_gas].copy()
        ctrl_ent = ctrl[~mascara_ctrl].copy()
        crec_ent = crec[~mascara_crec].copy()

        print(
            f"\nPliegue {d}, dia {dia:g}; "
            f"prueba n={len(gas_prueba)}",
            flush=True
        )

        micro_tuplas = ajustar_micro(ctrl_ent)
        micro = {dieta: v[:2] for dieta, v in micro_tuplas.items()}

        for dieta, (k_ref, dcs, frontera) in micro_tuplas.items():
            parametros.append(dict(
                dieta_pliegue=d, dia_pliegue=dia, modo="micro",
                parametro=f"k_ref_{dieta}", valor=k_ref,
                frontera="k_ref" in frontera,
            ))
            parametros.append(dict(
                dieta_pliegue=d, dia_pliegue=dia, modo="micro",
                parametro=f"delta_CS_{dieta}", valor=dcs,
                frontera="delta_CS" in frontera,
            ))

        for modo in MODOS:
            amax, rho, f_d4, frontera = ajustar_modo(
                modo, crec_ent, gas_ent, micro
            )
            print(
                f"  {modo}: a_max={amax:.4f}, rho={rho:.4f}, "
                f"f_D4={f_d4:.4f}, frontera={frontera}",
                flush=True
            )

            for nombre, valor in (
                ("a_max", amax), ("rho", rho), ("f_D4", f_d4)
            ):
                parametros.append(dict(
                    dieta_pliegue=d, dia_pliegue=dia, modo=modo,
                    parametro=nombre, valor=valor,
                    frontera=nombre in frontera,
                ))

            t, ppm, _ = simular(d, amax, rho, f_d4, micro, modo)

            for r in gas_prueba.itertuples(index=False):
                mod = float(np.interp(r.dia_larvario, t, ppm))
                obs = float(r.tasa_ppm_min_tiempo)
                residuo = mod - obs
                predicciones.append(dict(
                    dieta=d, dia_larvario=dia,
                    Muestra_ID=r.Muestra_ID, modo=modo,
                    obs_ppm_min=obs, mod_ppm_min=mod,
                    obs_ppm_dia=1440.0 * obs,
                    mod_ppm_dia=1440.0 * mod,
                    saturado=bool(r.Saturado),
                    residuo=residuo,
                    residuo_censurado=(
                        min(residuo, 0.0) if r.Saturado else residuo
                    ),
                ))

    pred = pd.DataFrame(predicciones)
    par = pd.DataFrame(parametros)
    metricas = []

    grupos = list(pred.groupby(["modo", "dieta"]))
    grupos += list(
        pred.assign(dieta="conjunto").groupby(["modo", "dieta"])
    )

    for (modo, dieta), g in grupos:
        ns = g[~g["saturado"]]
        sat = g[g["saturado"]]
        rmse_exactos = np.sqrt(np.mean(ns["residuo"] ** 2))
        rmse_censurado = np.sqrt(
            np.mean(g["residuo_censurado"] ** 2)
        )
        metricas.append(dict(
            modo=modo, dieta=dieta, n=len(g),
            n_exactos=len(ns),
            rmse_exactos_ppm_min=rmse_exactos,
            rmse_exactos_ppm_dia=1440.0 * rmse_exactos,
            sesgo_exactos_ppm_min=np.mean(ns["residuo"]),
            rmse_censurado_ppm_min=rmse_censurado,
            rmse_censurado_ppm_dia=1440.0 * rmse_censurado,
            n_saturados=len(sat),
            cobertura_saturados=np.mean(
                sat["mod_ppm_min"] >= sat["obs_ppm_min"]
            ),
        ))

    met = pd.DataFrame(metricas)
    destino = Path("resultados/validacion_co2")
    destino.mkdir(parents=True, exist_ok=True)

    pred.to_csv(destino / "predicciones.csv", index=False)
    par.to_csv(destino / "parametros_pliegues.csv", index=False)
    met.to_csv(destino / "metricas.csv", index=False)

    print("\n=== Metricas fuera de calibracion ===")
    print(met.round(4).to_string(index=False))
    print(f"\nResultados: {destino}")
    print("VALIDACION CRUZADA OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solo-verificacion", action="store_true")
    args = parser.parse_args()

    crec, ctrl, gas = cargar_datos()
    verificar_canonico(crec, ctrl, gas)

    if not args.solo_verificacion:
        ejecutar(crec, ctrl, gas)


if __name__ == "__main__":
    main()
