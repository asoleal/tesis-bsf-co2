#!/usr/bin/env python3
"""Reconstruye los intervalos usados en las tasas limpias de gas.

La busqueda reproduce cada Tasa_Produccion y R2 mediante regresion de la
concentracion contra entry_id. Despues recalcula la pendiente contra las
marcas de tiempo reales y reporta la tasa en ppm/min.

Ejemplo:
  python3 scripts/reconstruir_cierres.py \
    --feeds datos/crudos/feeds.csv \
    --limpios datos/procesados/datos_finales_PINN_corregidos.csv \
    --salida datos/procesados/cierres_canonicos.csv \
    --mes 2025-09
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress


def cargar_crudos(ruta):
    df = pd.read_csv(ruta)
    df["dt_utc"] = pd.to_datetime(df["created_at"], utc=True)
    df["dt_local"] = df["dt_utc"].dt.tz_convert("America/Bogota")
    return df.sort_values("entry_id").reset_index(drop=True)


def construir_ventanas(g, campo, min_n=4):
    """Todas las ventanas contiguas; x = entry_id, como en el CSV limpio."""
    x = g["entry_id"].to_numpy(float)
    t = (g["dt_local"].astype("int64") / 1e9 / 60).to_numpy(float)
    t = t - t[0]
    y = g[campo].to_numpy(float)
    valido = np.isfinite(y)
    inicios = np.where(valido & np.r_[True, ~valido[:-1]])[0]
    fines = np.where(valido & np.r_[~valido[1:], True])[0]
    salida = []

    for a, z in zip(inicios, fines):
        xx = x[a:z + 1]
        tt = t[a:z + 1]
        yy = y[a:z + 1]
        n = len(xx)
        cx = np.r_[0, np.cumsum(xx)]
        cy = np.r_[0, np.cumsum(yy)]
        cxx = np.r_[0, np.cumsum(xx * xx)]
        cyy = np.r_[0, np.cumsum(yy * yy)]
        cxy = np.r_[0, np.cumsum(xx * yy)]

        for i in range(n - min_n + 1):
            js = np.arange(i + min_n - 1, n)
            nn = js - i + 1
            sx = cx[js + 1] - cx[i]
            sy = cy[js + 1] - cy[i]
            sxx = cxx[js + 1] - cxx[i]
            syy = cyy[js + 1] - cyy[i]
            sxy = cxy[js + 1] - cxy[i]
            denx = nn * sxx - sx * sx
            deny = nn * syy - sy * sy
            cov = nn * sxy - sx * sy
            ok = (denx > 1e-12) & (deny > 1e-12)
            if not ok.any():
                continue
            jj = js[ok]
            pendientes = cov[ok] / denx[ok]
            r2 = cov[ok] ** 2 / (denx[ok] * deny[ok])
            for j, cantidad, pendiente, r2j in zip(jj, nn[ok], pendientes, r2):
                salida.append((a + i, a + j, int(cantidad), tt[j] - tt[i], pendiente, r2j))

    return pd.DataFrame(
        salida,
        columns=["i", "j", "n_registros", "duracion_min", "pendiente_registro", "r2_registro"],
    )


def pendiente_tiempo(g, campo):
    t = (g["dt_local"].astype("int64") / 1e9 / 60).to_numpy(float)
    t = t - t[0]
    y = g[campo].to_numpy(float)
    ok = np.isfinite(y)
    ajuste = linregress(t[ok], y[ok])
    intervalo = np.median(np.diff(t)) * 60 if len(t) > 1 else np.nan
    return ajuste.slope, ajuste.rvalue ** 2, intervalo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feeds", default="datos/crudos/feeds.csv")
    parser.add_argument("--limpios", default="datos/procesados/datos_finales_PINN_corregidos.csv")
    parser.add_argument("--salida", default="datos/procesados/cierres_canonicos.csv")
    parser.add_argument("--mes", default="2025-09", help="Mes de los cierres, AAAA-MM")
    args = parser.parse_args()

    crudos = cargar_crudos(args.feeds)
    limpios = pd.read_csv(args.limpios)
    cache = {}
    filas = []

    for _, original in limpios.iterrows():
        dia = int(original["Dia_Experimento"])
        gas = original["Gas"]
        campo = "field2" if gas == "co2" else "field6"
        mascara = (
            (crudos["dt_local"].dt.strftime("%Y-%m") == args.mes)
            & (crudos["dt_local"].dt.day == dia)
        )
        g = crudos[mascara].reset_index(drop=True)
        if g.empty:
            raise ValueError(f"No hay datos crudos para {args.mes}-{dia:02d}")

        clave = (dia, gas)
        if clave not in cache:
            cache[clave] = (g, construir_ventanas(g, campo))
        g, ventanas = cache[clave]

        objetivo = float(original["Tasa_Produccion"])
        r2_objetivo = float(original["R2"])
        error = (
            np.abs(ventanas["pendiente_registro"] - objetivo) / max(abs(objetivo), 1e-12)
            + np.abs(ventanas["r2_registro"] - r2_objetivo)
        )
        k = int(error.idxmin())
        w = ventanas.loc[k]
        i, j = int(w["i"]), int(w["j"])
        tramo = g.iloc[i:j + 1]
        tasa_min, r2_tiempo, intervalo_s = pendiente_tiempo(tramo, campo)

        fila = original.to_dict()
        fila.update(
            campo_crudo=campo,
            entry_ini=int(tramo["entry_id"].iloc[0]),
            entry_fin=int(tramo["entry_id"].iloc[-1]),
            inicio_local=tramo["dt_local"].iloc[0].isoformat(),
            fin_local=tramo["dt_local"].iloc[-1].isoformat(),
            n_registros=int(w["n_registros"]),
            duracion_min=float(w["duracion_min"]),
            intervalo_mediano_s=intervalo_s,
            tasa_ppm_registro=float(w["pendiente_registro"]),
            r2_registro=float(w["r2_registro"]),
            tasa_ppm_min_tiempo=tasa_min,
            r2_tiempo=r2_tiempo,
            factor_tiempo_vs_registro=tasa_min / objetivo,
            error_reconstruccion=float(error.loc[k]),
            reconstruccion_exacta=bool(error.loc[k] < 1e-8),
        )
        filas.append(fila)

    resultado = pd.DataFrame(filas)
    destino = Path(args.salida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(destino, index=False)

    print(f"Reconstruidos {len(resultado)} registros; exactos: {resultado['reconstruccion_exacta'].sum()}")
    if int(resultado["reconstruccion_exacta"].sum()) != len(resultado):
        raise AssertionError("Hay registros no reconstruidos exactamente")
    print(f"Archivo: {destino}")
    ayuno = resultado[resultado["Etiqueta"] == "Larvas_Ayuno"]
    if not ayuno.empty:
        print("\nLarvas solas:")
        print(
            ayuno[
                [
                    "Gas", "entry_ini", "entry_fin", "inicio_local", "fin_local",
                    "n_registros", "tasa_ppm_registro", "tasa_ppm_min_tiempo",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
