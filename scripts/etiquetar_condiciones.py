#!/usr/bin/env python3
"""Agrega columnas de condicion experimental a la tabla canonica de cierres.

No modifica ningun valor numerico reconstruido; solo anade:
  condicion, dieta, subtipo, replica, observacion, excluir_promedio.

Reglas de observacion:
  r2_bajo: R2 de la regresion original menor que 0.90.
  saturado: columna Saturado del CSV limpio en verdadero.
  duplicado_etiqueta: dos filas reconstruyen el mismo cierre fisico
    (mismo Gas, entry_ini y entry_fin). Decision experimental documentada:
    las filas con etiqueta o Muestra_ID de "generacion" duplican cierres
    reales de los experimentos; la fila canonica es la del experimento y
    la duplicada se excluye de promedios.
"""

import argparse
from pathlib import Path

import pandas as pd

MAPA = {
    "Control_Alimento_D1": ("control", "D1", ""),
    "Control_Alimento_D4": ("control", "D4", ""),
    "D1_Tratamiento": ("tratamiento", "D1", ""),
    "D4_Tratamiento": ("tratamiento", "D4", ""),
    "Larvas_Ayuno": ("larvas_solas", "ninguna", "ayuno"),
    "Larvas_Generacion_Base": ("larvas_solas", "ninguna", "generacion_base"),
}

ETIQUETA_DUPLICADA = "Larvas_Generacion_Base"


def agregar_observacion(df, mascara, marca):
    for i in df.index[mascara]:
        obs = df.at[i, "observacion"]
        df.at[i, "observacion"] = f"{obs};{marca}" if obs else marca


def marcar_duplicados(df):
    """Marca filas que reconstruyen el mismo cierre fisico que otra."""
    df["excluir_promedio"] = False
    es_generacion = (
        df["Muestra_ID"].astype(str).str.lower().str.startswith("generacion")
        | (df["Etiqueta"] == ETIQUETA_DUPLICADA)
    )
    clave = ["Gas", "entry_ini", "entry_fin"]
    for _, idx in df.groupby(clave).groups.items():
        if len(idx) < 2:
            continue
        sub = df.loc[idx]
        canonicas = sub[~es_generacion.loc[sub.index]]
        if len(canonicas) != 1:
            raise AssertionError(
                f"Cierre duplicado sin una unica fila canonica: "
                f"{sub[['Etiqueta','Muestra_ID']].to_dict('records')}"
            )
        cierre = canonicas["Etiqueta"].iloc[0]
        for i in sub.index.difference(canonicas.index):
            df.at[i, "excluir_promedio"] = True
            agregar_observacion(df, df.index == i, f"duplicado_etiqueta:cierre={cierre}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrada", default="datos/procesados/cierres_canonicos.csv")
    parser.add_argument("--salida", default="datos/procesados/cierres_etiquetados.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.entrada)
    faltantes = sorted(set(df["Etiqueta"]) - set(MAPA))
    if faltantes:
        raise AssertionError(f"Etiquetas sin mapeo: {faltantes}")

    mapeo = df["Etiqueta"].map(MAPA)
    df["condicion"] = [m[0] for m in mapeo]
    df["dieta"] = [m[1] for m in mapeo]
    df["subtipo"] = [m[2] for m in mapeo]

    df = df.sort_values(["Etiqueta", "Gas", "Dia_Experimento", "entry_ini"]).reset_index(drop=True)
    df["replica"] = (
        df.groupby(["Etiqueta", "Gas", "Dia_Experimento"]).cumcount() + 1
    )

    df["observacion"] = ""
    agregar_observacion(df, df["r2_registro"] < 0.90, "r2_bajo")
    agregar_observacion(df, df["Saturado"].astype(bool), "saturado")

    df = marcar_duplicados(df)

    destino = Path(args.salida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destino, index=False)

    print(f"Archivo: {destino}")
    print(f"Filas: {len(df)}; excluidas de promedios: {int(df['excluir_promedio'].sum())}")

    validas = df[~df["excluir_promedio"]]
    print("\nResumen por condicion/dieta/gas (sin excluidas):")
    print(
        validas.groupby(["condicion", "dieta", "subtipo", "Gas"])
        .agg(n=("Gas", "size"), replicas_extra=("replica", lambda s: int((s > 1).sum())))
        .to_string()
    )

    excluidas = df[df["excluir_promedio"]]
    if not excluidas.empty:
        print("\nFilas excluidas (etiqueta duplicada):")
        print(
            excluidas[["Etiqueta", "Muestra_ID", "Gas", "Dia_Experimento",
                       "entry_ini", "entry_fin", "observacion"]]
            .to_string(index=False)
        )

    n_r2 = int(df["observacion"].str.contains("r2_bajo").sum())
    n_sat = int(df["observacion"].str.contains("saturado").sum())
    print(f"\nFilas con r2_bajo: {n_r2}; con saturado: {n_sat}")


if __name__ == "__main__":
    main()
