#!/usr/bin/env python3
"""Construye la tabla canonica de crecimiento larvario.

Lee crecimiento_propio.csv (promedios de 3 replicas, mg PH/larva) y agrega:
  fecha: dia calendario segun la hipotesis de montaje 2025-09-05.
  es_dia_gas: el dia larvario tiene cierres de gas asociados.
  coincide_pesaje: el dia de gas tiene pesaje directo (no interpolado).

No convierte a materia seca ni carbono; eso se define en la etapa de
comparacion con el modelo.
"""

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

MONTAJE = date(2025, 9, 3)  # hipotesis documentada en datos/README.md
DIAS_GAS = [6, 8, 10, 14]   # cierres del 9, 11, 13 y 17 sep; montaje 3 sep
DIAS_GAS_CON_PESAJE = [6, 8, 10, 14]  # sincronizacion 4 de 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrada", default="datos/procesados/crecimiento_propio.csv")
    parser.add_argument("--salida", default="datos/procesados/crecimiento_canonico.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.entrada, comment="#")
    esperadas = {"dieta", "dia", "ph_mg", "de_mg"}
    if set(df.columns) != esperadas:
        raise AssertionError(f"Columnas inesperadas: {list(df.columns)}")
    if int(df["dia"].min()) != 0:
        raise AssertionError("El dia larvario minimo debe ser 0")

    df = df.rename(columns={"dia": "dia_larvario"}).sort_values(["dieta", "dia_larvario"])
    df["fecha"] = df["dia_larvario"].map(lambda d: (MONTAJE + timedelta(days=int(d))).isoformat())
    df["es_dia_gas"] = df["dia_larvario"].isin(DIAS_GAS)
    df["coincide_pesaje"] = df["dia_larvario"].isin(DIAS_GAS_CON_PESAJE)

    destino = Path(args.salida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destino, index=False)

    print(f"Archivo: {destino}")
    print(f"Filas: {len(df)}")
    print("\nDias de gas y disponibilidad de pesaje:")
    for d in DIAS_GAS:
        filas = df[df["dia_larvario"] == d]
        estado = "pesaje directo" if not filas.empty else "sin pesaje (interpolar)"
        print(f"  dia larvario {d:2d} ({(MONTAJE + timedelta(days=d)).isoformat()}): {estado}")
    print("\nTabla:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
