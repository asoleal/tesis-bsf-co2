# Datos

## Crudos

- `crudos/feeds.csv`: registros crudos de sensores.
- `crudos/Datos pesos trabajo Jonathan y Andres.xlsx`: libro original de crecimiento y datos experimentales asociados.

## Procesados heredados

- `procesados/crecimiento_propio.csv`: promedios y desviaciones estándar de crecimiento ya calculados.
- `procesados/datos_finales_PINN_corregidos.csv`: tasas históricas depuradas. Sus pendientes están en ppm por registro, no en ppm/min; se conservan para trazabilidad. Las tasas canónicas se recalcularán desde `feeds.csv`.

## Verificación

Desde la raíz del repositorio:

```bash
sha256sum -c datos/manifest_sha256.txt
