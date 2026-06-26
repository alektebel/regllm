---
id: "ops_user_access_sas_dataset"
type: "insight"
priority: 0.4
tags: [operational, access, permissions, SAS, dataset]
fields: []
articles: []
source: "Incidente SEG-2025-009 — permisos tabla mylib.ciclos_recuperacion"
feedback: false
---

# Permisos incorrectos en mylib.ciclos_recuperacion: usuarios con acceso de escritura

## Incidente
La revisión de seguridad trimestral reveló que 12 usuarios del equipo de negocio tenían permisos de escritura (`ALTER`) sobre `mylib.ciclos_recuperacion`. Esta tabla es la entrada principal del pipeline LGD. 3 de esos usuarios habían modificado valores de `COLATERAL_TIPO` manualmente para "corregir" errores que veían en informes.

## Impacto
- 47 ciclos con COLATERAL_TIPO modificado manualmente
- 2 ciclos con EAD modificada (±5% respecto al original)
- Las modificaciones no eran trazables (no hay auditoría a nivel de fila en SAS)

## Fix
1. Revocar permisos de escritura a usuarios de negocio
2. Implementar tabla de correcciones aprobadas (`mylib.correcciones_aprobadas`) con trazabilidad
3. El pipeline debe aplicar correcciones + loguearlas, no permitir modificaciones directas:
   ```sas
   /* Aplicar correcciones aprobadas */
   CREATE TABLE ciclos_corregidos AS
   SELECT a.*, COALESCE(b.COLATERAL_TIPO, a.COLATERAL_TIPO) AS COLATERAL_TIPO_CORREGIDO
   FROM mylib.ciclos_recuperacion a
   LEFT JOIN mylib.correcciones_aprobadas b ON a.ID_CONTRATO = b.ID_CONTRATO;
   ```

## Lección
En entornos SAS, los permisos de dataset son todo/nada. No hay row-level security. Proteger las tablas fuente del pipeline es crítico.
