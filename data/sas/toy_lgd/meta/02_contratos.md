# CONTRATOS — Contract Catalogue

One row per contract. Contains metadata about each contract,
including fusion/absorption status.

## Fields

| Field | Type | Length | Description |
|-------|------|--------|-------------|
| ID_CONTRATO | Char | 20 | Primary key |
| SW_FUSION | Num | 8 | 1 = contract absorbed in fusion, 0 = normal |
| ID_FUSION_FINAL | Char | 20 | Master ID of the fusion group (if SW_FUSION=1) |
| ENTIDAD_ORIGEN | Char | 30 | Originating bank/entity |
| FECHA_ALTA | Char | 10 | Contract creation date (YYYY-MM-DD) |
| PRODUCTO | Char | 20 | Product type (PR, TC, CR, LP) |

## Fusion logic

When SW_FUSION=1:
- The original contract was absorbed by another entity
- LGD_ESTIMADA may be missing (not reported by absorbed entity)
- OR_EAD lookups in BASILEA_MENSUAL should use ID_FUSION_FINAL
  instead of ID_CONTRATO
- ID_FUSION_FINAL is NOT unique — multiple contracts can map
  to the same fusion group

## Example

| ID_CONTRATO | SW_FUSION | ID_FUSION_FINAL | ENTIDAD_ORIGEN |
|-------------|-----------|-----------------|-----------------|
| CONT_001 | 0 | . | BANCO_A |
| CONT_005 | 1 | FUS_7712 | BANCO_B (absorbed) |
| CONT_006 | 1 | FUS_7712 | BANCO_C (absorbed) |

CONT_005 and CONT_006 share ID_FUSION_FINAL=FUS_7712.
A JOIN on ID_FUSION_FINAL will produce 2 rows per contract.
