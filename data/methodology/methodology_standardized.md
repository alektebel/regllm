# Metodología Estándar para Riesgo de Crédito

## Descripción General

El método estándar (Standardised Approach - SA) es la metodología básica para el cálculo de requisitos de capital por riesgo de crédito, establecida en el Reglamento (UE) 575/2013 (CRR). Utiliza ponderaciones de riesgo fijas determinadas por el regulador.

## Principio Fundamental

```
RWA = Exposición × Ponderación de Riesgo
```

```
Capital Requerido = RWA × 8%
```

## Categorías de Exposición y Ponderaciones

### Exposiciones a Administraciones Centrales y Bancos Centrales

| Rating (ECAI) | Ponderación |
|---------------|-------------|
| AAA a AA- | 0% |
| A+ a A- | 20% |
| BBB+ a BBB- | 50% |
| BB+ a BB- | 100% |
| B+ a B- | 100% |
| Por debajo de B- | 150% |
| Sin calificación | 100% |

### Exposiciones a Entidades de Crédito

**Opción 1**: Basada en rating del soberano

| Rating del Soberano | Ponderación |
|--------------------|-------------|
| AAA a AA- | 20% |
| A+ a A- | 50% |
| BBB+ a BBB- | 100% |
| BB+ a BB- | 100% |
| Por debajo de BB- | 150% |

**Opción 2**: Basada en rating de la entidad

| Rating de la Entidad | Ponderación |
|---------------------|-------------|
| AAA a AA- | 20% |
| A+ a A- | 50% |
| BBB+ a BBB- | 50% |
| BB+ a BB- | 100% |
| Por debajo de BB- | 150% |

### Exposiciones a Empresas

| Rating | Ponderación |
|--------|-------------|
| AAA a AA- | 20% |
| A+ a A- | 50% |
| BBB+ a BB- | 100% |
| Por debajo de BB- | 150% |
| Sin calificación | 100% |

### Exposiciones Minoristas

- **Ponderación general**: 75%
- **Requisitos**:
  - Exposición a persona física o PYME
  - Límite de 1 millón EUR por cliente
  - Parte de cartera diversificada

### Exposiciones Garantizadas con Bienes Inmuebles

| Tipo de Garantía | Ponderación |
|------------------|-------------|
| Residencial (LTV ≤ 80%) | 35% |
| Comercial (LTV ≤ 60%) | 50% |
| Residencial (LTV > 80%) | 75%-100% |
| Comercial (LTV > 60%) | 100% |

### Exposiciones en Mora

| Tipo | Ponderación |
|------|-------------|
| Parte no garantizada, provisión < 20% | 150% |
| Parte no garantizada, provisión ≥ 20% | 100% |
| Parte garantizada con inmueble | 100% |

### Factor de Apoyo a PYMES

Para exposiciones a PYMES se aplica un factor reductor:

```
RWA_PYME = RWA × 0.7619
```

Condiciones:
- Ventas anuales ≤ 50 millones EUR
- Exposición total ≤ 1.5 millones EUR

## Técnicas de Mitigación del Riesgo (CRM)

### Garantías Reales Financieras

**Método Simple**:
- Sustituye ponderación de la exposición por la del garante/colateral

**Método Integral**:
```
E* = max(0, E × (1 + He) - C × (1 - Hc - Hfx))
```

Donde:
- E* = Exposición ajustada
- He = Haircut de la exposición
- Hc = Haircut del colateral
- Hfx = Haircut por descalce de divisa

### Garantías Personales

- Sustitución de ponderación si el garante tiene menor riesgo
- Garantes elegibles: soberanos, bancos, empresas con rating

## Ventajas del Método Estándar

- Simplicidad de implementación
- Menor carga operativa
- Transparencia y comparabilidad
- Sin requisitos de validación de modelos

## Limitaciones

- Poca sensibilidad al riesgo real
- Ponderaciones pueden ser muy conservadoras
- No refleja experiencia propia de la entidad
- Dependencia de calificaciones externas

## Cambios en Basilea III Final

A partir de 2025:

1. **Nuevas categorías de exposición**
2. **Ponderaciones revisadas**
3. **Enfoque basado en due diligence** (reducción dependencia de ratings)
4. **Granularidad mejorada** para exposiciones inmobiliarias

## Referencias Normativas

- Reglamento (UE) 575/2013, Artículos 111-141
- Reglamento Delegado (UE) 2015/61 (LCR)
- EBA GL/2020/06 - Guía sobre CRM
- Basilea III: Finalización del marco (diciembre 2017)
