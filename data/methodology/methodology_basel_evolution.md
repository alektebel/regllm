# Evolución del Marco de Basilea: De Basilea III a Basilea IV

## Cronología del Marco de Basilea

| Acuerdo | Año | Enfoque Principal |
|---------|-----|-------------------|
| Basilea I | 1988 | Ratio de capital básico (8%) |
| Basilea II | 2004 | Tres pilares, métodos IRB |
| Basilea III | 2010-2017 | Calidad de capital, liquidez, apalancamiento |
| Basilea III Final ("IV") | 2017 | Output floor, revisión SA, restricciones IRB |

## Basilea III: Reformas Post-Crisis (2010)

### Calidad del Capital

| Componente | Basilea II | Basilea III |
|------------|-----------|-------------|
| CET1 mínimo | 2% | 4.5% |
| Tier 1 mínimo | 4% | 6% |
| Capital total | 8% | 8% |
| Colchón de conservación | - | 2.5% |
| Colchón anticíclico | - | 0-2.5% |

### Nuevos Requisitos

1. **Ratio de Apalancamiento**: ≥ 3% (Tier 1 / Exposición total)
2. **LCR (Liquidity Coverage Ratio)**: ≥ 100%
3. **NSFR (Net Stable Funding Ratio)**: ≥ 100%
4. **Colchones de capital sistémicos**: G-SIB, D-SIB

## Basilea III Final (2017) - "Basilea IV"

### Principales Cambios

#### 1. Output Floor

```
RWA_final = max(RWA_modelos_internos, 72.5% × RWA_estándar)
```

Implementación gradual:
| Año | Floor |
|-----|-------|
| 2023 | 50% |
| 2024 | 55% |
| 2025 | 60% |
| 2026 | 65% |
| 2027 | 70% |
| 2028+ | 72.5% |

#### 2. Revisión del Método Estándar

**Exposiciones a bancos**: Nueva categoría SCRA (Standardised Credit Risk Assessment)

**Exposiciones corporativas**:
| Tipo | Ponderación Anterior | Nueva Ponderación |
|------|---------------------|-------------------|
| SME sin rating | 100% | 85% |
| Investment grade | 50-100% | 65% |
| Subordinada | 150% | 150% |

**Exposiciones minoristas**:
| Tipo | Ponderación |
|------|-------------|
| Regulatory retail | 75% |
| Transactor (tarjetas) | 45% |
| Otros minoristas | 100% |

**Exposiciones inmobiliarias**: Enfoque basado en LTV

| LTV | Residencial | Comercial |
|-----|-------------|-----------|
| ≤50% | 20% | 60% |
| 50-60% | 25% | 70% |
| 60-80% | 30% | 80% |
| 80-90% | 40% | 90% |
| 90-100% | 50% | 110% |
| >100% | 70% | 110% |

#### 3. Restricciones a Modelos Internos

**Eliminación de A-IRB para**:
- Exposiciones a grandes corporaciones (ventas > 500M EUR)
- Exposiciones a entidades financieras
- Exposiciones de renta variable

**Nuevos floors de parámetros**:
| Parámetro | Floor |
|-----------|-------|
| PD (corporativo) | 0.05% |
| PD (retail) | 0.05% - 0.10% |
| LGD (senior sin garantía) | 25% |
| LGD (subordinado) | 50% |
| LGD (garantizado) | Variable según tipo |
| EAD (CCF) | 50% líneas comprometidas |

#### 4. Revisión del Riesgo Operacional

- Eliminación del método de medición avanzada (AMA)
- Nuevo método estándar único basado en:
  - Business Indicator Component (BIC)
  - Internal Loss Multiplier (ILM)

#### 5. CVA (Credit Valuation Adjustment)

- Nuevo método estándar (SA-CVA)
- Método básico (BA-CVA) simplificado
- Restricciones al método de modelo interno (IMA-CVA)

## Comparativa de Impacto

### Entidades con Modelos Internos

| Cartera | Impacto Estimado |
|---------|------------------|
| Corporativa | +15-30% RWA |
| Hipotecaria | +5-15% RWA |
| Minorista | +0-10% RWA |
| Soberanos | Neutral |

### Impacto del Output Floor

Para bancos con modelos internos avanzados:
- Reducción máxima de RWA vs estándar: 27.5% (antes ilimitada)
- Incremento medio estimado de capital: 10-25%

## Implementación en la UE

### CRR3/CRD6

| Elemento | Fecha de Aplicación |
|----------|-------------------|
| Publicación | Junio 2024 |
| Entrada en vigor | Enero 2025 |
| Output floor completo | Enero 2030 |
| Transitional arrangements | 5 años |

### Diferencias con Basilea

- Output floor a nivel consolidado europeo (no de entidad)
- Factor de apoyo PYME ampliado
- Factor de apoyo a infraestructuras
- Tratamiento diferenciado de exposiciones soberanas EU

## Impacto Estratégico

### Para Entidades con IRB

1. **Revisar portfolios**: Identificar carteras más afectadas por floor
2. **Optimizar modelos**: Recalibración dentro de nuevos límites
3. **Gestión de capital**: Planificación de colchones adicionales
4. **Pricing**: Ajuste de márgenes para reflejar mayor consumo de capital

### Para Entidades con Método Estándar

1. **Potencial beneficio**: Algunas ponderaciones reducidas
2. **CRM mejorado**: Mayor reconocimiento de mitigantes
3. **Complejidad**: Nueva granularidad requiere mejor datos

## Referencias

- BCBS d424: Basilea III: Finalización de las reformas post-crisis (2017)
- Reglamento (UE) 2024/XXX (CRR3)
- Directiva (UE) 2024/XXX (CRD6)
- EBA Impact Assessment (2020)
