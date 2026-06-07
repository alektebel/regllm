-- ============================================================
-- IRB Model Data Schema — Banco Ejemplo S.A.
-- Equipo de Modelos Avanzados / Riesgo de Crédito
-- ============================================================

-- ── CONTRATOS ───────────────────────────────────────────────
CREATE TABLE contratos (
    contrato_id         BIGSERIAL PRIMARY KEY,                -- Identificador único del contrato
    cliente_id          BIGINT       NOT NULL,                -- FK al cliente titular
    producto_id         INTEGER      NOT NULL,                -- Tipo de producto crediticio
    fecha_apertura      DATE         NOT NULL,                -- Fecha de formalización del contrato
    fecha_vencimiento   DATE,                                 -- Fecha de vencimiento contractual
    fecha_cierre        DATE,                                 -- Fecha de cierre efectivo (NULL si activo)
    importe_original    NUMERIC(18,2) NOT NULL,               -- Importe concedido original en EUR
    saldo_dispuesto     NUMERIC(18,2) NOT NULL DEFAULT 0,     -- Saldo vivo actualmente dispuesto
    saldo_no_dispuesto  NUMERIC(18,2) NOT NULL DEFAULT 0,     -- Límite no dispuesto (líneas de crédito)
    moneda              CHAR(3)      NOT NULL DEFAULT 'EUR',  -- ISO 4217 (EUR, USD, GBP…)
    tipo_interes        NUMERIC(8,5),                         -- Tipo de interés contractual anual (%)
    segmento_irb        VARCHAR(20)  NOT NULL,                -- Segmento IRB: RETAIL_HIP / RETAIL_CONS / CORP / PYME / SOB
    clase_activo        VARCHAR(30)  NOT NULL,                -- Clase de activo CRR: mortgage / revolving / other_retail / corporate
    estado              VARCHAR(20)  NOT NULL CHECK (estado IN ('ACTIVO','CERRADO','DEFAULT','REFINANCIADO','DUDOSO')),
    motivo_cierre       VARCHAR(50),                          -- Razón de cierre si estado = CERRADO
    FOREIGN KEY (cliente_id) REFERENCES clientes(cliente_id),
    FOREIGN KEY (producto_id) REFERENCES productos(producto_id)
);

COMMENT ON COLUMN contratos.saldo_dispuesto IS 'EAD componente: exposición en balance usada para el cálculo de capital regulatorio (CRR Art. 166)';
COMMENT ON COLUMN contratos.saldo_no_dispuesto IS 'Parte no dispuesta de líneas; se multiplica por CCF para obtener EAD off-balance (CRR Art. 166.8)';
COMMENT ON COLUMN contratos.segmento_irb IS 'Segmentación regulatoria IRB según CRR Art. 147. Determina función de ponderación de riesgo aplicable';
COMMENT ON COLUMN contratos.clase_activo IS 'Clase de exposición CRR Art. 147: mortgage / revolving / other_retail / corporate / sovereign / institution';

-- ── CLIENTES ────────────────────────────────────────────────
CREATE TABLE clientes (
    cliente_id          BIGSERIAL PRIMARY KEY,
    tipo_persona        CHAR(1)      NOT NULL CHECK (tipo_persona IN ('F','J')),  -- F=física, J=jurídica
    segmento            VARCHAR(20)  NOT NULL,                -- Segmento comercial: PARTICULARES / PYMES / EMPRESAS / INSTITUCIONES
    pais_residencia     CHAR(2)      NOT NULL DEFAULT 'ES',   -- ISO 3166-1 alpha-2
    fecha_alta          DATE         NOT NULL,
    ingresos_anuales    NUMERIC(15,2),                        -- Ingresos brutos anuales declarados (persona física)
    facturacion_anual   NUMERIC(15,2),                        -- Cifra de negocio anual (persona jurídica)
    empleados           INTEGER,                              -- Número de empleados (PYME/empresa)
    rating_interno      VARCHAR(10),                          -- Rating interno del modelo masterscale
    score_admision      NUMERIC(6,4),                        -- Score de admisión en el momento del alta [0,1]
    flag_vinculado      BOOLEAN      NOT NULL DEFAULT FALSE,  -- TRUE si el cliente tiene vinculación con la entidad
    flag_pep            BOOLEAN      NOT NULL DEFAULT FALSE   -- TRUE si el cliente es persona políticamente expuesta
);

COMMENT ON COLUMN clientes.tipo_persona IS 'Distingue personas físicas (F) de jurídicas (J) a efectos de segmentación IRB (CRR Art. 147.5)';
COMMENT ON COLUMN clientes.facturacion_anual IS 'Se usa para clasificar PYME: facturación < 50M EUR según CRR Art. 501 para aplicar el SME supporting factor';
COMMENT ON COLUMN clientes.rating_interno IS 'Grado de la masterscale interna (1=mejor calidad → 16=default). Debe mapear a PD calibrada (EBA GL 2017/16 §73)';

-- ── PARAMETROS_IRB ──────────────────────────────────────────
CREATE TABLE parametros_irb (
    param_id            BIGSERIAL PRIMARY KEY,
    contrato_id         BIGINT       NOT NULL,
    fecha_calculo       DATE         NOT NULL,                -- Fecha de referencia del cálculo
    pd_estimada         NUMERIC(10,8) NOT NULL,               -- PD calibrada del modelo [0,1]
    pd_suelo            NUMERIC(10,8) NOT NULL DEFAULT 0.0003,-- Suelo mínimo PD (0.03% CRR Art. 160.1)
    pd_final            NUMERIC(10,8) NOT NULL,               -- max(pd_estimada, pd_suelo)
    lgd_estimada        NUMERIC(10,8) NOT NULL,               -- LGD en condiciones normales [0,1]
    lgd_downturn        NUMERIC(10,8) NOT NULL,               -- LGD downturn (CRR Art. 181.1.b)
    lgd_suelo           NUMERIC(10,8),                        -- Suelo LGD según tipo colateral (CRR Art. 161.1)
    lgd_final           NUMERIC(10,8) NOT NULL,               -- max(lgd_downturn, lgd_suelo)
    ead_balance         NUMERIC(18,2) NOT NULL,               -- EAD en balance
    ead_fuera_balance   NUMERIC(18,2) NOT NULL DEFAULT 0,     -- EAD fuera de balance (saldo_no_dispuesto * CCF)
    ccf_estimado        NUMERIC(8,6),                         -- CCF modelo para la parte no dispuesta [0,1]
    ccf_suelo           NUMERIC(8,6) NOT NULL DEFAULT 0,      -- Suelo CCF regulatorio
    ead_total           NUMERIC(18,2) NOT NULL,               -- EAD total = ead_balance + ead_fuera_balance
    m_vencimiento       NUMERIC(6,4),                         -- Vencimiento efectivo M en años (CRR Art. 162)
    m_suelo             NUMERIC(6,4) NOT NULL DEFAULT 1.0,    -- Suelo vencimiento = 1 año
    m_techo             NUMERIC(6,4) NOT NULL DEFAULT 5.0,    -- Techo vencimiento = 5 años
    rwa                 NUMERIC(18,2) NOT NULL,               -- Activos ponderados por riesgo (RWA)
    capital_regulatorio NUMERIC(18,2) NOT NULL,               -- Capital regulatorio = RWA * 8%
    modelo_pd_id        VARCHAR(30),                          -- Identificador del modelo PD utilizado
    modelo_lgd_id       VARCHAR(30),                          -- Identificador del modelo LGD utilizado
    version_modelo      VARCHAR(20),                          -- Versión del modelo en producción
    FOREIGN KEY (contrato_id) REFERENCES contratos(contrato_id)
);

COMMENT ON COLUMN parametros_irb.pd_suelo IS 'Suelo mínimo de PD = 0.03% según CRR Art. 160.1 para exposiciones no en default';
COMMENT ON COLUMN parametros_irb.lgd_downturn IS 'LGD que refleja condiciones económicas adversas. Obligatorio según CRR Art. 181.1(b) y EBA GL 2019/03';
COMMENT ON COLUMN parametros_irb.lgd_suelo IS 'Suelos mínimos por tipo de colateral: 0% (sin colateral retail), 10% (financiero), 15% (inmobiliario comercial), 30% (hipotecario residencial) según CRR Art. 161.1';
COMMENT ON COLUMN parametros_irb.ccf_estimado IS 'Credit Conversion Factor estimado por modelo interno para exposiciones fuera de balance. No puede ser negativo (CRR Art. 182.1)';
COMMENT ON COLUMN parametros_irb.m_vencimiento IS 'Vencimiento efectivo en años. Para exposiciones corporativas, rango [1,5]. Para minoristas, M fijo = 2.5 (CRR Art. 162)';

-- ── DEFAULTS ────────────────────────────────────────────────
CREATE TABLE defaults (
    default_id          BIGSERIAL PRIMARY KEY,
    contrato_id         BIGINT       NOT NULL,
    cliente_id          BIGINT       NOT NULL,
    fecha_default       DATE         NOT NULL,               -- Fecha de clasificación en default
    causa_default       VARCHAR(30)  NOT NULL CHECK (causa_default IN ('90_DIAS_VENCIDO','IMPROBABLE_PAGO','QUIEBRA','REFINANCIACION_DETERIORO','CONDONACION')),
    dias_mora_trigger   INTEGER,                             -- Días en mora en el momento del trigger (si causa = 90_DIAS_VENCIDO)
    importe_vencido     NUMERIC(18,2),                       -- Importe vencido en fecha de default
    ead_en_default      NUMERIC(18,2) NOT NULL,              -- EAD en el momento del default
    fecha_salida        DATE,                                 -- Fecha de salida del default (NULL si continúa)
    tipo_salida         VARCHAR(30),                         -- CURA / LIQUIDACION / FALLIDO / VENTA_CARTERA
    importe_recuperado  NUMERIC(18,2) NOT NULL DEFAULT 0,    -- Total recuperado (incluye colateral + gestión)
    coste_recuperacion  NUMERIC(18,2) NOT NULL DEFAULT 0,    -- Costes directos del proceso de recuperación
    lgd_realizada       NUMERIC(10,8),                       -- LGD observada = 1 - (rec - costes) / EAD
    periodo_recuperacion INTEGER,                            -- Meses hasta cierre del proceso de recuperación
    FOREIGN KEY (contrato_id) REFERENCES contratos(contrato_id),
    FOREIGN KEY (cliente_id) REFERENCES clientes(cliente_id)
);

COMMENT ON COLUMN defaults.causa_default IS 'Causa de default conforme a CRR Art. 178. Debe registrarse al menos 90 días de mora para exposiciones minoristas e hipotecarias';
COMMENT ON COLUMN defaults.dias_mora_trigger IS 'Número de días en mora. Umbral de 90 días según CRR Art. 178.1(b). Posible extensión a 180 días para administraciones públicas';
COMMENT ON COLUMN defaults.lgd_realizada IS 'LGD observada ex-post. Se usa para calibración y backtesting del modelo LGD. Referencia: EBA GL 2017/16 §135-143';
COMMENT ON COLUMN defaults.periodo_recuperacion IS 'Período de recuperación observado en meses. EBA GL 2019/03 exige aplicar factor de descuento a los flujos recuperados';

-- ── COLATERALES ─────────────────────────────────────────────
CREATE TABLE colaterales (
    colateral_id        BIGSERIAL PRIMARY KEY,
    contrato_id         BIGINT       NOT NULL,
    tipo_colateral      VARCHAR(30)  NOT NULL CHECK (tipo_colateral IN ('HIPOTECA_RESIDENCIAL','HIPOTECA_COMERCIAL','GARANTIA_PERSONAL','PRENDA_FINANCIERA','PRENDA_INMOBILIARIA','SIN_COLATERAL')),
    valor_tasacion      NUMERIC(18,2),                       -- Valor de tasación vigente
    fecha_tasacion      DATE,                                 -- Fecha de la última tasación
    ltv_actual          NUMERIC(8,5),                        -- LTV actual = saldo_dispuesto / valor_tasacion
    haircut_regulatorio NUMERIC(8,5),                        -- Haircut regulatorio aplicado al colateral
    valor_neto          NUMERIC(18,2),                       -- Valor neto = valor_tasacion * (1 - haircut)
    elegible_crm        BOOLEAN      NOT NULL DEFAULT FALSE, -- TRUE si elegible para Credit Risk Mitigation (CRR Parte 3 Título II Cap. 4)
    FOREIGN KEY (contrato_id) REFERENCES contratos(contrato_id)
);

COMMENT ON COLUMN colaterales.tipo_colateral IS 'Tipo de colateral relevante para suelos LGD IRB: hipoteca residencial (suelo 30%), comercial (15%), prenda financiera (10%)';
COMMENT ON COLUMN colaterales.ltv_actual IS 'Loan-To-Value. Para hipotecas residenciales, LTV > 80% puede implicar suelo LGD más elevado según EBA GL 2020/06';
COMMENT ON COLUMN colaterales.elegible_crm IS 'Indica si el colateral cumple requisitos de elegibilidad CRM del CRR Parte 3 Título II Cap. 4 para reducir RWA';

-- ── CICLOS_RECUPERACION ─────────────────────────────────────
CREATE TABLE ciclos_recuperacion (
    ciclo_id            BIGSERIAL PRIMARY KEY,
    default_id          BIGINT       NOT NULL,
    mes_referencia      DATE         NOT NULL,               -- Mes del flujo de recuperación
    flujo_recuperado    NUMERIC(15,2) NOT NULL DEFAULT 0,    -- Flujo recuperado en el mes
    coste_directo       NUMERIC(15,2) NOT NULL DEFAULT 0,    -- Coste directo asociado al proceso
    tasa_descuento      NUMERIC(8,6)  NOT NULL,              -- Tasa de descuento aplicada al flujo
    flujo_descontado    NUMERIC(15,2) NOT NULL,              -- Flujo descontado a fecha de default
    fuente_recuperacion VARCHAR(30)   NOT NULL CHECK (fuente_recuperacion IN ('PAGO_VOLUNTARIO','SUBASTA','DACCION','SEGURO','GARANTIA','VENTA_DEUDA')),
    provision_periodo   INTEGER       NOT NULL,              -- Meses desde la fecha de default
    FOREIGN KEY (default_id) REFERENCES defaults(default_id)
);

COMMENT ON COLUMN ciclos_recuperacion.tasa_descuento IS 'Tasa de descuento para NPV de recuperaciones. EBA GL 2019/03 §85 requiere usar tipo contractual o tasa de referencia cuando el tipo sea desconocido';
COMMENT ON COLUMN ciclos_recuperacion.provision_periodo IS 'Meses transcurridos desde la fecha de default. Permite verificar si el ciclo de recuperación está completo (mínimo 7 años para RETAIL según EBA GL 2017/16)';

-- ── MODELOS_VALIDACION ──────────────────────────────────────
CREATE TABLE modelos_validacion (
    validacion_id       BIGSERIAL PRIMARY KEY,
    modelo_id           VARCHAR(30)  NOT NULL,               -- Identificador del modelo validado
    tipo_modelo         VARCHAR(20)  NOT NULL CHECK (tipo_modelo IN ('PD','LGD','EAD','CCF','SCORING')),
    fecha_validacion    DATE         NOT NULL,
    resultado           VARCHAR(20)  NOT NULL CHECK (resultado IN ('APROBADO','APROBADO_CON_RESERVAS','RECHAZADO','EN_REVISION')),
    test_discriminacion NUMERIC(8,6),                        -- AUC-ROC del test de discriminación
    test_calibracion    NUMERIC(8,6),                        -- Binomial/Hosmer-Lemeshow p-value
    test_estabilidad    NUMERIC(8,6),                        -- PSI (Population Stability Index)
    backtesting_lgd     NUMERIC(8,6),                        -- Ratio LGD estimada/observada para backtesting
    observaciones       TEXT,                                 -- Observaciones del equipo de validación
    proxima_revision    DATE                                  -- Fecha prevista de próxima revisión
);

COMMENT ON COLUMN modelos_validacion.test_discriminacion IS 'AUC-ROC: mide la capacidad discriminante del modelo PD. EBA GL 2017/16 §125 exige AUC > 0.70 como referencia mínima';
COMMENT ON COLUMN modelos_validacion.test_calibracion IS 'p-valor del test de calibración (Hosmer-Lemeshow o binomial). Valores < 0.05 indican mala calibración y pueden requerir recalibración (EBA GL 2017/16 §126)';
COMMENT ON COLUMN modelos_validacion.test_estabilidad IS 'PSI (Population Stability Index). PSI > 0.25 indica cambio significativo en la población y requiere revisión del modelo según política interna';
