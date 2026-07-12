export interface DqcItem {
  dqc_id: string;
  variable: string;
  descripcion: string;
  tipo: string;
  severidad: string;
  regla_sql: string;
  condicion_error: string;
  campos_entrada: string[];
  referencia_regulatoria: string;
  umbral: string;
  periodicidad: string;
  justificacion: string;
}

export interface GenerateResponse {
  dqcs: DqcItem[];
  dictionary_fields: number;
  context_summary: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  dqcs?: DqcItem[];
}

// ── Validation pipeline models (mirror api/routers/dqc.py) ─────────────

export interface CheckRecord {
  check_id: string;
  rule_id: string | null;
  name: string;
  description: string;
  severity: string;
  category: string;
  sql: string;
  visible: boolean;
  status: 'pending' | 'validated' | 'rejected';
  reward: number | null;
  variable: string | null;
  tipo: string | null;
  condicion_error: string | null;
  campos_entrada: string[];
  referencia_regulatoria: string | null;
  umbral: string | null;
  periodicidad: string | null;
  justificacion: string | null;
  created_at: string | null;
  validated_at: string | null;
}

export interface CountsResponse {
  pending_visible: number;
  validated: number;
  rejected: number;
  oculto: number;
  dashboard_ready: boolean;
}

export interface DashboardResponse {
  ready: boolean;
  pending_visible: number;
  validated: number;
  rejected: number;
  oculto: number;
  sql: string | null;
  checks: CheckRecord[];
}
