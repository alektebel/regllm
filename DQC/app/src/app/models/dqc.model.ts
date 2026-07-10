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

export interface RAGSource {
  document: string;
  heading: string;
  snippet: string;
  source_type: string;
}

export interface DqcResponse {
  variable: string;
  dqcs: DqcItem[];
  context_summary: string;
  sources: RAGSource[];
}

// Simple mode: natural-language expressions + at most one article, no RAG —
// see POST /dqc/generate/simple.
export interface SimpleDqcResponse {
  dqcs: DqcItem[];
  article_citation: string;
  article_text_used: string;
  context_summary: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  dqcs?: DqcItem[];
  sources?: RAGSource[];
}

// ── Validation pipeline models (mirror api/routers/dqc.py) ─────────────

export interface CheckRecord {
  check_id: string;
  rule_id: string | null;
  name: string;
  description: string;
  severity: string;        // HIGH | MED | LOW
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
