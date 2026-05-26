"use client";

import { useCallback, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Download,
  FileCode2,
  FileSpreadsheet,
  Loader2,
  ShieldCheck,
  Upload,
  XCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";

// ── Types ─────────────────────────────────────────────────────────────────────

type Mode = "data" | "annotate";
type EntityType = "ciclos" | "contratos" | "titularidades" | "generic";

const ENTITY_OPTIONS: { value: EntityType; label: string; description: string }[] = [
  { value: "ciclos",        label: "Ciclos",        description: "Recovery cycles — LGD, provision periods, DPDs" },
  { value: "contratos",     label: "Contratos",     description: "Loan contracts — segmentation, observation window" },
  { value: "titularidades", label: "Titularidades", description: "Account holders & guarantors — CRR Art. 213 eligibility" },
  { value: "generic",       label: "Genérico",      description: "Any credit-risk parameter table" },
];

interface UploadInfo {
  file_id: string;
  filename: string;
  size: number;
  detected_mode: Mode;
}

interface ComplianceFinding {
  row_id: string;
  verdict: "compliant" | "flagged" | "uncertain";
  resumen: string;
  flags: string[];
  articles: string[];
  row_data?: Record<string, unknown>;
}

interface ComplianceReport {
  table: string;
  run_date: string;
  rows_processed: number;
  rows_flagged: number;
  rows_uncertain: number;
  compliance_rate: number;
  llm_calls: number;
  findings: ComplianceFinding[];
  narrative?: { resumen?: string; advertencias?: string };
  tier_counts?: {
    tier1_flagged: number;
    tier2_rows: number;
    tier3_outlier_rows: number;
  };
}

interface AnnotationResult {
  code: string;
  blocks: number;
  warnings: { task: string; text: string }[];
}

// ── Constants ─────────────────────────────────────────────────────────────────

const API = "/api";

function authHeaders(): Record<string, string> {
  return {};
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function CompliancePage() {
  const [upload, setUpload] = useState<UploadInfo | null>(null);
  const [mode, setMode] = useState<Mode>("data");
  const [entityType, setEntityType] = useState<EntityType>("ciclos");
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<string[]>([]);
  const [report, setReport] = useState<ComplianceReport | null>(null);
  const [annotation, setAnnotation] = useState<AnnotationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [noRag, setNoRag] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  // ── Upload ──────────────────────────────────────────────────────────────────

  const handleFile = useCallback(async (file: File) => {
    setUploading(true);
    setError(null);
    setReport(null);
    setAnnotation(null);
    setProgress([]);

    const form = new FormData();
    form.append("file", file);

    try {
      const res = await fetch(`${API}/compliance/upload`, {
        method: "POST",
        headers: { ...authHeaders() },
        body: form,
      });
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.detail ?? `Upload failed (${res.status})`);
      }
      const info: UploadInfo = await res.json();
      setUpload(info);
      setMode(info.detected_mode);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setUploading(false);
    }
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  // ── Run ─────────────────────────────────────────────────────────────────────

  async function handleRun() {
    if (!upload || running) return;
    setRunning(true);
    setError(null);
    setReport(null);
    setAnnotation(null);
    setProgress([`Starting ${mode === "annotate" ? "SAS annotation" : "compliance check"}…`]);

    try {
      const res = await fetch(`${API}/compliance/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({
          file_id: upload.file_id,
          mode,
          entity_type: entityType,
          no_rag: noRag,
          tiered: true,
          backend: "ollama",
          ollama_model: "phi3:mini",
        }),
      });

      if (!res.ok || !res.body) throw new Error(`Run failed (${res.status})`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        const lines = buf.split("\n");
        buf = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const evt = JSON.parse(line.slice(6));
          if (evt.type === "progress") {
            setProgress((p) => [...p, evt.message]);
          } else if (evt.type === "result") {
            setReport(evt.report);
            setProgress((p) => [...p, "Done."]);
          } else if (evt.type === "annotated") {
            setAnnotation({ code: evt.code, blocks: evt.blocks, warnings: evt.warnings });
            setProgress((p) => [...p, `Annotated ${evt.blocks} code block(s).`]);
          } else if (evt.type === "error") {
            throw new Error(evt.message);
          }
        }
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-full max-w-5xl mx-auto px-6 py-6 gap-6 overflow-y-auto">
      {/* Header */}
      <div className="flex items-center gap-3">
        <ShieldCheck className="text-primary" size={24} />
        <div>
          <h1 className="text-xl font-semibold">Compliance Checker</h1>
          <p className="text-sm text-muted-foreground">
            Validate IRB/IFRS9 data against EBA guidelines and CRR regulation · Annotate SAS/EGP code
          </p>
        </div>
      </div>

      {/* Upload zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => !upload && fileRef.current?.click()}
        className={cn(
          "border-2 border-dashed rounded-xl p-8 text-center transition-colors",
          dragging ? "border-primary bg-primary/5" : "border-border hover:border-primary/50",
          !upload && "cursor-pointer"
        )}
      >
        <input
          ref={fileRef}
          type="file"
          className="hidden"
          accept=".csv,.parquet,.sas,.egp,.txt,.pdf"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
        />

        {uploading ? (
          <div className="flex flex-col items-center gap-2 text-muted-foreground">
            <Loader2 className="animate-spin" size={28} />
            <span className="text-sm">Uploading…</span>
          </div>
        ) : upload ? (
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {upload.detected_mode === "annotate" ? (
                <FileCode2 size={28} className="text-primary" />
              ) : (
                <FileSpreadsheet size={28} className="text-primary" />
              )}
              <div className="text-left">
                <p className="font-medium text-sm">{upload.filename}</p>
                <p className="text-xs text-muted-foreground">
                  {(upload.size / 1024).toFixed(1)} KB · detected: {upload.detected_mode === "annotate" ? "SAS code" : "data table"}
                </p>
              </div>
            </div>
            <button
              onClick={(e) => { e.stopPropagation(); setUpload(null); setReport(null); setAnnotation(null); setProgress([]); fileRef.current?.click(); }}
              className="text-xs text-muted-foreground hover:text-foreground underline"
            >
              Change file
            </button>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2 text-muted-foreground">
            <Upload size={28} />
            <p className="text-sm font-medium">Drop file here or click to browse</p>
            <p className="text-xs">CSV · Parquet · SAS · EGP · TXT · PDF</p>
            <a
              href={`${API}/compliance/sample`}
              onClick={(e) => e.stopPropagation()}
              className="mt-2 text-xs text-primary hover:underline flex items-center gap-1"
            >
              <Download size={12} /> Download sample data
            </a>
          </div>
        )}
      </div>

      {/* Mode + options */}
      {upload && (
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex rounded-lg border border-border overflow-hidden text-sm">
            {(["data", "annotate"] as Mode[]).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={cn(
                  "px-4 py-2 font-medium transition-colors",
                  mode === m
                    ? "bg-primary text-primary-foreground"
                    : "bg-background text-muted-foreground hover:text-foreground"
                )}
              >
                {m === "data" ? "Check Data" : "Annotate SAS Code"}
              </button>
            ))}
          </div>

          {mode === "data" && (
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs text-muted-foreground font-medium">Entity:</span>
              {ENTITY_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => setEntityType(opt.value)}
                  title={opt.description}
                  className={cn(
                    "rounded-md px-3 py-1.5 text-xs font-medium border transition-colors",
                    entityType === opt.value
                      ? "bg-primary/10 border-primary text-primary"
                      : "border-border text-muted-foreground hover:border-primary/50 hover:text-foreground"
                  )}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          )}

          <label className="flex items-center gap-2 text-sm text-muted-foreground cursor-pointer select-none">
            <input
              type="checkbox"
              checked={noRag}
              onChange={(e) => setNoRag(e.target.checked)}
              className="rounded"
            />
            Skip RAG (no DB)
          </label>

          <button
            onClick={handleRun}
            disabled={running}
            className="ml-auto flex items-center gap-2 rounded-lg bg-primary text-primary-foreground px-5 py-2.5 text-sm font-medium hover:bg-primary/90 disabled:opacity-50 transition-colors"
          >
            {running ? <Loader2 size={15} className="animate-spin" /> : <ShieldCheck size={15} />}
            {running ? "Running…" : "Run Check"}
          </button>
        </div>
      )}

      {/* Progress log */}
      {progress.length > 0 && (
        <div className="rounded-lg bg-muted/40 border border-border px-4 py-3 text-xs font-mono space-y-0.5 text-muted-foreground">
          {progress.map((msg, i) => (
            <div key={i} className="flex items-center gap-2">
              {i === progress.length - 1 && running ? (
                <Loader2 size={10} className="animate-spin shrink-0" />
              ) : (
                <span className="shrink-0">›</span>
              )}
              {msg}
            </div>
          ))}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex items-start gap-3 rounded-lg border border-destructive/50 bg-destructive/10 p-4 text-sm text-destructive">
          <XCircle size={16} className="shrink-0 mt-0.5" />
          <div>
            <p className="font-medium">Error</p>
            <p className="text-xs mt-1 opacity-80">{error}</p>
          </div>
        </div>
      )}

      {/* Results: Data validation report */}
      {report && <DataReport report={report} />}

      {/* Results: SAS annotation */}
      {annotation && <AnnotationView annotation={annotation} />}
    </div>
  );
}

// ── Data Report ───────────────────────────────────────────────────────────────

function DataReport({ report }: { report: ComplianceReport }) {
  const [expandedRow, setExpandedRow] = useState<string | null>(null);
  const flagged = report.findings.filter((f) => f.verdict === "flagged");
  const uncertain = report.findings.filter((f) => f.verdict === "uncertain");

  return (
    <div className="space-y-5">
      {/* Summary cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard label="Rows processed" value={report.rows_processed.toLocaleString()} />
        <StatCard
          label="Flagged"
          value={report.rows_flagged.toLocaleString()}
          variant={report.rows_flagged > 0 ? "danger" : "ok"}
        />
        <StatCard
          label="Uncertain"
          value={report.rows_uncertain.toLocaleString()}
          variant={report.rows_uncertain > 0 ? "warn" : "ok"}
        />
        <StatCard
          label="Compliance rate"
          value={`${(report.compliance_rate * 100).toFixed(1)}%`}
          variant={report.compliance_rate >= 0.99 ? "ok" : report.compliance_rate >= 0.95 ? "warn" : "danger"}
        />
      </div>

      {/* Tier breakdown */}
      {report.tier_counts && (
        <div className="text-xs text-muted-foreground flex flex-wrap gap-x-4 gap-y-1 px-1">
          <span>Tier 1 (hard rules): {report.tier_counts.tier1_flagged} flagged</span>
          <span>Tier 2 (cluster): {report.tier_counts.tier2_rows} rows evaluated</span>
          <span>Tier 3 (outlier): {report.tier_counts.tier3_outlier_rows} rows</span>
          <span>LLM calls: {report.llm_calls}</span>
        </div>
      )}

      {/* Narrative */}
      {report.narrative?.resumen && (
        <div className="rounded-lg border border-border bg-card p-4 text-sm">
          <p className="font-medium text-muted-foreground text-xs mb-1 uppercase tracking-wide">Summary</p>
          <p>{report.narrative.resumen}</p>
          {report.narrative.advertencias && (
            <p className="mt-2 text-amber-600 dark:text-amber-400 text-xs flex items-start gap-1.5">
              <AlertTriangle size={12} className="shrink-0 mt-0.5" />
              {report.narrative.advertencias}
            </p>
          )}
        </div>
      )}

      {/* Violations table */}
      {flagged.length > 0 && (
        <div className="space-y-2">
          <h2 className="text-sm font-semibold flex items-center gap-2">
            <XCircle size={15} className="text-destructive" />
            Flagged findings ({flagged.length})
          </h2>
          <div className="rounded-lg border border-border overflow-hidden text-sm">
            {flagged.map((f) => (
              <FindingRow
                key={f.row_id}
                finding={f}
                expanded={expandedRow === f.row_id}
                onToggle={() => setExpandedRow(expandedRow === f.row_id ? null : f.row_id)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Uncertain table */}
      {uncertain.length > 0 && (
        <div className="space-y-2">
          <h2 className="text-sm font-semibold flex items-center gap-2">
            <AlertTriangle size={15} className="text-amber-500" />
            Uncertain ({uncertain.length})
          </h2>
          <div className="rounded-lg border border-border overflow-hidden text-sm">
            {uncertain.map((f) => (
              <FindingRow
                key={f.row_id}
                finding={f}
                expanded={expandedRow === f.row_id}
                onToggle={() => setExpandedRow(expandedRow === f.row_id ? null : f.row_id)}
              />
            ))}
          </div>
        </div>
      )}

      {flagged.length === 0 && uncertain.length === 0 && (
        <div className="flex items-center gap-3 rounded-lg border border-green-200 dark:border-green-900 bg-green-50 dark:bg-green-950/30 p-4 text-sm text-green-700 dark:text-green-400">
          <CheckCircle2 size={18} />
          All rows passed compliance checks.
        </div>
      )}
    </div>
  );
}

function FindingRow({
  finding,
  expanded,
  onToggle,
}: {
  finding: ComplianceFinding;
  expanded: boolean;
  onToggle: () => void;
}) {
  const verdictColor =
    finding.verdict === "flagged"
      ? "text-destructive"
      : "text-amber-500";

  return (
    <div className="border-b border-border last:border-0">
      <button
        onClick={onToggle}
        className="w-full flex items-start gap-3 px-4 py-3 text-left hover:bg-muted/30 transition-colors"
      >
        <span className={cn("shrink-0 font-mono text-xs mt-0.5", verdictColor)}>
          {finding.verdict === "flagged" ? "✗" : "?"}
        </span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-mono text-xs text-muted-foreground">{finding.row_id}</span>
            {finding.articles.map((a) => (
              <span key={a} className="text-[10px] bg-primary/10 text-primary rounded px-1.5 py-0.5 font-medium">
                {a}
              </span>
            ))}
          </div>
          <p className="text-sm mt-0.5 truncate">{finding.resumen}</p>
        </div>
        {expanded ? <ChevronUp size={14} className="shrink-0 mt-1 text-muted-foreground" /> : <ChevronDown size={14} className="shrink-0 mt-1 text-muted-foreground" />}
      </button>

      {expanded && (
        <div className="px-4 pb-4 space-y-3 bg-muted/20">
          {finding.flags.length > 0 && (
            <ul className="text-xs space-y-1">
              {finding.flags.map((flag, i) => (
                <li key={i} className="flex items-start gap-2 text-destructive">
                  <span className="shrink-0">·</span>
                  {flag}
                </li>
              ))}
            </ul>
          )}
          {finding.row_data && (
            <pre className="text-[11px] bg-muted rounded p-3 overflow-x-auto text-muted-foreground">
              {JSON.stringify(finding.row_data, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

// ── Annotation view ───────────────────────────────────────────────────────────

function AnnotationView({ annotation }: { annotation: AnnotationResult }) {
  const lines = annotation.code.split("\n");

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 flex-wrap">
        <h2 className="text-sm font-semibold flex items-center gap-2">
          <FileCode2 size={15} />
          Annotated SAS Code ({annotation.blocks} block{annotation.blocks !== 1 ? "s" : ""})
        </h2>
        {annotation.warnings.length > 0 && (
          <span className="text-xs bg-amber-100 dark:bg-amber-950/40 text-amber-700 dark:text-amber-400 rounded px-2 py-0.5 font-medium">
            {annotation.warnings.length} potential issue{annotation.warnings.length !== 1 ? "s" : ""}
          </span>
        )}
        <button
          onClick={() => {
            const blob = new Blob([annotation.code], { type: "text/plain" });
            const a = document.createElement("a");
            a.href = URL.createObjectURL(blob);
            a.download = "annotated.sas";
            a.click();
          }}
          className="ml-auto flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground"
        >
          <Download size={12} /> Download
        </button>
      </div>

      {annotation.warnings.length > 0 && (
        <div className="rounded-lg border border-amber-200 dark:border-amber-900 bg-amber-50 dark:bg-amber-950/20 p-3 space-y-1">
          {annotation.warnings.map((w, i) => (
            <div key={i} className="text-xs text-amber-700 dark:text-amber-400 flex items-start gap-2">
              <AlertTriangle size={11} className="shrink-0 mt-0.5" />
              <span><span className="font-medium">[{w.task}]</span> {w.text}</span>
            </div>
          ))}
        </div>
      )}

      <div className="rounded-lg border border-border bg-[#1e1e2e] overflow-x-auto">
        <pre className="text-[12px] leading-relaxed p-5">
          {lines.map((line, i) => {
            const isComment = line.trimStart().startsWith("/*") || line.trimStart().startsWith("*");
            const isWarning = line.toLowerCase().includes("incumpl") || line.toLowerCase().includes("viola") || line.toLowerCase().includes("inferior");
            const isSectionHeader = line.startsWith("/* ══");
            return (
              <div
                key={i}
                className={cn(
                  "px-0",
                  isSectionHeader && "text-blue-400 font-semibold",
                  !isSectionHeader && isComment && isWarning && "text-red-400",
                  !isSectionHeader && isComment && !isWarning && "text-emerald-400",
                  !isComment && "text-slate-200"
                )}
              >
                {line || " "}
              </div>
            );
          })}
        </pre>
      </div>
    </div>
  );
}

// ── Helper components ─────────────────────────────────────────────────────────

function StatCard({
  label,
  value,
  variant = "neutral",
}: {
  label: string;
  value: string;
  variant?: "ok" | "warn" | "danger" | "neutral";
}) {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <p className="text-xs text-muted-foreground mb-1">{label}</p>
      <p
        className={cn(
          "text-2xl font-bold",
          variant === "ok" && "text-green-600 dark:text-green-400",
          variant === "warn" && "text-amber-600 dark:text-amber-400",
          variant === "danger" && "text-destructive"
        )}
      >
        {value}
      </p>
    </div>
  );
}
