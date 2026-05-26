"use client";

import { useEffect, useRef, useState } from "react";
import {
  Upload, Play, Database, GitBranch, ChevronDown, ChevronRight,
  CheckCircle2, XCircle, AlertTriangle, Loader2, RefreshCw, Zap,
} from "lucide-react";
import { cn } from "@/lib/utils";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Types ────────────────────────────────────────────────────────────────────

interface StatusData {
  corpus: { total_chunks: number; embedded: number; documents: { doc: string; chunks: number }[] };
  assertions: { total: number; with_sql: number; pending_sql: number; by_materiality: Record<string, number> };
  context: { chunks: number };
  graph: { nodes: number; edges: number };
}

interface Assertion {
  id: string;
  regulation_id: string;
  regulation_name: string;
  article: string;
  rule_text: string;
  check_type: string;
  materiality: string;
  conditions: string[];
  threshold: { field_hint: string; op: string; value: number; unit: string } | null;
  sql_template: string | null;
  notes: string;
}

interface Finding {
  assertion_id: string;
  regulation_name: string;
  article: string;
  rule_text: string;
  materiality: string;
  n_failing: number;
  status: "PASS" | "FAIL";
  downstream_nodes: number;
  result_sample: Record<string, unknown>[];
}

// ── Helpers ──────────────────────────────────────────────────────────────────

const MATERIALITY_COLOR: Record<string, string> = {
  critical: "text-red-500 bg-red-50 border-red-200",
  high:     "text-orange-500 bg-orange-50 border-orange-200",
  medium:   "text-yellow-600 bg-yellow-50 border-yellow-200",
  low:      "text-green-600 bg-green-50 border-green-200",
};

function Badge({ level }: { level: string }) {
  return (
    <span className={cn("text-xs font-medium px-1.5 py-0.5 rounded border", MATERIALITY_COLOR[level] ?? "text-muted-foreground bg-muted border-border")}>
      {level}
    </span>
  );
}

function StatCard({ label, value, sub }: { label: string; value: number | string; sub?: string }) {
  return (
    <div className="rounded-lg border border-border bg-card px-4 py-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-2xl font-semibold mt-0.5">{value}</p>
      {sub && <p className="text-xs text-muted-foreground mt-0.5">{sub}</p>}
    </div>
  );
}

// ── Section: Status ───────────────────────────────────────────────────────────

function StatusPanel({ status, onRefresh }: { status: StatusData | null; onRefresh: () => void }) {
  if (!status) return (
    <div className="flex items-center gap-2 text-sm text-muted-foreground py-8">
      <Loader2 size={14} className="animate-spin" /> Loading…
    </div>
  );
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard label="Regulation chunks" value={status.corpus.total_chunks} sub={`${status.corpus.embedded} embedded`} />
        <StatCard label="Assertions" value={status.assertions.total} sub={`${status.assertions.with_sql} with SQL`} />
        <StatCard label="Context chunks" value={status.context.chunks} sub="column/field mappings" />
        <StatCard label="Graph nodes" value={status.graph.nodes} sub={`${status.graph.edges} edges`} />
      </div>
      <div className="flex flex-wrap gap-2 items-center">
        {Object.entries(status.assertions.by_materiality).map(([m, n]) => (
          <span key={m} className="flex items-center gap-1 text-xs">
            <Badge level={m} /> {n}
          </span>
        ))}
        <button onClick={onRefresh} className="ml-auto text-xs text-muted-foreground hover:text-foreground flex items-center gap-1">
          <RefreshCw size={12} /> Refresh
        </button>
      </div>
    </div>
  );
}

// ── Section: DB Context Upload ────────────────────────────────────────────────

function ContextUploadPanel({ onDone }: { onDone: () => void }) {
  const [files, setFiles] = useState<File[]>([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleUpload() {
    if (!files.length) return;
    setLoading(true);
    setResult(null);
    const fd = new FormData();
    files.forEach(f => fd.append("files", f));
    try {
      const res = await fetch(`${API}/pipeline/ingest-context`, { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");
      setResult(`Stored ${data.stored} context chunks from ${data.files?.join(", ")}`);
      setFiles([]);
      onDone();
    } catch (e: unknown) {
      setResult(`Error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-3">
      <p className="text-sm text-muted-foreground">
        Upload Excel data dictionaries (.xlsx), SQL DDL files (.sql), or SAS code (.sas/.egp).
        These become the schema mapping layer that lets the SQL generator target your bank&apos;s actual columns.
      </p>
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={e => e.preventDefault()}
        onDrop={e => { e.preventDefault(); setFiles(Array.from(e.dataTransfer.files)); }}
        className="border-2 border-dashed border-border rounded-lg p-6 text-center cursor-pointer hover:border-primary/50 transition-colors"
      >
        <Upload size={20} className="mx-auto mb-2 text-muted-foreground" />
        <p className="text-sm text-muted-foreground">
          {files.length ? files.map(f => f.name).join(", ") : "Drop files here or click to browse"}
        </p>
        <p className="text-xs text-muted-foreground mt-1">.xlsx, .xls, .sql, .sas, .egp</p>
        <input ref={inputRef} type="file" multiple accept=".xlsx,.xls,.sql,.sas,.egp" className="hidden"
          onChange={e => setFiles(Array.from(e.target.files || []))} />
      </div>
      <div className="flex items-center gap-3">
        <button
          onClick={handleUpload}
          disabled={!files.length || loading}
          className="flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground text-sm font-medium disabled:opacity-50"
        >
          {loading ? <Loader2 size={14} className="animate-spin" /> : <Database size={14} />}
          Ingest context
        </button>
        {result && <p className={cn("text-sm", result.startsWith("Error") ? "text-destructive" : "text-green-600")}>{result}</p>}
      </div>
    </div>
  );
}

// ── Section: Assertion Extraction ─────────────────────────────────────────────

function ExtractionPanel({ onDone }: { onDone: () => void }) {
  const [log, setLog] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(false);
  const [model, setModel] = useState("qwen2.5:14b-instruct-q4_K_M");
  const [limit, setLimit] = useState(500);
  const logRef = useRef<HTMLDivElement>(null);

  async function run() {
    setRunning(true);
    setDone(false);
    setLog(["Starting assertion extraction…"]);

    const res = await fetch(`${API}/pipeline/extract-assertions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, limit, backend: "ollama" }),
    });

    const reader = res.body?.getReader();
    const decoder = new TextDecoder();
    if (!reader) return;

    while (true) {
      const { done: d, value } = await reader.read();
      if (d) break;
      const text = decoder.decode(value);
      for (const line of text.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        try {
          const msg = JSON.parse(line.slice(6));
          if (msg.event === "progress") {
            setLog(l => [...l.slice(-100), `[${msg.chunk}/${msg.total}] ${msg.assertions_stored} assertions stored`]);
          } else if (msg.event === "done") {
            setLog(l => [...l, `Done — ${msg.total} assertions extracted`]);
            setDone(true);
            onDone();
          } else if (msg.event === "error") {
            setLog(l => [...l, `Error: ${msg.message}`]);
          } else if (msg.event === "start" || msg.event === "info") {
            setLog(l => [...l, msg.message]);
          }
        } catch {}
      }
      logRef.current?.scrollTo(0, logRef.current.scrollHeight);
    }
    setRunning(false);
  }

  return (
    <div className="space-y-3">
      <p className="text-sm text-muted-foreground">
        Reads every regulation chunk and uses a local Ollama model to extract discrete testable rules
        (PD floors, LGD minimums, time limits, existence checks, etc.) into the assertions table.
      </p>
      <div className="flex gap-3 flex-wrap">
        <div className="flex flex-col gap-1">
          <label className="text-xs text-muted-foreground">Model</label>
          <input value={model} onChange={e => setModel(e.target.value)}
            className="border border-border rounded px-2 py-1 text-sm bg-background w-72" />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-muted-foreground">Max chunks</label>
          <input type="number" value={limit} onChange={e => setLimit(+e.target.value)}
            className="border border-border rounded px-2 py-1 text-sm bg-background w-24" />
        </div>
      </div>
      <button
        onClick={run}
        disabled={running}
        className="flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground text-sm font-medium disabled:opacity-50"
      >
        {running ? <Loader2 size={14} className="animate-spin" /> : <Zap size={14} />}
        {running ? "Extracting…" : "Extract assertions"}
      </button>
      {log.length > 0 && (
        <div ref={logRef} className="bg-muted rounded-md p-3 h-40 overflow-y-auto font-mono text-xs space-y-0.5">
          {log.map((l, i) => <p key={i}>{l}</p>)}
          {done && <p className="text-green-600 font-semibold">✓ Complete</p>}
        </div>
      )}
    </div>
  );
}

// ── Section: SQL Generation ───────────────────────────────────────────────────

function SQLGenPanel({ onDone }: { onDone: () => void }) {
  const [log, setLog] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(false);
  const [model, setModel] = useState("qwen2.5-coder:14b");
  const logRef = useRef<HTMLDivElement>(null);

  async function run() {
    setRunning(true);
    setDone(false);
    setLog(["Generating SQL compliance checks…"]);

    const res = await fetch(`${API}/pipeline/generate-sql`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, backend: "ollama" }),
    });

    const reader = res.body?.getReader();
    const decoder = new TextDecoder();
    if (!reader) return;

    while (true) {
      const { done: d, value } = await reader.read();
      if (d) break;
      const text = decoder.decode(value);
      for (const line of text.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        try {
          const msg = JSON.parse(line.slice(6));
          if (msg.event === "progress") {
            setLog(l => [...l.slice(-100), `[${msg.done}/${msg.total}] SQL generated for ${msg.article}`]);
          } else if (msg.event === "done") {
            setLog(l => [...l, `Done — ${msg.generated}/${msg.total} assertions now have SQL`]);
            setDone(true);
            onDone();
          } else if (msg.event === "error") {
            setLog(l => [...l, `Error: ${msg.message}`]);
          } else if (msg.event === "start" || msg.event === "info") {
            setLog(l => [...l, msg.message]);
          }
        } catch {}
      }
      logRef.current?.scrollTo(0, logRef.current.scrollHeight);
    }
    setRunning(false);
  }

  return (
    <div className="space-y-3">
      <p className="text-sm text-muted-foreground">
        For each assertion without a SQL template, uses <code className="text-xs bg-muted px-1 rounded">qwen2.5-coder</code> to
        generate a PostgreSQL query that returns failing rows. Requires context chunks for schema mapping.
      </p>
      <div className="flex flex-col gap-1">
        <label className="text-xs text-muted-foreground">Model</label>
        <input value={model} onChange={e => setModel(e.target.value)}
          className="border border-border rounded px-2 py-1 text-sm bg-background w-72" />
      </div>
      <button
        onClick={run}
        disabled={running}
        className="flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground text-sm font-medium disabled:opacity-50"
      >
        {running ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
        {running ? "Generating…" : "Generate SQL"}
      </button>
      {log.length > 0 && (
        <div ref={logRef} className="bg-muted rounded-md p-3 h-40 overflow-y-auto font-mono text-xs space-y-0.5">
          {log.map((l, i) => <p key={i}>{l}</p>)}
          {done && <p className="text-green-600 font-semibold">✓ Complete</p>}
        </div>
      )}
    </div>
  );
}

// ── Section: Run Audit ────────────────────────────────────────────────────────

function AuditPanel() {
  const [findings, setFindings] = useState<Finding[]>([]);
  const [log, setLog] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(false);
  const [summary, setSummary] = useState<{ total: number; failures: number; pass_rate: number } | null>(null);
  const [targetDsn, setTargetDsn] = useState("");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  async function run() {
    setRunning(true);
    setDone(false);
    setFindings([]);
    setLog(["Connecting to target database…"]);
    setSummary(null);

    const res = await fetch(`${API}/pipeline/run-assertions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target_dsn: targetDsn || null, limit_assertions: 100 }),
    });

    const reader = res.body?.getReader();
    const decoder = new TextDecoder();
    if (!reader) return;

    while (true) {
      const { done: d, value } = await reader.read();
      if (d) break;
      const text = decoder.decode(value);
      for (const line of text.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        try {
          const msg = JSON.parse(line.slice(6));
          if (msg.event === "finding") {
            setFindings(f => [...f, msg as Finding]);
            setLog(l => [...l.slice(-50), `[${msg.index}/${msg.total}] ${msg.article}: ${msg.status} (${msg.n_failing} failing)`]);
          } else if (msg.event === "done") {
            setSummary({ total: msg.total, failures: msg.failures, pass_rate: msg.pass_rate });
            setDone(true);
          } else if (msg.event === "sql_error") {
            setLog(l => [...l, `SQL error on ${msg.article}: ${msg.error}`]);
          } else if (msg.event === "start" || msg.event === "info") {
            setLog(l => [...l, msg.message]);
          }
        } catch {}
      }
    }
    setRunning(false);
  }

  const failures = findings.filter(f => f.status === "FAIL");
  const passes   = findings.filter(f => f.status === "PASS");

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Executes the SQL assertions against your database and surfaces audit findings.
        Leave the DSN empty to run against the metadata database (for testing).
      </p>
      <div className="flex gap-3 flex-wrap items-end">
        <div className="flex flex-col gap-1 flex-1 max-w-lg">
          <label className="text-xs text-muted-foreground">Target database DSN (optional)</label>
          <input
            placeholder="postgresql://user:pass@host:5432/bank_db"
            value={targetDsn}
            onChange={e => setTargetDsn(e.target.value)}
            className="border border-border rounded px-2 py-1 text-sm bg-background"
          />
        </div>
        <button
          onClick={run}
          disabled={running}
          className="flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground text-sm font-medium disabled:opacity-50"
        >
          {running ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
          {running ? "Running…" : "Run audit"}
        </button>
      </div>

      {/* Progress log */}
      {log.length > 0 && (
        <div className="bg-muted rounded-md p-3 h-28 overflow-y-auto font-mono text-xs space-y-0.5">
          {log.map((l, i) => <p key={i}>{l}</p>)}
        </div>
      )}

      {/* Summary */}
      {summary && (
        <div className="grid grid-cols-3 gap-3">
          <StatCard label="Assertions run" value={summary.total} />
          <StatCard label="Failures" value={summary.failures} />
          <StatCard label="Pass rate" value={`${Math.round(summary.pass_rate * 100)}%`} />
        </div>
      )}

      {/* Findings */}
      {findings.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium flex items-center gap-2">
            <XCircle size={14} className="text-destructive" /> Failures ({failures.length})
          </h4>
          {failures.map(f => (
            <FindingRow key={f.assertion_id} finding={f}
              expanded={expanded.has(f.assertion_id)}
              onToggle={() => setExpanded(s => { const n = new Set(s); n.has(f.assertion_id) ? n.delete(f.assertion_id) : n.add(f.assertion_id); return n; })}
            />
          ))}
          {passes.length > 0 && (
            <>
              <h4 className="text-sm font-medium flex items-center gap-2 pt-2">
                <CheckCircle2 size={14} className="text-green-500" /> Passing ({passes.length})
              </h4>
              {passes.map(f => (
                <FindingRow key={f.assertion_id} finding={f}
                  expanded={expanded.has(f.assertion_id)}
                  onToggle={() => setExpanded(s => { const n = new Set(s); n.has(f.assertion_id) ? n.delete(f.assertion_id) : n.add(f.assertion_id); return n; })}
                />
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}

function FindingRow({ finding: f, expanded, onToggle }: { finding: Finding; expanded: boolean; onToggle: () => void }) {
  return (
    <div className={cn("rounded-lg border text-sm", f.status === "FAIL" ? "border-destructive/30 bg-destructive/5" : "border-green-200 bg-green-50/50")}>
      <button onClick={onToggle} className="w-full flex items-center gap-2 px-3 py-2 text-left">
        {f.status === "FAIL"
          ? <XCircle size={14} className="text-destructive shrink-0" />
          : <CheckCircle2 size={14} className="text-green-500 shrink-0" />}
        <span className="font-medium truncate flex-1">{f.article}</span>
        <Badge level={f.materiality} />
        {f.n_failing > 0 && (
          <span className="text-xs text-destructive font-mono">{f.n_failing} failing</span>
        )}
        {f.downstream_nodes > 0 && (
          <span className="flex items-center gap-0.5 text-xs text-muted-foreground">
            <GitBranch size={11} /> {f.downstream_nodes}
          </span>
        )}
        {expanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
      </button>
      {expanded && (
        <div className="px-3 pb-3 space-y-2 border-t border-border/50 pt-2">
          <p className="text-xs text-muted-foreground">{f.regulation_name}</p>
          <p className="text-sm">{f.rule_text}</p>
          {f.downstream_nodes > 0 && (
            <p className="text-xs text-muted-foreground flex items-center gap-1">
              <GitBranch size={11} />
              This assertion propagates to <strong>{f.downstream_nodes}</strong> downstream graph nodes.
            </p>
          )}
          {f.result_sample.length > 0 && (
            <div className="overflow-x-auto">
              <table className="text-xs w-full border-collapse">
                <thead>
                  <tr className="border-b border-border">
                    {Object.keys(f.result_sample[0]).map(k => (
                      <th key={k} className="text-left px-1.5 py-1 text-muted-foreground font-medium">{k}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {f.result_sample.map((row, i) => (
                    <tr key={i} className="border-b border-border/50">
                      {Object.values(row).map((v, j) => (
                        <td key={j} className="px-1.5 py-1 font-mono">{String(v)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Section: Assertions Table ─────────────────────────────────────────────────

function AssertionsTable() {
  const [assertions, setAssertions] = useState<Assertion[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState<"all" | "critical" | "high" | "no_sql">("all");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  async function load() {
    setLoading(true);
    const params = new URLSearchParams({ limit: "100" });
    if (filter === "no_sql") params.set("has_sql", "false");
    else if (filter !== "all") params.set("materiality", filter);

    const res = await fetch(`${API}/pipeline/assertions?${params}`);
    const data = await res.json();
    setAssertions(data.items || []);
    setTotal(data.total || 0);
    setLoading(false);
  }

  useEffect(() => { load(); }, [filter]);

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 flex-wrap">
        {(["all", "critical", "high", "no_sql"] as const).map(f => (
          <button key={f} onClick={() => setFilter(f)}
            className={cn("text-xs px-2.5 py-1 rounded-full border transition-colors",
              filter === f ? "bg-primary text-primary-foreground border-primary" : "border-border text-muted-foreground hover:text-foreground"
            )}>
            {f === "no_sql" ? "Missing SQL" : f}
          </button>
        ))}
        <span className="text-xs text-muted-foreground ml-auto">{total} total</span>
      </div>
      {loading && <p className="text-sm text-muted-foreground flex items-center gap-2"><Loader2 size={13} className="animate-spin" /> Loading…</p>}
      <div className="space-y-1">
        {assertions.map(a => (
          <div key={a.id} className="rounded-md border border-border text-sm">
            <button onClick={() => setExpanded(s => { const n = new Set(s); n.has(a.id) ? n.delete(a.id) : n.add(a.id); return n; })}
              className="w-full flex items-center gap-2 px-3 py-2 text-left">
              {a.sql_template
                ? <CheckCircle2 size={13} className="text-green-500 shrink-0" />
                : <AlertTriangle size={13} className="text-yellow-500 shrink-0" />}
              <span className="truncate flex-1 text-xs text-muted-foreground">{a.article}</span>
              <span className="truncate flex-1 text-sm">{a.rule_text.slice(0, 80)}{a.rule_text.length > 80 ? "…" : ""}</span>
              <Badge level={a.materiality} />
              {expanded.has(a.id) ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            </button>
            {expanded.has(a.id) && (
              <div className="border-t border-border px-3 pb-3 pt-2 space-y-2">
                <p className="text-xs text-muted-foreground">{a.regulation_name}</p>
                <p className="text-sm">{a.rule_text}</p>
                {a.conditions?.length > 0 && (
                  <p className="text-xs text-muted-foreground">Scope: {a.conditions.join(" AND ")}</p>
                )}
                {a.threshold && (
                  <p className="text-xs font-mono text-blue-600">
                    {a.threshold.field_hint} {a.threshold.op} {a.threshold.value} {a.threshold.unit}
                  </p>
                )}
                {a.sql_template && (
                  <pre className="text-xs bg-muted rounded p-2 overflow-x-auto whitespace-pre-wrap">{a.sql_template}</pre>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Section wrapper ───────────────────────────────────────────────────────────

function Section({ title, icon, children, defaultOpen = false }:
  { title: string; icon: React.ReactNode; children: React.ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-lg border border-border bg-card">
      <button onClick={() => setOpen(o => !o)} className="w-full flex items-center gap-2 px-4 py-3 text-left">
        {icon}
        <span className="font-medium text-sm">{title}</span>
        <span className="ml-auto">{open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}</span>
      </button>
      {open && <div className="px-4 pb-4 border-t border-border pt-3">{children}</div>}
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function PipelinePage() {
  const [status, setStatus] = useState<StatusData | null>(null);

  async function loadStatus() {
    try {
      const res = await fetch(`${API}/pipeline/status`);
      setStatus(await res.json());
    } catch {}
  }

  useEffect(() => { loadStatus(); }, []);

  return (
    <div className="max-w-4xl mx-auto px-6 py-8 space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Audit Pipeline</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Step-by-step: ingest schema context → extract assertions from regulations → generate SQL checks → run audit findings.
        </p>
      </div>

      <StatusPanel status={status} onRefresh={loadStatus} />

      <div className="space-y-3">
        <Section title="1. Ingest DB schema context" icon={<Database size={15} />} defaultOpen>
          <ContextUploadPanel onDone={loadStatus} />
        </Section>

        <Section title="2. Extract assertions from corpus" icon={<Zap size={15} />}>
          <ExtractionPanel onDone={loadStatus} />
        </Section>

        <Section title="3. Generate SQL compliance checks" icon={<Play size={15} />}>
          <SQLGenPanel onDone={loadStatus} />
        </Section>

        <Section title="4. Run audit" icon={<AlertTriangle size={15} />}>
          <AuditPanel />
        </Section>

        <Section title="Assertions library" icon={<CheckCircle2 size={15} />}>
          <AssertionsTable />
        </Section>

        <Section title="Knowledge graph" icon={<GitBranch size={15} />}>
          <GraphPanel />
        </Section>
      </div>
    </div>
  );
}

// ── Graph Panel ───────────────────────────────────────────────────────────────

function GraphPanel() {
  const [stats, setStats] = useState<{ nodes: number; edges: number; byType: Record<string, number> } | null>(null);
  const [loading, setLoading] = useState(false);

  async function load() {
    setLoading(true);
    try {
      const res = await fetch(`${API}/pipeline/graph?limit=1000`);
      const data = await res.json();
      const byType: Record<string, number> = {};
      for (const n of data.nodes) byType[n.type] = (byType[n.type] || 0) + 1;
      setStats({ nodes: data.nodes.length, edges: data.edges.length, byType });
    } catch {}
    setLoading(false);
  }

  useEffect(() => { load(); }, []);

  return (
    <div className="space-y-3">
      <p className="text-sm text-muted-foreground">
        The knowledge graph links regulation articles → assertions → risk parameters → DB columns → capital metrics.
        When an assertion fails, graph traversal shows all downstream nodes affected — this is how materiality is scored.
      </p>
      {loading && <p className="text-sm text-muted-foreground flex items-center gap-2"><Loader2 size={13} className="animate-spin" /> Loading graph…</p>}
      {stats && (
        <div className="space-y-2">
          <div className="grid grid-cols-2 gap-3">
            <StatCard label="Nodes" value={stats.nodes} />
            <StatCard label="Edges" value={stats.edges} />
          </div>
          <div className="flex flex-wrap gap-2">
            {Object.entries(stats.byType).map(([type, count]) => (
              <span key={type} className="text-xs bg-muted border border-border rounded px-2 py-0.5">
                {type}: {count}
              </span>
            ))}
          </div>
        </div>
      )}
      <button onClick={load} className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1">
        <RefreshCw size={12} /> Refresh graph stats
      </button>
      <p className="text-xs text-muted-foreground italic">
        Interactive graph visualization (D3/force-directed) — planned for next iteration.
      </p>
    </div>
  );
}
