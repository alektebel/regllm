"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Play,
  Loader2,
  RotateCw,
  Search,
  Sparkles,
  GitBranch,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  BarChart3,
  Network,
  Wand2,
  SlidersHorizontal,
} from "lucide-react";
import { cn } from "@/lib/utils";
import LineageGraph, {
  type LineageEdge,
  type LineageNode,
  type NodeOverlay,
} from "@/components/diff/LineageGraph";
import AskAgent, { type Citation } from "@/components/diff/AskAgent";
import SasEditor, { type CodeDiffSummary } from "@/components/diff/SasEditor";
import CodeVsDataDelta, { type CodeVsData } from "@/components/diff/CodeVsDataDelta";
import LineageTracePanel, { type LineageTrace } from "@/components/diff/LineageTracePanel";

const API = "/api";

// ── Types ────────────────────────────────────────────────────────────────────

interface RowPair {
  pk: string;
  v2: Record<string, unknown>;
  v3: Record<string, unknown>;
  target_v2: unknown;
  target_v3: unknown;
}

interface Attribution {
  field: string;
  delta_input: number | [unknown, unknown];
  contribution: number;
  is_numeric?: boolean;
  avg_gradient?: number;
}

interface BranchFlip {
  condition: string;
  data_step: string;
  kind: string;
  v2_taken: boolean;
  v3_taken: boolean;
  vars_in_condition: string[];
}

interface GradientReport {
  attributions: Attribution[];
  branch_flips: BranchFlip[];
  unexplained_residual: number;
  steps: number;
}

interface ShapleyReport {
  attributions: Attribution[];
  unexplained_residual: number;
  method: string;
  n_evaluations: number;
}

interface FieldJustification {
  field: string;
  justified: boolean;
  confidence: number;
  rationale: string;
  evidence: { document?: string; heading?: string; quote?: string; snippet?: string }[];
}

interface GraphRAGReport {
  overall: string;
  per_field: FieldJustification[];
  backend: string;
  model: string;
}

interface DiscrepancyReport {
  target: string;
  y_v2: number | null;
  y_v3: number | null;
  delta_y: number | null;
  excluded_v2: boolean;
  excluded_v3: boolean;
  field_deltas: { field: string; v2: unknown; v3: unknown; is_numeric: boolean }[];
  branch_flips: BranchFlip[];
  gradient: GradientReport | null;
  shapley: ShapleyReport | null;
  suspects: string[];
  graphrag?: GraphRAGReport | null;
  code_differs?: boolean;
  code_diff?: CodeDiffSummary | null;
  code_vs_data?: CodeVsData | null;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

const NUMERIC_FIELDS = new Set([
  "PD_ESTIMADA", "LGD_ESTIMADA", "EAD", "DPDS", "STAGE_IFRS9", "ECL",
  "PROVISION_PERIOD_MONTHS", "VENTANA_OBSERVACION_YEARS",
  "VENTANA_CALIBRACION_YEARS", "RATING_GRADO", "LGD_FLOOR_APLICADO",
  "MOC", "CURE_FLAG", "RWA",
]);

function fmt(v: unknown, digits = 6): string {
  if (v === null || v === undefined || v === "") return "—";
  if (typeof v === "number") {
    if (Number.isInteger(v)) return v.toString();
    return v.toFixed(digits).replace(/\.?0+$/, "");
  }
  return String(v);
}

function fmtSigned(v: number, digits = 4): string {
  const sign = v >= 0 ? "+" : "";
  return sign + v.toFixed(digits).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
}

function rowDiffFields(v2: Record<string, unknown>, v3: Record<string, unknown>): string[] {
  const out: string[] = [];
  const keys = new Set([...Object.keys(v2), ...Object.keys(v3)]);
  for (const k of Array.from(keys).sort()) {
    if (v2[k] !== v3[k]) out.push(k);
  }
  return out;
}

// ── Page ─────────────────────────────────────────────────────────────────────

type ViewMode = "graph" | "waterfall";
type Mode = "manual" | "ask";

export default function DiffPage() {
  // V3 SAS is the *primary* code: drives the lineage graph and the
  // gradient/Shapley attribution. V2 SAS is used for the V2-vs-V3 diff
  // and the code-vs-data Δ decomposition.
  const [sasCode, setSasCode] = useState<string>("");        // alias for V3
  const [sasV2, setSasV2] = useState<string>("");
  const [pairs, setPairs] = useState<RowPair[]>([]);
  const [target, setTarget] = useState<string>("ECL");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<RowPair | null>(null);
  const [report, setReport] = useState<DiscrepancyReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [useLLM, setUseLLM] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [llmBackend, setLlmBackend] = useState<string>("");
  const [llmModel, setLlmModel] = useState<string>("");
  const [view, setView] = useState<ViewMode>("graph");
  const [mode, setMode] = useState<Mode>("manual");
  const [agentHighlight, setAgentHighlight] = useState<string[]>([]);
  const [agentCitations, setAgentCitations] = useState<Citation[]>([]);

  // Lineage graph data — fetched whenever SAS code changes
  const [lineage, setLineage] = useState<{ nodes: LineageNode[]; edges: LineageEdge[] }>({
    nodes: [],
    edges: [],
  });
  const [lineageTrace, setLineageTrace] = useState<LineageTrace | null>(null);
  const [traceRoot, setTraceRoot] = useState<string>("ECL");
  const [ancestorsOnly, setAncestorsOnly] = useState(true);
  const [allLineageFields, setAllLineageFields] = useState<string[]>([]);

  // Bootstrap: load V2 + V3 SAS pair, LLM status
  useEffect(() => {
    fetch(`${API}/diff/sas-pair`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.statusText)))
      .then((d) => {
        setSasCode(d.v3 || "");
        setSasV2(d.v2 || "");
      })
      .catch(() => {
        // Fallback to single sample
        fetch(`${API}/sas/sample`)
          .then((r) => r.json())
          .then((d) => {
            setSasCode(d.code || "");
            setSasV2(d.code || "");
          })
          .catch(() => setError("Could not load sample SAS"));
      });
    fetch(`${API}/kb/llm-status`)
      .then((r) => r.json())
      .then((d) => {
        setLlmBackend(d.backend || "stub");
        setLlmModel(d.model || "");
      })
      .catch(() => setLlmBackend("stub"));
  }, []);

  // Load paired rows whenever the target changes
  useEffect(() => {
    fetch(`${API}/diff/sample-rows?target=${encodeURIComponent(target)}&only_changed=true&limit=200`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.statusText)))
      .then((d) => {
        setPairs(d.items || []);
        if (d.items && d.items.length > 0) setSelected(d.items[0]);
      })
      .catch((e) => setError(`Could not load paired rows: ${e}`));
  }, [target]);

  // Keep trace root in sync when the discrepancy target changes
  useEffect(() => {
    setTraceRoot(target);
  }, [target]);

  // Fetch full field list (for target dropdown) when SAS code changes
  const fieldsDebounce = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (!sasCode) return;
    if (fieldsDebounce.current) clearTimeout(fieldsDebounce.current);
    fieldsDebounce.current = setTimeout(() => {
      fetch(`${API}/sas/lineage`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: sasCode }),
      })
        .then((r) => r.json())
        .then((d) => {
          const ids = (d.nodes || []).map((n: LineageNode) => n.id.toUpperCase());
          setAllLineageFields(ids);
        })
        .catch(() => {});
    }, 350);
    return () => {
      if (fieldsDebounce.current) clearTimeout(fieldsDebounce.current);
    };
  }, [sasCode]);

  // Fetch backward trace subgraph for the active trace root
  const lineageDebounce = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (!sasCode || !traceRoot) return;
    if (lineageDebounce.current) clearTimeout(lineageDebounce.current);
    lineageDebounce.current = setTimeout(() => {
      fetch(`${API}/sas/lineage`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          code: sasCode,
          target: traceRoot,
          ancestors_only: ancestorsOnly,
        }),
      })
        .then((r) => r.json())
        .then((d) => {
          if (ancestorsOnly) {
            setLineage({ nodes: d.nodes || [], edges: d.edges || [] });
            setLineageTrace(d as LineageTrace);
          } else {
            setLineage({ nodes: d.nodes || [], edges: d.edges || [] });
            setLineageTrace((d.trace as LineageTrace) ?? null);
          }
        })
        .catch(() => {});
    }, 350);
    return () => {
      if (lineageDebounce.current) clearTimeout(lineageDebounce.current);
    };
  }, [sasCode, traceRoot, ancestorsOnly]);

  const filteredPairs = useMemo(() => {
    if (!search) return pairs;
    const q = search.toUpperCase();
    return pairs.filter((p) => p.pk?.toUpperCase().includes(q));
  }, [pairs, search]);

  const fieldOptions = useMemo(() => {
    const merged = new Set([...NUMERIC_FIELDS, ...allLineageFields]);
    return Array.from(merged).sort();
  }, [allLineageFields]);

  const traceDepthMap = useMemo(
    () => lineageTrace?.depth ?? {},
    [lineageTrace],
  );

  // ── Compose graph overlays from selection + report ────────────────────────

  const overlays = useMemo<Record<string, NodeOverlay>>(() => {
    const out: Record<string, NodeOverlay> = {};
    if (selected) {
      const v2 = selected.v2 || {};
      const v3 = selected.v3 || {};
      const allKeys = Array.from(new Set([...Object.keys(v2), ...Object.keys(v3)]));
      for (const k of allKeys) {
        const ku = k.toUpperCase();
        const a = v2[k];
        const b = v3[k];
        const numA = typeof a === "number" || (typeof a === "string" && !isNaN(Number(a)));
        const numB = typeof b === "number" || (typeof b === "string" && !isNaN(Number(b)));
        const isNumeric = numA && numB;
        const delta = isNumeric ? Number(b) - Number(a) : undefined;
        out[ku] = { v2: a, v3: b, isNumeric, delta };
      }
    }
    // Agent-suggested highlight (always applied regardless of selection)
    for (const f of agentHighlight) {
      const fu = f.toUpperCase();
      out[fu] = { ...(out[fu] || {}), agentHighlight: true };
    }
    // Code-change overlay: every field touched by a modified/added/removed
    // data step in the V2-vs-V3 SAS diff gets a fuchsia ring.
    if (report?.code_diff) {
      const cd = report.code_diff;
      const touched = new Set<string>();
      const collect = (writes?: string[], reads?: string[]) => {
        for (const w of writes ?? []) touched.add(String(w).toUpperCase());
        for (const r of reads ?? []) touched.add(String(r).toUpperCase());
      };
      for (const m of cd.modified_steps ?? []) {
        collect(m.writes_v2, m.reads_v2);
        collect(m.writes_v3, m.reads_v3);
      }
      for (const a of cd.added_steps ?? []) {
        collect((a as { writes?: string[] }).writes, (a as { reads?: string[] }).reads);
      }
      for (const r of cd.removed_steps ?? []) {
        collect((r as { writes?: string[] }).writes, (r as { reads?: string[] }).reads);
      }
      for (const fu of Array.from(touched)) {
        out[fu] = { ...(out[fu] || {}), codeChanged: true };
      }
    }
    if (report) {
      // Merge in attributions
      for (const a of report.gradient?.attributions ?? []) {
        const f = a.field.toUpperCase();
        out[f] = { ...(out[f] || {}), contributionGrad: a.contribution };
      }
      for (const a of report.shapley?.attributions ?? []) {
        const f = a.field.toUpperCase();
        out[f] = { ...(out[f] || {}), contributionShap: a.contribution };
      }
      // Mark target
      const t = (report.target || target).toUpperCase();
      out[t] = { ...(out[t] || {}), isTarget: true };
      // Excluded? mark all input nodes as excluded for visual hint
      if (report.excluded_v2 || report.excluded_v3) {
        // don't blanket-mark; just mark the target as excluded
        out[t] = { ...(out[t] || {}), excluded: true };
      }
    } else if (selected) {
      const t = target.toUpperCase();
      out[t] = { ...(out[t] || {}), isTarget: true };
    }
    return out;
  }, [selected, report, target, agentHighlight]);

  const branchFlips = useMemo(() => {
    if (!report) return [];
    return report.branch_flips
      .filter((b) => b.v2_taken !== b.v3_taken)
      .map((b) => ({
        data_step: b.data_step,
        vars_in_condition: b.vars_in_condition || [],
      }));
  }, [report]);

  // ── Run analysis ──────────────────────────────────────────────────────────

  async function runAnalysis() {
    if (!selected) return;
    setLoading(true);
    setError(null);
    setReport(null);
    try {
      const r = await fetch(`${API}/diff/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sas_v3: sasCode,
          sas_v2: sasV2 || sasCode,
          row_v2: selected.v2,
          row_v3: selected.v3,
          target_field: target,
          method: "both",
          grad_steps: 32,
          shapley_n_samples: 256,
          use_gemma: useLLM,
          top_k_for_llm: 5,
          include_code_diff: true,
          include_code_vs_data: true,
        }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = (await r.json()) as DiscrepancyReport;
      setReport(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  // ── Layout: 5/7 split (controls / visualization) ──────────────────────────

  return (
    <div className="h-screen flex flex-col bg-background text-foreground overflow-hidden">
      <Header llmBackend={llmBackend} llmModel={llmModel} />

      <div className="flex-1 grid grid-cols-12 gap-3 p-3 min-h-0 overflow-hidden">
        {/* ── Left rail (5/12): controls + code + row picker ──────────────── */}
        <aside className="col-span-5 flex flex-col gap-3 min-h-0 overflow-hidden">
          <SasEditor
            v3={sasCode}
            v2={sasV2}
            onV3Change={setSasCode}
            onV2Change={setSasV2}
            codeDiff={report?.code_diff ?? null}
            fieldsCount={lineage.nodes.length}
            edgesCount={lineage.edges.length}
          />

          <div className="rounded-lg border border-border bg-card p-3">
            <div className="flex items-center gap-1 bg-muted/40 rounded p-0.5 mb-3 w-fit">
              <ViewButton
                active={mode === "manual"}
                onClick={() => setMode("manual")}
                icon={<SlidersHorizontal className="h-3.5 w-3.5" />}
                label="Manual"
              />
              <ViewButton
                active={mode === "ask"}
                onClick={() => setMode("ask")}
                icon={<Wand2 className="h-3.5 w-3.5" />}
                label="Ask"
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-muted-foreground">Target field</label>
                <select
                  value={target}
                  onChange={(e) => setTarget(e.target.value)}
                  className="w-full bg-muted border border-border rounded px-2 py-1.5 text-sm mt-1"
                >
                  {fieldOptions.map((t) => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </div>
              <div className="flex flex-col">
                <label className="text-xs text-muted-foreground">LLM (GraphRAG)</label>
                <div className="mt-1 px-2 py-1.5 bg-muted/30 border border-border rounded text-xs font-mono flex items-center justify-between gap-2">
                  <span className="truncate" title={llmModel || llmBackend}>
                    {llmBackend === "stub" || !llmBackend
                      ? "stub"
                      : (llmModel ? llmModel.split(":")[0] : llmBackend)}
                  </span>
                  <label className="flex items-center gap-1 text-[11px] shrink-0">
                    <input
                      type="checkbox"
                      checked={useLLM}
                      onChange={(e) => setUseLLM(e.target.checked)}
                    />
                    use
                  </label>
                </div>
              </div>
            </div>
            {mode === "manual" && (
              <button
                onClick={runAnalysis}
                disabled={!selected || loading}
                className="mt-3 w-full inline-flex items-center justify-center gap-2 bg-primary text-primary-foreground rounded px-3 py-2 text-sm hover:bg-primary/90 disabled:opacity-50"
              >
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                Explain Δ {target} for {selected?.pk ?? "(none)"}
              </button>
            )}
          </div>

          {mode === "manual" ? (
            <div className="rounded-lg border border-border bg-card p-3 flex-1 min-h-0 flex flex-col">
              <div className="flex items-center gap-2 mb-2">
                <Search className="h-4 w-4 text-muted-foreground" />
                <input
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Filter by primary key…"
                  className="flex-1 bg-muted/30 border border-border rounded px-2 py-1 text-xs"
                />
                <span className="text-xs text-muted-foreground">{filteredPairs.length}</span>
              </div>
              <div className="flex-1 overflow-y-auto border border-border rounded">
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-muted">
                    <tr>
                      <th className="text-left px-2 py-1 font-medium">PK</th>
                      <th className="text-right px-2 py-1 font-medium">V2 {target}</th>
                      <th className="text-right px-2 py-1 font-medium">V3 {target}</th>
                      <th className="text-right px-2 py-1 font-medium">Δ</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredPairs.map((p) => {
                      const isSelected = selected?.pk === p.pk;
                      const v2 = typeof p.target_v2 === "number" ? p.target_v2 : Number(p.target_v2);
                      const v3 = typeof p.target_v3 === "number" ? p.target_v3 : Number(p.target_v3);
                      const delta = Number.isFinite(v3 - v2) ? v3 - v2 : null;
                      return (
                        <tr
                          key={p.pk}
                          onClick={() => { setSelected(p); setReport(null); }}
                          className={cn(
                            "cursor-pointer hover:bg-muted/40",
                            isSelected && "bg-primary/10",
                          )}
                        >
                          <td className="px-2 py-1 font-mono">{p.pk}</td>
                          <td className="px-2 py-1 text-right tabular-nums">{fmt(p.target_v2)}</td>
                          <td className="px-2 py-1 text-right tabular-nums">{fmt(p.target_v3)}</td>
                          <td className={cn(
                            "px-2 py-1 text-right tabular-nums",
                            delta !== null && delta > 0 && "text-amber-400",
                            delta !== null && delta < 0 && "text-sky-400",
                          )}>
                            {delta !== null ? fmtSigned(delta, 2) : "—"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="rounded-lg border border-border bg-card p-3 flex-1 min-h-0">
              <AskAgent
                onLineageHighlight={setAgentHighlight}
                onCitations={setAgentCitations}
                initialQuestion={
                  selected
                    ? `Why does ${selected.pk} have a different ${target} between V2 and V3?`
                    : ""
                }
                className="h-full"
              />
            </div>
          )}
        </aside>

        {/* ── Right pane (7/12): graph + side panels ─────────────────────── */}
        <section className="col-span-7 flex flex-col gap-3 min-h-0 overflow-y-auto">
          {error && (
            <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive flex items-center gap-2">
              <AlertTriangle className="h-4 w-4" /> {error}
              <button className="ml-auto" onClick={() => setError(null)} aria-label="dismiss">
                <RotateCw className="h-3.5 w-3.5" />
              </button>
            </div>
          )}

          {report && <DeltaSummary report={report} />}
          {report?.code_vs_data && (
            <CodeVsDataDelta cvd={report.code_vs_data} target={target} />
          )}

          <div className="rounded-lg border border-border bg-card p-2 flex flex-col h-[640px] min-h-[420px]">
            <div className="flex items-center justify-between px-2 py-1 gap-2 flex-wrap">
              <h2 className="text-sm font-semibold flex items-center gap-2">
                <Network className="h-4 w-4 text-primary" /> Lineage graph
                {traceRoot !== target && (
                  <span className="text-[10px] font-mono text-muted-foreground">
                    tracing <span className="text-sky-300">{traceRoot}</span>
                  </span>
                )}
              </h2>
              <div className="flex items-center gap-2">
                <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                  <input
                    type="checkbox"
                    checked={ancestorsOnly}
                    onChange={(e) => setAncestorsOnly(e.target.checked)}
                  />
                  ancestors only
                </label>
                <div className="flex items-center gap-1 bg-muted/40 rounded p-0.5">
                  <ViewButton
                    active={view === "graph"}
                    onClick={() => setView("graph")}
                    icon={<Network className="h-3.5 w-3.5" />}
                    label="Graph"
                  />
                  <ViewButton
                    active={view === "waterfall"}
                    onClick={() => setView("waterfall")}
                    icon={<BarChart3 className="h-3.5 w-3.5" />}
                    label="Waterfall"
                  />
                </div>
              </div>
            </div>

            <div className="relative flex-1 min-h-0">
              {view === "graph" ? (
                lineage.nodes.length > 0 ? (
                  <ResizableGraph
                    nodes={lineage.nodes}
                    edges={lineage.edges}
                    overlays={overlays}
                    branchFlips={branchFlips}
                    target={traceRoot}
                    depthMap={traceDepthMap}
                    onTraceTarget={setTraceRoot}
                    resetSeed={`${selected?.pk ?? ""}-${traceRoot}-${ancestorsOnly}`}
                  />
                ) : (
                  <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
                    Loading lineage…
                  </div>
                )
              ) : report ? (
                <div className="p-2 overflow-y-auto h-full">
                  <Waterfall report={report} />
                </div>
              ) : (
                <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
                  Run the analyzer to see attributions.
                </div>
              )}
            </div>
          </div>

          <LineageTracePanel
            trace={lineageTrace}
            activeTarget={traceRoot}
            onTraceField={setTraceRoot}
          />

          {report?.code_diff && (
            <CodeChangesPanel codeDiff={report.code_diff} target={target} />
          )}
          {report && report.branch_flips && report.branch_flips.length > 0 && (
            <BranchFlipsPanel flips={report.branch_flips} />
          )}
          {report?.graphrag && <GemmaPanel rep={report.graphrag} />}
        </section>
      </div>
    </div>
  );
}

// ── Resizable graph wrapper ─────────────────────────────────────────────────

function ResizableGraph(
  props: React.ComponentProps<typeof LineageGraph>,
) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState({ w: 800, h: 480 });
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(() => {
      const rect = ref.current!.getBoundingClientRect();
      setSize({ w: Math.max(400, rect.width), h: Math.max(300, rect.height) });
    });
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return (
    <div ref={ref} className="absolute inset-0">
      <LineageGraph {...props} width={size.w} height={size.h} />
    </div>
  );
}

// ── Sub-components ───────────────────────────────────────────────────────────

function ViewButton({
  active, onClick, icon, label,
}: {
  active: boolean; onClick: () => void; icon: React.ReactNode; label: string;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "px-2 py-1 rounded text-xs flex items-center gap-1 transition-colors",
        active ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground",
      )}
    >
      {icon}
      {label}
    </button>
  );
}

function Header({ llmBackend, llmModel }: { llmBackend: string; llmModel: string }) {
  const dotColor = llmBackend === "stub" || !llmBackend
    ? "bg-muted-foreground/40"
    : "bg-emerald-400";
  return (
    <header className="flex items-center justify-between px-4 py-3 border-b border-border bg-card/60">
      <div className="flex items-center gap-3">
        <Sparkles className="h-5 w-5 text-primary" />
        <div>
          <h1 className="text-base font-semibold">SAS Diff Explainer</h1>
          <p className="text-xs text-muted-foreground">
            Path-integrated gradients · Shapley values · Change-log GraphRAG
          </p>
        </div>
      </div>
      <div className="text-xs text-muted-foreground flex items-center gap-2">
        <span className={cn("inline-block w-2 h-2 rounded-full", dotColor)} />
        <span>local LLM:</span>
        <span className="font-mono text-foreground/90">
          {llmBackend || "?"}
          {llmModel && llmModel !== "stub" ? ` · ${llmModel}` : ""}
        </span>
      </div>
    </header>
  );
}

function DeltaSummary({ report }: { report: DiscrepancyReport }) {
  const dy = report.delta_y;
  const sign = dy === null || dy === undefined ? "" : dy >= 0 ? "+" : "";
  return (
    <div className="rounded-lg border border-border bg-card p-3 flex items-center gap-6">
      <div>
        <div className="text-xs text-muted-foreground">Δ {report.target}</div>
        <div className="text-2xl font-mono font-semibold">
          {dy === null || dy === undefined ? "—" : `${sign}${dy.toFixed(4)}`}
        </div>
      </div>
      <div className="flex-1 grid grid-cols-2 gap-4 text-xs text-muted-foreground">
        <div>V2 = <span className="font-mono text-sky-300">{fmt(report.y_v2)}</span></div>
        <div>V3 = <span className="font-mono text-amber-300">{fmt(report.y_v3)}</span></div>
        {report.gradient && (
          <div>residual<sub>grad</sub> = <span className="font-mono">{fmtSigned(report.gradient.unexplained_residual, 4)}</span></div>
        )}
        {report.shapley && (
          <div>residual<sub>shap</sub> = <span className="font-mono">{fmtSigned(report.shapley.unexplained_residual, 4)}</span></div>
        )}
      </div>
      {(report.excluded_v2 || report.excluded_v3) && (
        <div className="text-xs text-amber-400 flex items-center gap-1">
          <AlertTriangle className="h-3 w-3" />
          row excluded by WHERE filter
          {report.excluded_v2 && " in V2"}
          {report.excluded_v3 && " in V3"}
        </div>
      )}
    </div>
  );
}

function Waterfall({ report }: { report: DiscrepancyReport }) {
  const grad = report.gradient?.attributions ?? [];
  const shap = report.shapley?.attributions ?? [];
  const fields = Array.from(
    new Set([...grad.map((a) => a.field), ...shap.map((a) => a.field)]),
  );
  const byField = (arr: Attribution[]) => Object.fromEntries(arr.map((a) => [a.field, a.contribution]));
  const g = byField(grad);
  const s = byField(shap);
  const combined = fields
    .map((f) => ({
      field: f,
      grad: g[f] ?? 0,
      shap: s[f] ?? 0,
      score: Math.abs(g[f] ?? 0) + Math.abs(s[f] ?? 0),
    }))
    .sort((a, b) => b.score - a.score);
  const max = Math.max(...combined.map((c) => Math.max(Math.abs(c.grad), Math.abs(c.shap))), 1e-12);
  return (
    <div className="space-y-1.5">
      <div className="grid grid-cols-12 text-[10px] text-muted-foreground uppercase">
        <span className="col-span-4">Field</span>
        <span className="col-span-4 text-right">Gradient (path-integrated)</span>
        <span className="col-span-4 text-right">Shapley</span>
      </div>
      {combined.map((c) => (
        <div key={c.field} className="grid grid-cols-12 items-center gap-2 text-xs">
          <span className="col-span-4 font-mono truncate">{c.field}</span>
          <Bar value={c.grad} max={max} className="col-span-4" />
          <Bar value={c.shap} max={max} className="col-span-4" tone="shapley" />
        </div>
      ))}
    </div>
  );
}

function Bar({ value, max, className, tone }: { value: number; max: number; className?: string; tone?: "shapley" }) {
  const pct = Math.min(100, (Math.abs(value) / max) * 100);
  const positive = value >= 0;
  const color = tone === "shapley"
    ? (positive ? "bg-emerald-500/70" : "bg-rose-500/70")
    : (positive ? "bg-amber-500/70" : "bg-sky-500/70");
  return (
    <div className={cn("flex items-center gap-2", className)}>
      <div className="flex-1 h-2 bg-muted/40 rounded relative overflow-hidden">
        <div
          className={cn("absolute h-full top-0 rounded", color)}
          style={{
            width: `${pct / 2}%`,
            left: positive ? "50%" : `${50 - pct / 2}%`,
          }}
        />
        <div className="absolute left-1/2 top-0 h-full border-l border-border" />
      </div>
      <span className="font-mono tabular-nums w-20 text-right">{fmtSigned(value, 4)}</span>
    </div>
  );
}

function CodeChangesPanel({ codeDiff, target }: { codeDiff: CodeDiffSummary; target: string }) {
  const added = codeDiff.added_steps ?? [];
  const removed = codeDiff.removed_steps ?? [];
  const modified = codeDiff.modified_steps ?? [];
  const total = added.length + removed.length + modified.length;
  if (total === 0) return null;
  return (
    <div className="rounded-lg border border-fuchsia-500/30 bg-fuchsia-500/5 p-3">
      <h2 className="text-sm font-semibold mb-2 flex items-center gap-2 text-fuchsia-300">
        <GitBranch className="h-4 w-4" /> Code changes affecting <span className="font-mono text-fuchsia-200">{target}</span> ({total})
      </h2>
      <p className="text-[11px] text-muted-foreground mb-2">
        Data steps in the V2 vs V3 SAS pipeline that produce or read fields
        on <span className="font-mono">{target}</span>'s lineage. Anything
        else is filtered out.
      </p>
      <div className="space-y-1.5 text-xs">
        {modified.map((m) => (
          <details key={`m-${m.output}`} className="border border-amber-500/30 rounded bg-amber-500/5">
            <summary className="cursor-pointer select-none px-2 py-1.5 flex items-center gap-2 hover:bg-amber-500/10">
              <span className="text-amber-400 font-mono text-[10px]">MOD</span>
              <span className="font-mono">{m.output}</span>
              {m.writes_added && m.writes_added.length > 0 && (
                <span className="text-emerald-400 text-[10px] font-mono">+{m.writes_added.join(", ")}</span>
              )}
              {m.writes_removed && m.writes_removed.length > 0 && (
                <span className="text-rose-400 text-[10px] font-mono">−{m.writes_removed.join(", ")}</span>
              )}
            </summary>
            <div className="grid grid-cols-2 gap-2 p-2 text-[11px]">
              <div>
                <div className="text-rose-400 text-[9px] uppercase tracking-wider mb-0.5">V2</div>
                <pre className="whitespace-pre-wrap break-words font-mono text-[10.5px] bg-rose-500/5 border border-rose-500/20 rounded p-1.5 max-h-40 overflow-y-auto">{m.body_v2 ?? "—"}</pre>
              </div>
              <div>
                <div className="text-emerald-400 text-[9px] uppercase tracking-wider mb-0.5">V3</div>
                <pre className="whitespace-pre-wrap break-words font-mono text-[10.5px] bg-emerald-500/5 border border-emerald-500/20 rounded p-1.5 max-h-40 overflow-y-auto">{m.body_v3 ?? "—"}</pre>
              </div>
            </div>
          </details>
        ))}
        {added.map((s) => (
          <div key={`a-${s.output}`} className="px-2 py-1.5 border border-emerald-500/30 rounded bg-emerald-500/5 flex items-center gap-2">
            <span className="text-emerald-400 font-mono text-[10px]">ADD</span>
            <span className="font-mono">{s.output}</span>
            <span className="text-[10px] text-muted-foreground">new in V3</span>
          </div>
        ))}
        {removed.map((s) => (
          <div key={`r-${s.output}`} className="px-2 py-1.5 border border-rose-500/30 rounded bg-rose-500/5 flex items-center gap-2">
            <span className="text-rose-400 font-mono text-[10px]">DEL</span>
            <span className="font-mono">{s.output}</span>
            <span className="text-[10px] text-muted-foreground">removed in V3</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function BranchFlipsPanel({ flips }: { flips: BranchFlip[] }) {
  return (
    <div className="rounded-lg border border-amber-500/40 bg-amber-500/5 p-3">
      <h2 className="text-sm font-semibold mb-2 flex items-center gap-2 text-amber-400">
        <GitBranch className="h-4 w-4" /> Branch flips ({flips.length})
      </h2>
      <ul className="space-y-1.5 text-xs">
        {flips.map((f, i) => (
          <li key={i} className="font-mono">
            <span className="text-muted-foreground">[{f.kind} · {f.data_step}]</span>{" "}
            <code>{f.condition}</code>{" "}
            <span className={cn(f.v2_taken ? "text-emerald-400" : "text-rose-400")}>
              V2:{f.v2_taken ? "T" : "F"}
            </span>{" → "}
            <span className={cn(f.v3_taken ? "text-emerald-400" : "text-rose-400")}>
              V3:{f.v3_taken ? "T" : "F"}
            </span>
            {f.vars_in_condition.length > 0 && (
              <span className="text-muted-foreground"> · vars: {f.vars_in_condition.join(", ")}</span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

function GemmaPanel({ rep }: { rep: GraphRAGReport }) {
  const modelLabel = rep.model && rep.model !== "stub"
    ? rep.model
    : "no model";
  return (
    <div className="rounded-lg border border-primary/40 bg-primary/5 p-3">
      <h2 className="text-sm font-semibold mb-2 flex items-center gap-2 text-primary">
        <Sparkles className="h-4 w-4" /> Local LLM + GraphRAG
        <span className="ml-auto text-[11px] text-muted-foreground font-mono">
          {rep.backend} · {modelLabel}
        </span>
      </h2>
      <p className="text-xs mb-3">{rep.overall}</p>
      <ul className="space-y-2">
        {rep.per_field.map((f) => (
          <li key={f.field} className="border border-border/50 rounded p-2 bg-card/40">
            <div className="flex items-center gap-2 text-xs">
              {f.justified ? (
                <CheckCircle2 className="h-4 w-4 text-emerald-400" />
              ) : (
                <XCircle className="h-4 w-4 text-rose-400" />
              )}
              <span className="font-mono">{f.field}</span>
              <span className="ml-auto text-muted-foreground">
                conf {f.confidence.toFixed(2)}
              </span>
            </div>
            {f.rationale && (
              <p className="text-xs mt-1 text-muted-foreground">{f.rationale}</p>
            )}
            {f.evidence && f.evidence.length > 0 && (
              <details className="mt-1 text-[11px]">
                <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                  {f.evidence.length} citation{f.evidence.length === 1 ? "" : "s"}
                </summary>
                <ul className="mt-1 space-y-1">
                  {f.evidence.map((e, i) => (
                    <li key={i} className="font-mono">
                      <span className="text-primary">{e.document || ""}</span>
                      {e.heading && <span className="text-muted-foreground"> · {e.heading}</span>}
                      {(e.quote || e.snippet) && (
                        <div className="text-muted-foreground/80 italic mt-0.5">
                          “{(e.quote || e.snippet || "").slice(0, 200)}”
                        </div>
                      )}
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
