"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { Play, Loader2, GitBranch, Zap, ChevronDown, Upload, FileCode } from "lucide-react";
import { cn } from "@/lib/utils";

const API = "/api";

// ── Types ─────────────────────────────────────────────────────────────────────

interface LineageNode {
  id: string;
  label: string;
  kind: "input" | "modified" | "computed";
  layer: number;
  data_steps: string[];
}

interface LineageEdge {
  source: string;
  target: string;
  kind: "assigns" | "conditions";
  data_step: string;
  expr: string;
}

interface LineageGraph {
  nodes: LineageNode[];
  edges: LineageEdge[];
  data_steps: string[];
}

interface TraceStep {
  kind: string;
  label: string;
  var: string | null;
  old_val: unknown;
  new_val: unknown;
}

interface SimResult {
  ciclo_id?: string;
  initial: Record<string, unknown>;
  final: Record<string, unknown>;
  row_passes_filter: boolean;
  steps: TraceStep[];
}

interface Cycle {
  CICLO_ID: string;
  CONTRATO: string;
  SEGMENTO: string;
  PD_ESTIMADA: number;
  LGD_ESTIMADA: number;
  EAD: number;
  DPDS: number;
  STAGE_IFRS9: number;
  PROVISION_PERIOD_MONTHS: number;
  COLATERAL_TIPO: string;
  [key: string]: unknown;
}

// ── SVG constants ─────────────────────────────────────────────────────────────

const NODE_W = 170;
const NODE_H = 34;
const LAYER_GAP = 100;
const ROW_H = 58;
const PAD_X = 24;
const PAD_Y = 40;

const KIND_FILL: Record<string, string> = {
  input:    "#eff6ff",
  modified: "#fefce8",
  computed: "#f0fdf4",
};
const KIND_STROKE: Record<string, string> = {
  input:    "#3b82f6",
  modified: "#ca8a04",
  computed: "#16a34a",
};
const KIND_LABEL: Record<string, string> = {
  input:    "Input field",
  modified: "Modified (floor/rule applied)",
  computed: "Computed output",
};

// ── Layout helpers ────────────────────────────────────────────────────────────

function computeLayout(graph: LineageGraph) {
  const byLayer: Record<number, LineageNode[]> = {};
  for (const n of graph.nodes) {
    if (!byLayer[n.layer]) byLayer[n.layer] = [];
    byLayer[n.layer].push(n);
  }

  const layers = Object.keys(byLayer).map(Number);
  const maxInLayer = Math.max(...layers.map((l) => byLayer[l].length));
  const numLayers = Math.max(...layers) + 1;
  const svgH = maxInLayer * ROW_H + PAD_Y * 2;
  const svgW = numLayers * (NODE_W + LAYER_GAP) + PAD_X * 2;

  const pos: Record<string, { x: number; y: number }> = {};
  for (const layer of layers) {
    const nodes = byLayer[layer];
    const layerH = nodes.length * ROW_H;
    const startY = (svgH - layerH) / 2;
    nodes.forEach((n, i) => {
      pos[n.id] = {
        x: PAD_X + layer * (NODE_W + LAYER_GAP),
        y: startY + i * ROW_H,
      };
    });
  }

  return { pos, svgW, svgH };
}

// ── Lineage Graph SVG ─────────────────────────────────────────────────────────

function LineageGraph({ graph }: { graph: LineageGraph }) {
  const [selected, setSelected] = useState<string | null>(null);
  const { pos, svgW, svgH } = computeLayout(graph);
  type Pos = { x: number; y: number };

  const connectedEdges = selected
    ? graph.edges.filter((e) => e.source === selected || e.target === selected)
    : [];
  const connectedIds = new Set(
    connectedEdges.flatMap((e) => [e.source, e.target])
  );

  function edgePath(e: LineageEdge) {
    const s: Pos | undefined = pos[e.source];
    const t: Pos | undefined = pos[e.target];
    if (!s || !t) return null;
    const x1 = s.x + NODE_W, y1 = s.y + NODE_H / 2;
    const x2 = t.x, y2 = t.y + NODE_H / 2;
    const mx = (x1 + x2) / 2;
    return `M ${x1} ${y1} C ${mx} ${y1} ${mx} ${y2} ${x2} ${y2}`;
  }

  const isActive = (e: LineageEdge) =>
    !selected || e.source === selected || e.target === selected;

  const layerHeaders = Array.from(
    new Map(
      graph.nodes.map((n) => [
        n.layer,
        n.data_steps[0] ?? (n.layer === 0 ? "Source fields" : `Step ${n.layer}`),
      ])
    ).entries()
  ).sort((a, b) => a[0] - b[0]);

  return (
    <div className="overflow-x-auto rounded-lg border border-border bg-card">
      <svg width={svgW} height={svgH + 28} style={{ display: "block" }}>
        {/* Arrow marker */}
        <defs>
          <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L0,6 L8,3 z" fill="#64748b" />
          </marker>
          <marker id="arrow-hi" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L0,6 L8,3 z" fill="#6366f1" />
          </marker>
          <marker id="arrow-cond" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L0,6 L8,3 z" fill="#f59e0b" />
          </marker>
        </defs>

        {/* Layer headers */}
        {layerHeaders.map(([layer, label]) => {
          const x = PAD_X + layer * (NODE_W + LAYER_GAP);
          return (
            <text key={layer} x={x + NODE_W / 2} y={18} textAnchor="middle"
              fontSize={10} fill="#94a3b8" fontFamily="monospace">
              {label.replace("work.", "")}
            </text>
          );
        })}

        {/* Edges */}
        {graph.edges.map((e, i) => {
          const d = edgePath(e);
          if (!d) return null;
          const active = isActive(e);
          const highlight = selected && (e.source === selected || e.target === selected);
          const isCond = e.kind === "conditions";
          return (
            <path
              key={i}
              d={d}
              fill="none"
              stroke={highlight ? "#6366f1" : isCond ? "#f59e0b" : "#64748b"}
              strokeWidth={highlight ? 2 : 1.5}
              strokeDasharray={isCond ? "5 3" : undefined}
              markerEnd={highlight ? "url(#arrow-hi)" : isCond ? "url(#arrow-cond)" : "url(#arrow)"}
              opacity={active ? (highlight ? 1 : 0.55) : 0.12}
            />
          );
        })}

        {/* Nodes */}
        {graph.nodes.map((n) => {
          const p: Pos | undefined = pos[n.id];
          if (!p) return null;
          const isSelected = selected === n.id;
          const dimmed = selected && !connectedIds.has(n.id) && selected !== n.id;
          return (
            <g
              key={n.id}
              transform={`translate(${p.x},${p.y + 28})`}
              style={{ cursor: "pointer" }}
              onClick={() => setSelected(isSelected ? null : n.id)}
            >
              <rect
                width={NODE_W}
                height={NODE_H}
                rx={6}
                fill={KIND_FILL[n.kind]}
                stroke={isSelected ? "#6366f1" : KIND_STROKE[n.kind]}
                strokeWidth={isSelected ? 2.5 : 1.5}
                opacity={dimmed ? 0.25 : 1}
              />
              <text
                x={NODE_W / 2}
                y={NODE_H / 2 + 4}
                textAnchor="middle"
                fontSize={11}
                fontFamily="monospace"
                fill={dimmed ? "#94a3b8" : "#1e293b"}
                fontWeight={isSelected ? "600" : "400"}
              >
                {n.label.length > 20 ? n.label.slice(0, 18) + "…" : n.label}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Selected-node detail */}
      {selected && (() => {
        const node = graph.nodes.find((n) => n.id === selected);
        const ins = graph.edges.filter((e) => e.target === selected);
        const outs = graph.edges.filter((e) => e.source === selected);
        return (
          <div className="border-t border-border px-4 py-3 text-xs space-y-1.5">
            <p className="font-semibold text-sm">{selected}</p>
            <p className="text-muted-foreground">{KIND_LABEL[node?.kind ?? "input"]}</p>
            {ins.length > 0 && (
              <p><span className="text-muted-foreground">Depends on: </span>
                {ins.map((e) => <span key={e.source} className={cn("mr-2 font-mono", e.kind === "conditions" ? "text-amber-600" : "text-slate-700")}>{e.source}</span>)}
              </p>
            )}
            {outs.length > 0 && (
              <p><span className="text-muted-foreground">Feeds into: </span>
                {outs.map((e) => <span key={e.target} className="mr-2 font-mono text-indigo-600">{e.target}</span>)}
              </p>
            )}
            {ins.filter(e => e.expr).map((e, i) => (
              <p key={i} className="font-mono text-muted-foreground bg-muted rounded px-2 py-1 text-[11px]">
                {e.expr}
              </p>
            ))}
          </div>
        );
      })()}
    </div>
  );
}

// ── Legend ────────────────────────────────────────────────────────────────────

function Legend() {
  return (
    <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
      {Object.entries(KIND_LABEL).map(([kind, label]) => (
        <span key={kind} className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-sm border"
            style={{ background: KIND_FILL[kind], borderColor: KIND_STROKE[kind] }} />
          {label}
        </span>
      ))}
      <span className="flex items-center gap-1.5">
        <svg width="24" height="10"><line x1="0" y1="5" x2="24" y2="5" stroke="#64748b" strokeWidth="1.5" /></svg>
        assigns
      </span>
      <span className="flex items-center gap-1.5">
        <svg width="24" height="10"><line x1="0" y1="5" x2="24" y2="5" stroke="#f59e0b" strokeWidth="1.5" strokeDasharray="4 2" /></svg>
        conditions
      </span>
    </div>
  );
}

// ── Upload zone ───────────────────────────────────────────────────────────────

interface UploadResult {
  filename: string;
  blocks: { task_name: string; block_type: string; chars: number }[];
  combined_code: string;
  lineage: LineageGraph;
}

function UploadZone({
  onResult,
}: {
  onResult: (r: UploadResult) => void;
}) {
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function processFile(file: File) {
    setLoading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API}/sas/upload`, { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail ?? `HTTP ${res.status}`);
      }
      onResult(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) processFile(file);
  }, []);

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      className={cn(
        "flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-6 py-8 cursor-pointer transition-colors",
        dragging ? "border-indigo-400 bg-indigo-50" : "border-border hover:border-muted-foreground/50 hover:bg-muted/30"
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".egp,.zip,.sas,.txt"
        className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) processFile(f); }}
      />
      {loading ? (
        <Loader2 size={20} className="animate-spin text-muted-foreground" />
      ) : (
        <Upload size={20} className="text-muted-foreground" />
      )}
      <p className="text-sm text-muted-foreground">
        {loading ? "Parsing file…" : "Drop a .egp, .zip or .sas file here, or click to browse"}
      </p>
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}

// ── Trace Step ────────────────────────────────────────────────────────────────

function TraceRow({ step }: { step: TraceStep }) {
  const changed = step.kind === "assign" && step.old_val !== step.new_val;
  const taken = step.kind === "if_taken";
  const skipped = step.kind === "if_skipped";
  const filterFail = step.kind === "filter_block";
  const filterPass = step.kind === "filter_pass";

  return (
    <div className={cn(
      "flex items-start gap-3 rounded-md px-3 py-2 text-sm",
      changed && "bg-amber-50 border border-amber-100",
      taken && "bg-green-50 border border-green-100",
      skipped && "bg-slate-50 border border-slate-100",
      filterFail && "bg-red-50 border border-red-200",
      filterPass && "bg-blue-50 border border-blue-100",
      !changed && !taken && !skipped && !filterFail && !filterPass && "bg-muted/40 border border-border/50",
    )}>
      <span className={cn(
        "mt-0.5 w-20 shrink-0 text-xs font-mono uppercase",
        changed && "text-amber-600",
        taken && "text-green-600",
        skipped && "text-slate-400",
        filterFail && "text-red-500",
        filterPass && "text-blue-500",
        !changed && !taken && !skipped && !filterFail && !filterPass && "text-muted-foreground",
      )}>
        {step.kind === "assign" ? "ASSIGN" :
         step.kind === "if_taken" ? "IF ✓" :
         step.kind === "if_skipped" ? "IF ✗" :
         step.kind === "filter_pass" ? "WHERE ✓" :
         step.kind === "filter_block" ? "WHERE ✗" : step.kind}
      </span>
      <div className="flex-1 font-mono text-xs break-all">
        <span>{step.label}</span>
        {changed && (
          <span className="ml-2 text-amber-700">
            {String(step.old_val)} → {String(step.new_val)}
          </span>
        )}
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function SASPage() {
  const [tab, setTab] = useState<"lineage" | "simulate">("lineage");
  const [lineage, setLineage] = useState<LineageGraph | null>(null);
  const [sampleCode, setSampleCode] = useState<string>("");
  const [lineageLoading, setLineageLoading] = useState(true);
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);

  const [cycles, setCycles] = useState<Cycle[]>([]);
  const [cyclesLoading, setCyclesLoading] = useState(true);
  const [selectedCycle, setSelectedCycle] = useState<string>("");
  const [simResult, setSimResult] = useState<SimResult | null>(null);
  const [simLoading, setSimLoading] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);

  function handleUpload(r: UploadResult) {
    setUploadResult(r);
    setLineage(r.lineage);
    setSampleCode(r.combined_code);
    setSimResult(null);
  }

  // Load sample code + lineage on mount
  useEffect(() => {
    (async () => {
      try {
        const sampleRes = await fetch(`${API}/sas/sample`);
        if (!sampleRes.ok) throw new Error("Sample not found");
        const { code } = await sampleRes.json();
        setSampleCode(code);

        const lgRes = await fetch(`${API}/sas/lineage`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ code }),
        });
        const lg = await lgRes.json();
        setLineage(lg);
      } catch (e) {
        console.error(e);
      } finally {
        setLineageLoading(false);
      }
    })();
  }, []);

  // Load cycles
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API}/sas/cycles?limit=200`);
        const data = await res.json();
        setCycles(data.items ?? []);
        if (data.items?.length) setSelectedCycle(data.items[0].CICLO_ID);
      } catch (e) {
        console.error(e);
      } finally {
        setCyclesLoading(false);
      }
    })();
  }, []);

  async function runSimulation() {
    if (!selectedCycle) return;
    setSimLoading(true);
    setSimError(null);
    setSimResult(null);
    try {
      const res = await fetch(`${API}/sas/simulate-cycle`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ciclo_id: selectedCycle }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail ?? `HTTP ${res.status}`);
      }
      setSimResult(await res.json());
    } catch (e: unknown) {
      setSimError(e instanceof Error ? e.message : String(e));
    } finally {
      setSimLoading(false);
    }
  }

  const cycleDetails = cycles.find((c) => c.CICLO_ID === selectedCycle);
  const changedFields = simResult
    ? Object.entries(simResult.final).filter(
        ([k, v]) => simResult.initial[k] !== v
      )
    : [];

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-6xl mx-auto px-6 py-8 space-y-6">

        {/* Header */}
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold flex items-center gap-2">
            <GitBranch size={22} />
            SAS Compiler
          </h1>
          <p className="text-sm text-muted-foreground">
            Field lineage graph and cycle-level simulation for IRB/IFRS9 DATA steps
          </p>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 border-b border-border">
          {(["lineage", "simulate"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={cn(
                "px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors",
                tab === t
                  ? "border-foreground text-foreground"
                  : "border-transparent text-muted-foreground hover:text-foreground"
              )}
            >
              {t === "lineage" ? "Field Lineage" : "Cycle Simulation"}
            </button>
          ))}
        </div>

        {/* ── Lineage tab ─────────────────────────────────────── */}
        {tab === "lineage" && (
          <div className="space-y-4">

            {/* Upload zone */}
            <UploadZone onResult={handleUpload} />

            {/* Uploaded file summary */}
            {uploadResult && (
              <div className="rounded-lg border border-border bg-card px-4 py-3 space-y-2">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <FileCode size={14} />
                  {uploadResult.filename}
                  <span className="text-xs text-muted-foreground font-normal ml-1">
                    — {uploadResult.blocks.length} code block{uploadResult.blocks.length !== 1 ? "s" : ""} found
                  </span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {uploadResult.blocks.map((b, i) => (
                    <span key={i} className="text-xs font-mono bg-muted rounded px-2 py-0.5">
                      {b.task_name}
                      <span className="text-muted-foreground ml-1">({b.block_type}, {b.chars} chars)</span>
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div className="flex items-center justify-between">
              <p className="text-sm text-muted-foreground">
                {uploadResult ? "Lineage from uploaded file." : "Showing built-in sample (sample_lgd.sas)."}{" "}
                Click any field to explore its upstream / downstream dependencies.
              </p>
              {lineage && (
                <p className="text-xs text-muted-foreground">
                  {lineage.nodes.length} fields · {lineage.edges.length} edges · {lineage.data_steps.length} DATA steps
                </p>
              )}
            </div>

            {lineageLoading ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground py-12">
                <Loader2 size={14} className="animate-spin" /> Loading lineage…
              </div>
            ) : lineage ? (
              <>
                <LineageGraph graph={lineage} />
                <Legend />

                {/* DATA step list */}
                <div className="rounded-lg border border-border bg-card p-4 space-y-2">
                  <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                    DATA step pipeline
                  </p>
                  {lineage.data_steps.map((step, i) => (
                    <div key={step} className="flex items-center gap-2 text-sm font-mono">
                      <span className="text-xs text-muted-foreground w-5">{i + 1}.</span>
                      <span>{step}</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <p className="text-sm text-muted-foreground py-8">
                Could not load lineage. Is the API running?
              </p>
            )}
          </div>
        )}

        {/* ── Simulate tab ────────────────────────────────────── */}
        {tab === "simulate" && (
          <div className="space-y-4">

            {/* Cycle selector */}
            <div className="rounded-lg border border-border bg-card p-4 space-y-3">
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide flex items-center gap-1.5">
                <Zap size={12} /> Select Cycle
              </p>

              {cyclesLoading ? (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 size={13} className="animate-spin" /> Loading cycles…
                </div>
              ) : (
                <div className="flex flex-wrap gap-3 items-end">
                  <div className="space-y-1">
                    <label className="text-xs text-muted-foreground">Cycle ID</label>
                    <div className="relative">
                      <select
                        value={selectedCycle}
                        onChange={(e) => setSelectedCycle(e.target.value)}
                        className="h-9 rounded-md border border-input bg-background px-3 pr-8 text-sm appearance-none focus:outline-none focus:ring-1 focus:ring-ring"
                      >
                        {cycles.map((c) => (
                          <option key={c.CICLO_ID} value={c.CICLO_ID}>
                            {c.CICLO_ID} — {c.SEGMENTO} / {c.COLATERAL_TIPO}
                          </option>
                        ))}
                      </select>
                      <ChevronDown size={14} className="absolute right-2.5 top-2.5 text-muted-foreground pointer-events-none" />
                    </div>
                  </div>
                  <button
                    onClick={runSimulation}
                    disabled={!selectedCycle || simLoading}
                    className="h-9 flex items-center gap-2 px-4 rounded-md bg-foreground text-background text-sm font-medium disabled:opacity-50"
                  >
                    {simLoading ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
                    Simulate
                  </button>
                </div>
              )}

              {/* Cycle preview */}
              {cycleDetails && (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-6 gap-y-1 pt-2 border-t border-border text-xs font-mono">
                  {["SEGMENTO", "COLATERAL_TIPO", "PD_ESTIMADA", "LGD_ESTIMADA", "EAD", "DPDS", "STAGE_IFRS9", "PROVISION_PERIOD_MONTHS"].map((k) => (
                    <div key={k}>
                      <span className="text-muted-foreground">{k}: </span>
                      <span>{String(cycleDetails[k])}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {simError && (
              <div className="rounded-md bg-destructive/10 border border-destructive/30 px-4 py-3 text-sm text-destructive">
                {simError}
              </div>
            )}

            {simResult && (
              <div className="space-y-4">
                {/* Summary banner */}
                <div className={cn(
                  "rounded-md px-4 py-2 text-sm font-medium",
                  simResult.row_passes_filter
                    ? "bg-green-50 border border-green-200 text-green-700"
                    : "bg-red-50 border border-red-200 text-red-700"
                )}>
                  {simResult.row_passes_filter
                    ? "✓ Row passes all WHERE filters"
                    : "✗ Row excluded by WHERE filter"}
                </div>

                {/* Changed fields */}
                {changedFields.length > 0 && (
                  <div className="rounded-lg border border-border bg-card p-4 space-y-2">
                    <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                      Fields modified by simulation
                    </p>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      {changedFields.map(([k, v]) => (
                        <div key={k} className="flex items-center gap-2 text-sm font-mono bg-amber-50 rounded px-2 py-1.5">
                          <span className="text-muted-foreground w-44 truncate">{k}</span>
                          <span className="text-slate-400 text-xs">{String(simResult.initial[k])}</span>
                          <span className="text-xs">→</span>
                          <span className="text-amber-700 font-semibold">{String(v)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Trace */}
                <div className="rounded-lg border border-border bg-card p-4 space-y-2">
                  <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                    Execution trace ({simResult.steps.length} steps)
                  </p>
                  <div className="space-y-1 max-h-[480px] overflow-y-auto">
                    {simResult.steps.map((s, i) => (
                      <TraceRow key={i} step={s} />
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
