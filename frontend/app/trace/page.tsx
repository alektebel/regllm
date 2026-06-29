"use client";

import { useState, useEffect, useMemo, useRef, useCallback } from "react";
import { useSearchParams } from "next/navigation";
import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCenter,
  forceCollide,
  type SimulationNodeDatum,
  type SimulationLinkDatum,
} from "d3-force";
import { GitBranch, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

// ─── Types ────────────────────────────────────────────────────────────────────

interface FieldNode {
  id: string;
  label: string;
  kind: "input" | "modified" | "computed";
  layer: number;
  data_steps: string[];
}

interface FieldEdge {
  source: string;
  target: string;
  kind: string;
  data_step: string;
  expr: string;
}

interface FieldLineage {
  nodes: FieldNode[];
  edges: FieldEdge[];
  data_steps: string[];
}

interface TraceStep {
  kind: string;
  label: string;
  var: string | null;
  old_val: unknown;
  new_val: unknown;
  data_step: string;
}

interface EvalResult {
  initial: Record<string, unknown>;
  final: Record<string, unknown>;
  steps: TraceStep[];
  row_passes_filter: boolean;
  filter_results: { data_step: string; condition: string; passed: boolean }[];
}

interface UninterpretedToken {
  line: number;
  statement: string;
  context: string;
}

interface ValidateResult {
  diagnostics: { level: string; code: string; message: string; field?: string; dataset?: string; table?: string }[];
  uninterpreted: UninterpretedToken[];
  total_uninterpreted: number;
}

// ─── Simulation types ─────────────────────────────────────────────────────────

interface SimNode extends SimulationNodeDatum {
  id: string;
  label: string;
  kind: string;
  layer: number;
  data_steps: string[];
}

interface SimLink extends SimulationLinkDatum<SimNode> {
  source: string | SimNode;
  target: string | SimNode;
  kind: string;
  data_step: string;
  expr: string;
}

// ─── Constants ────────────────────────────────────────────────────────────────

const KIND_COLOR: Record<string, string> = {
  input:    "#38bdf8",
  modified: "#f59e0b",
  computed: "#34d399",
};

const KIND_LABEL: Record<string, string> = {
  input:    "input",
  modified: "modified",
  computed: "computed",
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

function displayStep(step: string): string {
  return step
    .replace(/MVAR\[[^\]]+\]/g, "·")
    .replace(/·+/g, "·")
    .replace(/^·_?|_?·$/g, "")
    .replace(/_·_/g, "_")
    .replace(/^_|_$/g, "")
    .trim() || step;
}

function collapseExpr(expr: string): string {
  return expr.replace(/\s+/g, " ").trim();
}

function deriveInputValues(nodes: FieldNode[], evalResult: EvalResult | null): Record<string, unknown> {
  if (evalResult && Object.keys(evalResult.initial).length > 0) {
    return evalResult.initial;
  }
  const vals: Record<string, unknown> = {};
  for (const n of nodes) {
    if (n.kind === "input") {
      vals[n.id] = 0;
    }
  }
  return vals;
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function WhatIfPanel({
  initial,
  onEvaluate,
  evalResult,
  loading,
}: {
  initial: Record<string, unknown>;
  onEvaluate: (values: Record<string, unknown>) => void;
  evalResult: EvalResult | null;
  loading: boolean;
}) {
  const [values, setValues] = useState<Record<string, string>>({});

  useEffect(() => {
    const init: Record<string, string> = {};
    for (const [k, v] of Object.entries(initial)) {
      init[k] = String(v ?? "");
    }
    setValues(init);
  }, [initial]);

  const changedFields = useMemo(() => {
    if (!evalResult) return new Set<string>();
    const s = new Set<string>();
    for (const step of evalResult.steps) {
      if (step.kind === "assign" && step.var && step.old_val !== step.new_val) {
        s.add(step.var);
      }
    }
    return s;
  }, [evalResult]);

  const entries = Object.entries(values);
  if (entries.length === 0) {
    return (
      <div className="text-xs text-muted-foreground p-4">
        No input variables available for this code.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 p-4">
      <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
        Input Variables
      </div>
      <div className="flex flex-col gap-1.5">
        {entries.map(([k, v]) => (
          <div key={k} className="flex items-center gap-2">
            <label className="text-xs font-mono text-foreground w-28 shrink-0 truncate" title={k}>
              {k}
            </label>
            <input
              type="text"
              value={v}
              onChange={(e) => setValues((prev) => ({ ...prev, [k]: e.target.value }))}
              className={cn(
                "flex-1 rounded border border-border bg-card px-2 py-1 text-xs font-mono",
                "outline-none focus:ring-1 focus:ring-primary/40 transition-all",
              )}
            />
          </div>
        ))}
      </div>
      <button
        type="button"
        disabled={loading}
        onClick={() => {
          const parsed: Record<string, unknown> = {};
          for (const [k, v] of Object.entries(values)) {
            const n = Number(v);
            parsed[k] = v === "" ? null : Number.isNaN(n) ? v : n;
          }
          onEvaluate(parsed);
        }}
        className="self-start rounded bg-primary px-4 py-1.5 text-xs font-medium text-primary-foreground
                   hover:brightness-110 disabled:opacity-50 transition-all"
      >
        {loading ? "Evaluating…" : "Evaluate"}
      </button>

      {evalResult && (
        <div className="mt-2 flex flex-col gap-1">
          <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            Result
          </div>
          {Object.entries(evalResult.final).map(([k, v]) => {
            const orig = evalResult.initial[k];
            const changed = changedFields.has(k);
            return (
              <div key={k} className="flex items-center gap-2 text-xs font-mono">
                <span className="w-28 truncate shrink-0" title={k}>
                  {k}
                </span>
                <span className={cn(changed ? "text-amber-400" : "text-muted-foreground")}>
                  {String(v ?? "NULL")}
                </span>
                {changed && (
                  <span className="text-[10px] text-muted-foreground">
                    (was {String(orig ?? "NULL")})
                  </span>
                )}
              </div>
            );
          })}
          {!evalResult.row_passes_filter && (
            <div className="mt-1 text-xs text-destructive italic">
              Row excluded by WHERE or DELETE
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ValidationPanel({
  result,
  loading,
}: {
  result: ValidateResult | null;
  loading: boolean;
}) {
  if (loading) {
    return <div className="flex items-center gap-2 p-4 text-xs text-muted-foreground"><Loader2 className="h-3 w-3 animate-spin" /> Validating…</div>;
  }
  if (!result) {
    return <div className="p-4 text-xs text-muted-foreground">Run validation to see results.</div>;
  }

  const hasIssues = result.uninterpreted.length > 0 || result.diagnostics.length > 0;

  return (
    <div className="flex flex-col gap-4 p-4">
      {!hasIssues && (
        <div className="text-xs text-emerald-400">All statements recognised, no issues found.</div>
      )}

      {result.uninterpreted.length > 0 && (
        <div>
          <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
            Uninterpreted Statements ({result.total_uninterpreted})
          </div>
          <div className="flex flex-col gap-1 max-h-64 overflow-y-auto">
            {result.uninterpreted.map((u, i) => (
              <div key={i} className="flex items-start gap-2 text-xs font-mono bg-card/50 rounded px-2 py-1">
                <span className="text-muted-foreground shrink-0 w-8">L{u.line}</span>
                <span className="text-amber-400/80 truncate" title={u.statement}>
                  {u.statement}
                </span>
                <span className="text-muted-foreground/60 shrink-0 text-[10px]">({u.context})</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {result.diagnostics.length > 0 && (
        <div>
          <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
            Diagnostics ({result.diagnostics.length})
          </div>
          <div className="flex flex-col gap-1 max-h-64 overflow-y-auto">
            {result.diagnostics.map((d, i) => (
              <div key={i} className={cn(
                "flex items-start gap-2 text-xs bg-card/50 rounded px-2 py-1",
                d.level === "warning" ? "text-amber-400/80" : "text-muted-foreground",
              )}>
                <span className={cn(
                  "shrink-0 uppercase text-[10px] font-semibold",
                  d.level === "warning" ? "text-amber-400" : "text-sky-400",
                )}>
                  {d.code}
                </span>
                <span>{d.message}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Graph component ──────────────────────────────────────────────────────────

function ForceGraphView({
  nodes,
  edges,
  nodeMap: _nodeMap,
  changedFields,
  onSelectField,
}: {
  nodes: FieldNode[];
  edges: FieldEdge[];
  nodeMap: Map<string, FieldNode>;
  changedFields: Set<string>;
  onSelectField: (id: string) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const simRef = useRef<ReturnType<typeof forceSimulation<SimNode>> | null>(null);
  const simNodesRef = useRef<SimNode[]>([]);
  const simEdgesRef = useRef<SimLink[]>([]);
  const transformRef = useRef({ x: 0, y: 0, k: 0.6 });
  const changedRef = useRef(changedFields);
  const hoveredNodeRef = useRef<string | null>(null);
  const hoveredEdgeRef = useRef<SimLink | null>(null);

  changedRef.current = changedFields;

  useEffect(() => {
    if (nodes.length === 0 || !containerRef.current) return;

    const container = containerRef.current;
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;

    // Clear
    container.innerHTML = "";
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", String(width));
    svg.setAttribute("height", String(height));
    svg.style.width = "100%";
    svg.style.height = "100%";
    svg.style.cursor = "grab";
    container.appendChild(svg);
    svgRef.current = svg;

    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    svg.appendChild(g);

    // Build simulation
    const simNodes: SimNode[] = nodes.map((n) => ({
      ...n, x: width / 2 + (Math.random() - 0.5) * 200,
      y: height / 2 + (Math.random() - 0.5) * 200,
    }));
    const idMap = new Map(simNodes.map((n) => [n.id, n]));
    const simEdges: SimLink[] = edges.map((e) => ({
      source: idMap.get(e.source) ?? e.source,
      target: idMap.get(e.target) ?? e.target,
      kind: e.kind,
      data_step: e.data_step,
      expr: e.expr,
    }));

    simNodesRef.current = simNodes;
    simEdgesRef.current = simEdges;

    // Create SVG elements
    const edgeLines: SVGLineElement[] = simEdges.map(() => {
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("stroke", "currentColor");
      line.setAttribute("class", "stroke-border/60");
      line.setAttribute("stroke-width", "1");
      g.appendChild(line);
      return line;
    });

    const nodeGroups: SVGGElement[] = simNodes.map((n) => {
      const grp = document.createElementNS("http://www.w3.org/2000/svg", "g");
      grp.setAttribute("data-node-id", n.id);
      grp.style.cursor = "pointer";
      grp.style.pointerEvents = "all";

      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("r", "6");
      circle.setAttribute("fill", KIND_COLOR[n.kind] ?? "#6b7280");
      grp.appendChild(circle);

      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("class", "fill-foreground");
      text.setAttribute("font-size", "10");
      text.setAttribute("font-family", "monospace");
      text.textContent = n.label;
      grp.appendChild(text);

      const sub = document.createElementNS("http://www.w3.org/2000/svg", "text");
      sub.setAttribute("class", "fill-muted-foreground");
      sub.setAttribute("font-size", "8");
      sub.setAttribute("font-family", "monospace");
      sub.style.display = "none";
      sub.textContent = `${displayStep(n.data_steps[0] ?? "")} · ${KIND_LABEL[n.kind] ?? n.kind}`;
      grp.appendChild(sub);

      grp.addEventListener("pointerenter", () => {
        hoveredNodeRef.current = n.id;
        circle.setAttribute("r", "8");
        sub.style.display = "block";
      });
      grp.addEventListener("pointerleave", () => {
        hoveredNodeRef.current = null;
        circle.setAttribute("r", "6");
        sub.style.display = "none";
      });
      grp.addEventListener("click", () => onSelectField(n.id));

      g.appendChild(grp);
      return grp;
    });

    // Edge hover
    const edgeLabels: SVGTextElement[] = [];
    for (let i = 0; i < simEdges.length; i++) {
      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("class", "fill-muted-foreground");
      label.setAttribute("font-size", "9");
      label.setAttribute("font-family", "monospace");
      label.setAttribute("text-anchor", "middle");
      label.style.display = "none";
      label.textContent = collapseExpr(simEdges[i].expr).slice(0, 40);
      g.appendChild(label);
      edgeLabels.push(label);

      edgeLines[i].addEventListener("pointerenter", () => {
        hoveredEdgeRef.current = simEdges[i];
        edgeLines[i].setAttribute("class", "stroke-primary");
        edgeLines[i].setAttribute("stroke-width", "2");
        label.style.display = "block";
      });
      edgeLines[i].addEventListener("pointerleave", () => {
        hoveredEdgeRef.current = null;
        edgeLines[i].setAttribute("class", "stroke-border/60");
        edgeLines[i].setAttribute("stroke-width", "1");
        label.style.display = "none";
      });
    }

    // Simulation
    const sim = forceSimulation<SimNode>(simNodes)
      .force("link", forceLink<SimNode, SimLink>(simEdges).id((d) => d.id).distance(120))
      .force("charge", forceManyBody().strength(-200))
      .force("center", forceCenter(0, 0))
      .force("collide", forceCollide(30))
      .alphaDecay(0.02)
      .on("tick", () => {
        const { x: tx, y: ty, k } = transformRef.current;
        g.setAttribute("transform", `translate(${tx},${ty}) scale(${k})`);
        const changed = changedRef.current;

        for (let i = 0; i < simEdges.length; i++) {
          const e = simEdges[i];
          const s = e.source as SimNode;
          const t = e.target as SimNode;
          if (s.x == null || s.y == null || t.x == null || t.y == null) continue;
          edgeLines[i].setAttribute("x1", String(s.x));
          edgeLines[i].setAttribute("y1", String(s.y));
          edgeLines[i].setAttribute("x2", String(t.x));
          edgeLines[i].setAttribute("y2", String(t.y));
          edgeLabels[i].setAttribute("x", String((s.x + t.x) / 2));
          edgeLabels[i].setAttribute("y", String(((s.y ?? 0) + (t.y ?? 0)) / 2 - 10));
        }

        for (let i = 0; i < simNodes.length; i++) {
          const n = simNodes[i];
          if (n.x == null || n.y == null) continue;
          const grp = nodeGroups[i];
          const circle = grp.children[0] as SVGCircleElement;
          const text = grp.children[1] as SVGTextElement;
          const sub = grp.children[2] as SVGTextElement;

          circle.setAttribute("cx", String(n.x));
          circle.setAttribute("cy", String(n.y));
          circle.setAttribute("stroke", changed.has(n.id) ? "#fbbf24" : "none");
          circle.setAttribute("stroke-width", changed.has(n.id) ? "2" : "0");
          text.setAttribute("x", String(n.x + 12));
          text.setAttribute("y", String((n.y ?? 0) + 4));
          sub.setAttribute("x", String(n.x + 12));
          sub.setAttribute("y", String((n.y ?? 0) + 16));
        }
      });

    simRef.current = sim;

    // Pan/zoom
    let isPanning = false;
    let panStart = { x: 0, y: 0, tx: 0, ty: 0 };

    const onPointerDown = (e: PointerEvent) => {
      const target = e.target as Element;
      if (target.closest("[data-node-id]")) return;
      isPanning = true;
      panStart = { x: e.clientX, y: e.clientY, tx: transformRef.current.x, ty: transformRef.current.y };
      svg.setPointerCapture(e.pointerId);
    };
    const onPointerMove = (e: PointerEvent) => {
      if (isPanning) {
        transformRef.current.x = panStart.tx + (e.clientX - panStart.x);
        transformRef.current.y = panStart.ty + (e.clientY - panStart.y);
      }
    };
    const onPointerUp = () => { isPanning = false; };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 0.9 : 1.1;
      transformRef.current.k = Math.max(0.1, Math.min(5, transformRef.current.k * factor));
    };

    svg.addEventListener("pointerdown", onPointerDown);
    svg.addEventListener("pointermove", onPointerMove);
    svg.addEventListener("pointerup", onPointerUp);
    svg.addEventListener("wheel", onWheel, { passive: false });

    return () => {
      sim.stop();
      svg.removeEventListener("pointerdown", onPointerDown);
      svg.removeEventListener("pointermove", onPointerMove);
      svg.removeEventListener("pointerup", onPointerUp);
      svg.removeEventListener("wheel", onWheel);
    };
  }, [nodes, edges, onSelectField]);

  return <div ref={containerRef} className="flex-1 overflow-hidden" />;
}

// ─── Main Page ────────────────────────────────────────────────────────────────

type TabId = "graph" | "whatif" | "validation";

export default function TracePage() {
  const searchParams = useSearchParams();
  const initialField = searchParams?.get("field") ?? "";

  const [lineage, setLineage] = useState<FieldLineage | null>(null);
  const [code, setCode] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState(initialField);
  const [activeTab, setActiveTab] = useState<TabId>("graph");
  const [evalResult, setEvalResult] = useState<EvalResult | null>(null);
  const [evalLoading, setEvalLoading] = useState(false);
  const [validateResult, setValidateResult] = useState<ValidateResult | null>(null);
  const [validateLoading, setValidateLoading] = useState(false);
  const [changedFields, setChangedFields] = useState<Set<string>>(new Set());

  // Load sample code + lineage on mount
  useEffect(() => {
    async function load() {
      const sampleRes = await fetch("/api/sas/sample");
      if (!sampleRes.ok) throw new Error("Failed to load sample SAS");
      const { code: sasCode } = await sampleRes.json();
      setCode(sasCode);
      const lineageRes = await fetch("/api/sas/lineage", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: sasCode }),
      });
      if (!lineageRes.ok) throw new Error("Failed to compute lineage");
      setLineage(await lineageRes.json());
      setLoading(false);
    }
    load().catch((e) => { console.error(e); setLoading(false); });
  }, []);

  // Auto-validate on load
  useEffect(() => {
    if (!code) return;
    setValidateLoading(true);
    fetch("/api/sas/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
    })
      .then((r) => r.json())
      .then((r) => setValidateResult(r))
      .catch(() => {})
      .finally(() => setValidateLoading(false));
  }, [code]);

  // Derived lookups
  const { nodeMap, suggestions } = useMemo(() => {
    if (!lineage) return { nodeMap: new Map(), suggestions: [] as FieldNode[] };
    const nm = new Map<string, FieldNode>();
    for (const n of lineage.nodes) nm.set(n.id.toUpperCase(), n);
    const sorted = [...lineage.nodes].sort((a, b) => a.id.localeCompare(b.id));
    return { nodeMap: nm, suggestions: sorted };
  }, [lineage]);

  const filtered = useMemo(() => {
    if (!searchQuery.trim()) return suggestions;
    const q = searchQuery.toUpperCase();
    return suggestions.filter(
      (n) => n.id.toUpperCase().includes(q) || (n.data_steps[0] ?? "").toUpperCase().includes(q),
    );
  }, [searchQuery, suggestions]);

  async function handleEvaluate(values: Record<string, unknown>) {
    setEvalLoading(true);
    try {
      const res = await fetch("/api/sas/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code, values }),
      });
      const result: EvalResult = await res.json();
      setEvalResult(result);
      // Compute changed fields
      const changed = new Set<string>();
      for (const step of result.steps) {
        if (step.kind === "assign" && step.var && step.old_val !== step.new_val) {
          changed.add(step.var);
        }
      }
      setChangedFields(changed);
    } catch (e) {
      console.error(e);
    }
    setEvalLoading(false);
  }

  // Merge lineage edges with what-if eval trace for display
  const graphEdges = useMemo(() => {
    if (!lineage) return [];
    return lineage.edges;
  }, [lineage]);

  const handleSelectField = useCallback((id: string) => {
    setSearchQuery(id);
    setActiveTab("graph");
  }, []);

  return (
    <div className="h-screen bg-background text-foreground flex flex-col overflow-hidden">
      {/* Header */}
      <header className="flex items-center gap-3 px-6 py-2.5 border-b border-border bg-card/50 shrink-0">
        <h1 className="text-sm font-semibold flex items-center gap-2">
          <GitBranch className="h-4 w-4 text-primary" />
          Field Dependency Trace
        </h1>
        <div className="ml-auto flex items-center gap-3 text-[10px]">
          <span className="flex items-center gap-1 text-sky-400">
            <span className="w-2 h-2 rounded-full bg-sky-500 inline-block" /> input
          </span>
          <span className="flex items-center gap-1 text-amber-400">
            <span className="w-2 h-2 rounded-full bg-amber-500 inline-block" /> modified
          </span>
          <span className="flex items-center gap-1 text-emerald-400">
            <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" /> computed
          </span>
        </div>
      </header>

      {/* Tabs + Search */}
      <div className="flex items-center gap-2 px-6 py-2 border-b border-border shrink-0">
        <div className="flex gap-1">
          {["graph", "whatif", "validation"].map((tab) => (
            <button
              key={tab}
              type="button"
              onClick={() => setActiveTab(tab as TabId)}
              className={cn(
                "rounded px-3 py-1 text-[11px] font-medium transition-colors",
                activeTab === tab
                  ? "bg-card text-foreground border border-border"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {tab === "graph" ? "Graph" : tab === "whatif" ? "What-If" : "Validation"}
            </button>
          ))}
        </div>
        {activeTab === "graph" && (
          <div className="relative ml-auto max-w-[240px] w-full">
            <input
              type="text"
              placeholder="Search field…"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full rounded border border-border bg-card px-3 py-1.5 text-xs
                         placeholder:text-muted-foreground outline-none focus:ring-1 focus:ring-primary/40"
            />
            {searchQuery && filtered.length > 0 && (
              <ul className="absolute z-50 top-full mt-0.5 w-full max-h-40 overflow-y-auto rounded border border-border bg-card shadow-xl">
                {filtered.slice(0, 40).map((n) => (
                  <li key={n.id}>
                    <button
                      type="button"
                      onPointerDown={(e) => { e.preventDefault(); handleSelectField(n.id); }}
                      className="w-full text-left px-3 py-1.5 text-xs hover:bg-muted/40 flex items-baseline gap-2"
                    >
                      <span className="font-mono font-semibold truncate">{n.id}</span>
                      <span className={cn(
                        "ml-auto text-[9px] shrink-0",
                        n.kind === "input" ? "text-sky-400/70" : n.kind === "modified" ? "text-amber-400/70" : "text-emerald-400/70",
                      )}>
                        {n.kind}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 flex overflow-hidden">
        {loading && (
          <div className="flex items-center gap-2 m-auto text-muted-foreground text-sm">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading…
          </div>
        )}

        {!loading && activeTab === "graph" && lineage && (
          <ForceGraphView
            nodes={lineage.nodes}
            edges={graphEdges}
            nodeMap={nodeMap}
            changedFields={changedFields}
            onSelectField={handleSelectField}
          />
        )}

        {!loading && activeTab === "whatif" && (
          <>
            <div className="w-80 border-r border-border overflow-y-auto shrink-0">
              <WhatIfPanel
                initial={lineage ? deriveInputValues(lineage.nodes, evalResult) : {}}
                onEvaluate={handleEvaluate}
                evalResult={evalResult}
                loading={evalLoading}
              />
            </div>
            <ForceGraphView
              nodes={lineage?.nodes ?? []}
              edges={graphEdges}
              nodeMap={nodeMap}
              changedFields={changedFields}
              onSelectField={handleSelectField}
            />
          </>
        )}

        {!loading && activeTab === "validation" && (
          <div className="w-full max-w-2xl mx-auto">
            <ValidationPanel result={validateResult} loading={validateLoading} />
          </div>
        )}
      </div>
    </div>
  );
}
