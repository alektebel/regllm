"use client";

import { useState, useEffect, useMemo, useRef } from "react";
import { GitBranch, Loader2, ArrowLeft } from "lucide-react";
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
  kind: "assigns" | "conditions";
  data_step: string;
  expr: string;
}

interface FieldLineage {
  nodes: FieldNode[];
  edges: FieldEdge[];
  data_steps: string[];
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** BFS backward from root using predMap; returns depth map keyed by UPPERCASE id. */
function computeBFS(rootId: string, predMap: Map<string, FieldEdge[]>): Map<string, number> {
  const depth = new Map<string, number>();
  const queue: string[] = [rootId.toUpperCase()];
  depth.set(rootId.toUpperCase(), 0);
  while (queue.length > 0) {
    const curr = queue.shift()!;
    const d = depth.get(curr)!;
    for (const edge of predMap.get(curr) ?? []) {
      const key = edge.source.toUpperCase();
      if (!depth.has(key)) {
        depth.set(key, d + 1);
        queue.push(key);
      }
    }
  }
  return depth;
}

/**
 * Deduplicate edges: for the same (source, target) pair keep assigns over
 * conditions. This prevents a field showing twice when it appears in both
 * the predicate and the expression of the same assignment.
 */
function dedupeEdges(edges: FieldEdge[]): FieldEdge[] {
  const seen = new Map<string, FieldEdge>();
  for (const e of edges) {
    const key = `${e.source.toUpperCase()}::${e.target.toUpperCase()}`;
    const prev = seen.get(key);
    if (!prev || e.kind === "assigns") seen.set(key, e);
  }
  return Array.from(seen.values());
}

/** Group deduplicated edges by their expression string. */
function groupByExpr(edges: FieldEdge[]): [string, FieldEdge[]][] {
  const map = new Map<string, FieldEdge[]>();
  for (const e of dedupeEdges(edges)) {
    const key = e.expr?.trim() || "__conditions__";
    const g = map.get(key) ?? [];
    g.push(e);
    map.set(key, g);
  }
  return Array.from(map.entries());
}

const KIND_STYLE: Record<string, string> = {
  input:    "border-sky-500/60 bg-sky-500/10 text-sky-200",
  modified: "border-amber-500/60 bg-amber-500/10 text-amber-200",
  computed: "border-emerald-500/60 bg-emerald-500/10 text-emerald-200",
};
const KIND_MUTED: Record<string, string> = {
  input:    "text-sky-400/70",
  modified: "text-amber-400/70",
  computed: "text-emerald-400/70",
};

// ─── Sub-components ───────────────────────────────────────────────────────────

interface FieldCardProps {
  nodeId: string;
  instanceId: string;
  currentDepth: number;
  nodeMap: Map<string, FieldNode>;
  predMap: Map<string, FieldEdge[]>;
  srcMap: Map<string, FieldEdge[]>;
  bfsDepth: Map<string, number>;
  expanded: Map<string, boolean>;
  onToggle: (instanceId: string) => void;
}

function FieldCard({
  nodeId, instanceId, currentDepth,
  nodeMap, predMap, srcMap, bfsDepth,
  expanded, onToggle,
}: FieldCardProps) {
  const node = nodeMap.get(nodeId.toUpperCase());
  const allPreds = predMap.get(nodeId.toUpperCase()) ?? [];
  const isExpanded = expanded.get(instanceId) ?? false;
  const kind = node?.kind ?? "computed";

  // Table = step where field is *written*, fall back to step where it is *read*
  const tableName =
    node?.data_steps[0] ??
    srcMap.get(nodeId.toUpperCase())?.[0]?.data_step ??
    "—";

  // Split predecessors into:
  //   direct  — BFS depth === currentDepth + 1  (show as child cards)
  //   shallow — BFS depth < currentDepth + 1  (already shown at a shallower level)
  //   unknown — not in BFS trace  (show as child cards anyway)
  const uniquePreds = dedupeEdges(allPreds);
  const directPreds: FieldEdge[] = [];
  const shallowPreds: { source: string; depth: number }[] = [];

  for (const e of uniquePreds) {
    const d = bfsDepth.get(e.source.toUpperCase());
    if (d === undefined || d === currentDepth + 1) {
      directPreds.push(e);
    } else if (d < currentDepth + 1) {
      // Deduplicate shallow refs too
      if (!shallowPreds.find((s) => s.source.toUpperCase() === e.source.toUpperCase())) {
        shallowPreds.push({ source: e.source, depth: d });
      }
    }
  }

  const isLeaf = directPreds.length === 0 && shallowPreds.length === 0;
  const exprGroups = isExpanded ? groupByExpr(directPreds) : [];

  return (
    <div className="flex flex-col items-center">
      {/* Card button */}
      <button
        type="button"
        title={isLeaf ? "Leaf input — no upstream dependencies" : undefined}
        onClick={isLeaf ? undefined : () => onToggle(instanceId)}
        className={cn(
          "rounded-xl border-2 px-5 py-3 min-w-[148px] text-center transition-all select-none",
          KIND_STYLE[kind] ?? KIND_STYLE.computed,
          isLeaf ? "opacity-60 cursor-default" : "cursor-pointer hover:brightness-125 hover:scale-[1.03]",
          isExpanded && "ring-2 ring-white/20",
        )}
      >
        <div className={cn("text-[10px] font-normal truncate max-w-[160px]", KIND_MUTED[kind])}>
          {tableName}
        </div>
        <div className="text-[13px] font-bold font-mono mt-0.5 tracking-wide">
          {node?.id ?? nodeId}
        </div>
        {!isLeaf && (
          <div className="text-[9px] text-current/40 mt-1">
            depth {currentDepth}
          </div>
        )}
      </button>

      {/* Expanded children */}
      {isExpanded && (
        <div className="flex flex-col items-center w-full">
          {/* Shallow references (already shown higher in the tree) */}
          {shallowPreds.length > 0 && (
            <>
              <div className="w-px h-5 bg-border/60" />
              <div className="flex flex-row gap-2 flex-wrap justify-center">
                {shallowPreds.map(({ source, depth: d }) => (
                  <div
                    key={source}
                    title={`${source} is already reachable at depth ${d} from the root`}
                    className="px-3 py-1.5 rounded-lg border border-dashed border-zinc-600/60
                               bg-zinc-900/40 text-[10px] font-mono text-zinc-500 flex items-center gap-1.5"
                  >
                    <span className="text-zinc-600">↗</span>
                    {source}
                    <span className="text-zinc-600">depth {d}</span>
                  </div>
                ))}
              </div>
            </>
          )}

          {/* Direct predecessors via expression bridges */}
          {exprGroups.map(([expr, edges], groupIdx) => (
            <div key={expr} className="flex flex-col items-center w-full">
              {groupIdx > 0 && (
                <div className="flex items-center gap-2 my-2 w-full max-w-[280px]">
                  <div className="flex-1 border-t border-dashed border-border/40" />
                  <span className="text-[9px] uppercase tracking-widest text-muted-foreground">or</span>
                  <div className="flex-1 border-t border-dashed border-border/40" />
                </div>
              )}
              <ExprBridge
                expr={expr}
                edges={edges}
                parentInstanceId={instanceId}
                currentDepth={currentDepth}
                nodeMap={nodeMap}
                predMap={predMap}
                srcMap={srcMap}
                bfsDepth={bfsDepth}
                expanded={expanded}
                onToggle={onToggle}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

interface ExprBridgeProps {
  expr: string;
  edges: FieldEdge[];
  parentInstanceId: string;
  currentDepth: number;
  nodeMap: Map<string, FieldNode>;
  predMap: Map<string, FieldEdge[]>;
  srcMap: Map<string, FieldEdge[]>;
  bfsDepth: Map<string, number>;
  expanded: Map<string, boolean>;
  onToggle: (instanceId: string) => void;
}

function ExprBridge({
  expr, edges, parentInstanceId, currentDepth,
  nodeMap, predMap, srcMap, bfsDepth, expanded, onToggle,
}: ExprBridgeProps) {
  const isConditions = expr === "__conditions__";

  return (
    <div className="flex flex-col items-center">
      <div className="w-px h-5 bg-border/60" />
      <div
        className={cn(
          "px-3 py-1 rounded-full border border-border/70 bg-zinc-900/80",
          "text-[10px] font-mono text-zinc-300 max-w-[320px] truncate my-0.5",
          isConditions && "italic text-zinc-500",
        )}
        title={isConditions ? "conditional branch" : expr}
      >
        {isConditions ? "condition" : `= ${expr}`}
      </div>
      <div className="w-px h-5 bg-border/60" />
      <div className="flex flex-row gap-6 flex-wrap justify-center items-start">
        {edges.map((edge) => (
          <FieldCard
            key={`${parentInstanceId}::${edge.source}`}
            nodeId={edge.source}
            instanceId={`${parentInstanceId}::${edge.source}`}
            currentDepth={currentDepth + 1}
            nodeMap={nodeMap}
            predMap={predMap}
            srcMap={srcMap}
            bfsDepth={bfsDepth}
            expanded={expanded}
            onToggle={onToggle}
          />
        ))}
      </div>
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function TracePage() {
  const [lineage, setLineage] = useState<FieldLineage | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [rootField, setRootField] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Map<string, boolean>>(new Map());
  const [showDropdown, setShowDropdown] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);

  // ── Load lineage on mount
  useEffect(() => {
    async function load() {
      const sampleRes = await fetch("/api/sas/sample");
      if (!sampleRes.ok) throw new Error("Failed to load sample SAS");
      const { code } = await sampleRes.json();
      const lineageRes = await fetch("/api/sas/lineage", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      });
      if (!lineageRes.ok) throw new Error("Failed to compute lineage");
      setLineage(await lineageRes.json());
      setLoading(false);
    }
    load().catch((e) => { setError(String(e)); setLoading(false); });
  }, []);

  // ── Close dropdown on outside click
  useEffect(() => {
    function onPointerDown(e: PointerEvent) {
      if (searchRef.current && !searchRef.current.contains(e.target as Node))
        setShowDropdown(false);
    }
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, []);

  // ── Derived lookups (recomputed only when lineage changes)
  const { nodeMap, predMap, srcMap, suggestions } = useMemo(() => {
    if (!lineage) return { nodeMap: new Map(), predMap: new Map(), srcMap: new Map(), suggestions: [] };

    const nodeMap = new Map<string, FieldNode>();
    for (const n of lineage.nodes) nodeMap.set(n.id.toUpperCase(), n);

    const predMap = new Map<string, FieldEdge[]>();
    const srcMap = new Map<string, FieldEdge[]>();
    for (const e of lineage.edges) {
      const tk = e.target.toUpperCase();
      const ta = predMap.get(tk) ?? []; ta.push(e); predMap.set(tk, ta);
      const sk = e.source.toUpperCase();
      const sa = srcMap.get(sk) ?? []; sa.push(e); srcMap.set(sk, sa);
    }

    const sorted = [...lineage.nodes].sort((a, b) => a.id.localeCompare(b.id));
    return { nodeMap, predMap, srcMap, suggestions: sorted };
  }, [lineage]);

  // ── BFS depth from current root (recomputed when root or lineage changes)
  const bfsDepth = useMemo(() => {
    if (!rootField) return new Map<string, number>();
    return computeBFS(rootField, predMap);
  }, [rootField, predMap]);

  // ── Search filter
  const filtered = useMemo(() => {
    if (!query.trim()) return suggestions;
    const q = query.toUpperCase();
    return suggestions.filter(
      (n) => n.id.toUpperCase().includes(q) || (n.data_steps[0] ?? "").toUpperCase().includes(q),
    );
  }, [query, suggestions]);

  function selectField(id: string) {
    setRootField(id);
    setQuery(id);
    setExpanded(new Map());
    setShowDropdown(false);
  }

  function handleToggle(instanceId: string) {
    setExpanded((prev) => {
      const next = new Map(prev);
      next.set(instanceId, !(prev.get(instanceId) ?? false));
      return next;
    });
  }

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      {/* Header */}
      <header className="flex items-center gap-3 px-6 py-3 border-b border-border bg-card/50 shrink-0">
        <a href="/diff" className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors">
          <ArrowLeft className="h-3 w-3" /> Diff Explainer
        </a>
        <div className="w-px h-4 bg-border" />
        <h1 className="text-sm font-semibold flex items-center gap-2">
          <GitBranch className="h-4 w-4 text-primary" />
          Field Dependency Trace
        </h1>
        <div className="ml-auto flex items-center gap-3 text-[11px]">
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

      {/* Search */}
      <div className="flex justify-center px-6 pt-6 pb-2 shrink-0">
        <div ref={searchRef} className="relative w-full max-w-sm">
          <input
            type="text"
            placeholder={loading ? "Loading fields…" : "Search field or table…"}
            disabled={loading}
            value={query}
            onChange={(e) => { setQuery(e.target.value); setShowDropdown(true); }}
            onFocus={() => setShowDropdown(true)}
            className={cn(
              "w-full rounded-lg border border-border bg-card px-4 py-2.5 text-sm",
              "placeholder:text-muted-foreground outline-none focus:ring-2 focus:ring-primary/40 transition-all",
              loading && "opacity-50 cursor-not-allowed",
            )}
          />
          {showDropdown && filtered.length > 0 && (
            <ul className="absolute z-50 top-full mt-1 w-full max-h-56 overflow-y-auto rounded-lg border border-border bg-card shadow-xl">
              {filtered.slice(0, 60).map((node) => (
                <li key={node.id}>
                  <button
                    type="button"
                    onPointerDown={(e) => { e.preventDefault(); selectField(node.id); }}
                    className="w-full text-left px-4 py-2 text-sm hover:bg-muted/40 flex items-baseline gap-2"
                  >
                    <span className="text-[10px] text-muted-foreground font-mono min-w-0 truncate max-w-[120px]">
                      {node.data_steps[0] ?? "—"}
                    </span>
                    <span className="font-mono font-semibold text-foreground truncate">{node.id}</span>
                    <span className={cn("ml-auto text-[9px] shrink-0", KIND_MUTED[node.kind])}>{node.kind}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {/* Main canvas */}
      <main className="flex-1 flex flex-col items-center px-6 py-4 overflow-auto">
        {loading && (
          <div className="flex items-center gap-2 mt-16 text-muted-foreground text-sm">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading lineage…
          </div>
        )}
        {error && <div className="mt-16 text-destructive text-sm">{error}</div>}
        {!loading && !error && !rootField && (
          <div className="mt-16 text-muted-foreground text-sm">
            Search for a field above to start tracing its dependencies backward.
          </div>
        )}
        {!loading && rootField && (
          <div className="overflow-x-auto w-full">
            <div className="flex flex-col items-center min-w-max pb-16 pt-2">
              <FieldCard
                nodeId={rootField}
                instanceId={rootField}
                currentDepth={0}
                nodeMap={nodeMap}
                predMap={predMap}
                srcMap={srcMap}
                bfsDepth={bfsDepth}
                expanded={expanded}
                onToggle={handleToggle}
              />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
