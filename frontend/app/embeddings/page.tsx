"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import {
  ArrowLeft, Boxes, Loader2, RefreshCw, Sparkles, Network,
  BarChart3, Search, Grid3X3,
} from "lucide-react";
import { cn } from "@/lib/utils";

const Plot = dynamic(() => import("@/components/embeddings/PlotlyScatter"), {
  ssr: false,
  loading: () => <LoadingPane />,
});

const ForceGraph = dynamic(() => import("@/components/embeddings/ForceGraph"), {
  ssr: false,
  loading: () => <LoadingPane />,
});

const PlotlyHeatmap = dynamic(() => import("@/components/embeddings/PlotlyHeatmap"), {
  ssr: false,
  loading: () => <LoadingPane />,
});

function LoadingPane() {
  return (
    <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
      <Loader2 className="h-4 w-4 animate-spin mr-2" />Loading...
    </div>
  );
}

const API = "/api";

// ── Types ────────────────────────────────────────────────────────────────────

interface Point {
  id: string; path: string; heading: string; snippet: string;
  level: number; x: number; y: number; z?: number;
}

interface GraphNode {
  id: string; heading: string; path: string; cluster: number;
}

interface GraphEdge {
  source: string; target: string; weight: number;
}

interface AnisotropyResult {
  avg_cosine: number; std: number; n_pairs: number; verdict: string;
}

interface ClusterResult {
  method: string; n_clusters: number; silhouette: number | null;
  clusters: Record<string, { id: string; heading: string }[]>;
}

interface NNItem {
  id: string; heading: string;
  neighbors: { id: string; heading: string; cosine: number }[];
}

// ── Tabs ─────────────────────────────────────────────────────────────────────

type Tab = "scatter" | "graph" | "anisotropy" | "clusters" | "nn-audit" | "heatmap";

const TABS: { key: Tab; label: string; icon: React.ReactNode }[] = [
  { key: "scatter", label: "Scatter", icon: <Boxes className="h-3.5 w-3.5" /> },
  { key: "graph", label: "Graph", icon: <Network className="h-3.5 w-3.5" /> },
  { key: "anisotropy", label: "Anisotropy", icon: <BarChart3 className="h-3.5 w-3.5" /> },
  { key: "clusters", label: "Clusters", icon: <Grid3X3 className="h-3.5 w-3.5" /> },
  { key: "nn-audit", label: "NN Audit", icon: <Search className="h-3.5 w-3.5" /> },
  { key: "heatmap", label: "Heatmap", icon: <Grid3X3 className="h-3.5 w-3.5" /> },
];

const METHOD_LABEL: Record<string, string> = { pca: "PCA", tsne: "t-SNE", umap: "UMAP" };

// ── Main page ────────────────────────────────────────────────────────────────

export default function EmbeddingsPage() {
  const [tab, setTab] = useState<Tab>("scatter");
  const [methods, setMethods] = useState<string[]>(["pca", "tsne"]);
  const [method, setMethod] = useState("pca");
  const [dims, setDims] = useState<2 | 3>(3);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Scatter state
  const [points, setPoints] = useState<Point[]>([]);
  const [selected, setSelected] = useState<Point | null>(null);

  // Graph state
  const [graphNodes, setGraphNodes] = useState<GraphNode[]>([]);
  const [graphEdges, setGraphEdges] = useState<GraphEdge[]>([]);
  const [graphK, setGraphK] = useState(5);
  const [graphThreshold, setGraphThreshold] = useState(0.3);
  const [selectedGraphNode, setSelectedGraphNode] = useState<GraphNode | null>(null);

  // Inspection state
  const [anisotropy, setAnisotropy] = useState<AnisotropyResult | null>(null);
  const [clusters, setClusters] = useState<ClusterResult | null>(null);
  const [nnAudit, setNnAudit] = useState<NNItem[]>([]);
  const [heatmapData, setHeatmapData] = useState<{ ids: string[]; matrix: number[][] } | null>(null);

  useEffect(() => {
    fetch(`${API}/embeddings/methods`)
      .then((r) => r.json())
      .then((d) => d.methods && setMethods(d.methods))
      .catch(() => {});
  }, []);

  // ── Loaders ────────────────────────────────────────────────────────────

  const loadScatter = useCallback(async (rebuild = false) => {
    setLoading(true); setError(null);
    try {
      const res = await fetch(`${API}/embeddings/project`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ method, n_components: dims, rebuild }),
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
      const data = await res.json();
      setPoints(data.points);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); setPoints([]); }
    finally { setLoading(false); }
  }, [method, dims]);

  const loadGraph = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const res = await fetch(`${API}/embeddings/inspect/graph`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ k: graphK, threshold: graphThreshold }),
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
      const data = await res.json();
      setGraphNodes(data.nodes);
      setGraphEdges(data.edges);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  }, [graphK, graphThreshold]);

  const loadAnisotropy = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API}/embeddings/inspect/anisotropy`);
      setAnisotropy(await res.json());
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  }, []);

  const loadClusters = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API}/embeddings/inspect/clusters`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ method: "hdbscan" }),
      });
      setClusters(await res.json());
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  }, []);

  const loadNNAudit = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API}/embeddings/inspect/nn-audit?k=5`);
      const data = await res.json();
      setNnAudit(data.items || []);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  }, []);

  const loadHeatmap = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API}/embeddings/inspect/similarity?max_points=150`);
      setHeatmapData(await res.json());
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  }, []);

  // Auto-load on tab change
  useEffect(() => {
    setError(null);
    if (tab === "scatter") loadScatter();
    else if (tab === "graph") loadGraph();
    else if (tab === "anisotropy") loadAnisotropy();
    else if (tab === "clusters") loadClusters();
    else if (tab === "nn-audit") loadNNAudit();
    else if (tab === "heatmap") loadHeatmap();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab]);

  // Reload scatter on method/dim change
  useEffect(() => { if (tab === "scatter") loadScatter(); }, [method, dims]);

  const pathColors = useMemo(() => {
    const palette = ["#60a5fa","#f472b6","#34d399","#fbbf24","#a78bfa","#fb923c","#22d3ee","#f87171"];
    const map = new Map<string, string>();
    Array.from(new Set(points.map((p) => p.path))).forEach((p, i) => map.set(p, palette[i % palette.length]));
    return map;
  }, [points]);

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-screen bg-background text-foreground">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 border-b border-border bg-card/60">
        <div className="flex items-center gap-3">
          <Boxes className="h-5 w-5 text-primary" />
          <div>
            <h1 className="text-base font-semibold">Embedding Inspector</h1>
            <p className="text-xs text-muted-foreground">
              Scatter · Force graph · Anisotropy · Clusters · NN audit · Heatmap
            </p>
          </div>
        </div>
        <Link href="/diff" className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground">
          <ArrowLeft className="h-3.5 w-3.5" /> Back
        </Link>
      </header>

      {/* Tab bar */}
      <div className="flex items-center gap-1 px-4 py-2 border-b border-border bg-card/30">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={cn(
              "flex items-center gap-1.5 px-3 py-1.5 text-xs rounded transition-colors",
              tab === t.key ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground hover:bg-muted/40"
            )}
          >
            {t.icon} {t.label}
          </button>
        ))}
        {loading && <Loader2 className="h-3.5 w-3.5 animate-spin ml-auto text-muted-foreground" />}
      </div>

      {/* Body */}
      <div className="flex-1 grid grid-cols-12 gap-3 p-3 min-h-0 overflow-hidden">
        {/* Main area */}
        <div className="col-span-9 flex flex-col border border-border rounded bg-card/40 min-h-0">
          {/* Scatter controls */}
          {tab === "scatter" && (
            <div className="flex items-center gap-3 px-3 py-2 border-b border-border flex-wrap">
              <div className="flex items-center gap-1 bg-muted/40 rounded p-0.5">
                {methods.map((m) => (
                  <button key={m} onClick={() => setMethod(m)}
                    className={cn("px-2.5 py-1 text-xs rounded transition-colors",
                      method === m ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")}>
                    {METHOD_LABEL[m] || m}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-1 bg-muted/40 rounded p-0.5">
                {[2, 3].map((d) => (
                  <button key={d} onClick={() => setDims(d as 2 | 3)}
                    className={cn("px-2.5 py-1 text-xs rounded transition-colors",
                      dims === d ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")}>
                    {d}D
                  </button>
                ))}
              </div>
              <button onClick={() => loadScatter(true)} disabled={loading}
                className="flex items-center gap-1.5 px-2.5 py-1 text-xs rounded bg-muted/40 hover:bg-muted/60 disabled:opacity-50">
                <RefreshCw className="h-3.5 w-3.5" /> Rebuild
              </button>
              <span className="text-xs text-muted-foreground ml-auto">{points.length} sections</span>
            </div>
          )}

          {/* Graph controls */}
          {tab === "graph" && (
            <div className="flex items-center gap-3 px-3 py-2 border-b border-border flex-wrap">
              <label className="text-xs text-muted-foreground">k</label>
              <input type="number" min={1} max={20} value={graphK}
                onChange={(e) => setGraphK(Number(e.target.value))}
                className="w-14 px-1.5 py-1 text-xs rounded bg-muted/40 border border-border" />
              <label className="text-xs text-muted-foreground">threshold</label>
              <input type="number" min={0} max={1} step={0.05} value={graphThreshold}
                onChange={(e) => setGraphThreshold(Number(e.target.value))}
                className="w-16 px-1.5 py-1 text-xs rounded bg-muted/40 border border-border" />
              <button onClick={loadGraph} disabled={loading}
                className="flex items-center gap-1.5 px-2.5 py-1 text-xs rounded bg-muted/40 hover:bg-muted/60 disabled:opacity-50">
                <RefreshCw className="h-3.5 w-3.5" /> Reload
              </button>
              <span className="text-xs text-muted-foreground ml-auto">
                {graphNodes.length} nodes · {graphEdges.length} edges
              </span>
            </div>
          )}

          {/* Main panel */}
          <div className="flex-1 min-h-0 relative">
            {error && (
              <div className="absolute inset-0 flex items-center justify-center text-sm text-destructive p-4 text-center">{error}</div>
            )}

            {tab === "scatter" && !error && (
              <Plot points={points} dims={dims} pathColors={pathColors} onSelect={setSelected} />
            )}

            {tab === "graph" && !error && graphNodes.length > 0 && (
              <ForceGraph nodes={graphNodes} edges={graphEdges} onSelect={setSelectedGraphNode} />
            )}

            {tab === "anisotropy" && !error && anisotropy && (
              <div className="p-6 space-y-4">
                <div className="grid grid-cols-3 gap-4">
                  <MetricCard label="Avg cosine" value={anisotropy.avg_cosine.toFixed(4)} />
                  <MetricCard label="Std" value={anisotropy.std.toFixed(4)} />
                  <MetricCard label="Pairs sampled" value={anisotropy.n_pairs.toLocaleString()} />
                </div>
                <div className={cn("p-4 rounded border text-sm",
                  anisotropy.avg_cosine > 0.7 ? "border-destructive/50 bg-destructive/10 text-destructive" :
                  anisotropy.avg_cosine > 0.4 ? "border-yellow-500/50 bg-yellow-500/10 text-yellow-200" :
                  "border-green-500/50 bg-green-500/10 text-green-200")}>
                  {anisotropy.verdict}
                </div>
              </div>
            )}

            {tab === "clusters" && !error && clusters && (
              <div className="p-4 space-y-3 overflow-y-auto max-h-full">
                <div className="grid grid-cols-3 gap-4">
                  <MetricCard label="Clusters" value={String(clusters.n_clusters)} />
                  <MetricCard label="Method" value={clusters.method} />
                  <MetricCard label="Silhouette" value={clusters.silhouette?.toFixed(4) ?? "n/a"} />
                </div>
                {Object.entries(clusters.clusters).map(([cid, members]) => (
                  <div key={cid} className="border border-border rounded p-3">
                    <div className="text-xs font-semibold mb-1">Cluster {cid} ({members.length})</div>
                    <div className="flex flex-wrap gap-1">
                      {members.slice(0, 20).map((m) => (
                        <span key={m.id} className="text-[11px] px-1.5 py-0.5 rounded bg-muted/40">{m.heading.slice(0, 40)}</span>
                      ))}
                      {members.length > 20 && <span className="text-[11px] text-muted-foreground">+{members.length - 20} more</span>}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {tab === "nn-audit" && !error && nnAudit.length > 0 && (
              <div className="p-4 space-y-2 overflow-y-auto max-h-full">
                {nnAudit.map((item) => (
                  <div key={item.id} className="border border-border rounded p-3">
                    <div className="text-xs font-semibold mb-1">{item.heading}</div>
                    <div className="space-y-0.5">
                      {item.neighbors.map((nb, i) => (
                        <div key={i} className="flex justify-between text-[11px]">
                          <span className="text-muted-foreground truncate mr-2">{nb.heading}</span>
                          <span className={cn("font-mono shrink-0",
                            nb.cosine > 0.9 ? "text-green-400" : nb.cosine > 0.7 ? "text-yellow-400" : "text-red-400"
                          )}>{nb.cosine.toFixed(4)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {tab === "heatmap" && !error && heatmapData && (
              <PlotlyHeatmap ids={heatmapData.ids} matrix={heatmapData.matrix} />
            )}
          </div>
        </div>

        {/* Sidebar */}
        <div className="col-span-3 flex flex-col border border-border rounded bg-card/40 min-h-0 overflow-y-auto p-3">
          <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground mb-2">
            <Sparkles className="h-3.5 w-3.5" /> Details
          </div>
          {tab === "scatter" && selected && (
            <div className="text-xs space-y-2">
              <div className="font-mono text-[11px] text-muted-foreground break-all">{selected.path}</div>
              <div className="font-semibold">{selected.heading}</div>
              <div className="text-muted-foreground whitespace-pre-wrap">{selected.snippet}</div>
            </div>
          )}
          {tab === "graph" && selectedGraphNode && (
            <div className="text-xs space-y-2">
              <div className="font-mono text-[11px] text-muted-foreground break-all">{selectedGraphNode.path}</div>
              <div className="font-semibold">{selectedGraphNode.heading}</div>
              <div className="text-muted-foreground">Cluster {selectedGraphNode.cluster}</div>
              <div className="text-muted-foreground mt-2">
                Edges: {graphEdges.filter(e => e.source === selectedGraphNode.id || e.target === selectedGraphNode.id).length}
              </div>
            </div>
          )}
          {!selected && !selectedGraphNode && (
            <div className="text-xs text-muted-foreground">
              {tab === "scatter" ? "Click a point to see details." :
               tab === "graph" ? "Click a node to see details. Drag to rearrange. Scroll to zoom." :
               "Select a tab to inspect the embedding space."}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-border rounded p-3 bg-muted/20">
      <div className="text-[11px] text-muted-foreground">{label}</div>
      <div className="text-lg font-semibold mt-0.5">{value}</div>
    </div>
  );
}
