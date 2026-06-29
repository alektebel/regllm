"use client";

import { useCallback, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import {
  Upload, Loader2, Sparkles, Network, Table2,
  AlertTriangle, Layers, ChevronRight, Search,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { TabularGraphNode } from "@/components/tabular/TabularLatentGraph";

const TabularLatentGraph = dynamic(
  () => import("@/components/tabular/TabularLatentGraph"),
  { ssr: false, loading: () => <LoadingPane /> },
);

function LoadingPane() {
  return (
    <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
      <Loader2 className="h-4 w-4 animate-spin mr-2" />Loading graph...
    </div>
  );
}

const API = "/api";

interface UploadResult {
  session_id: string;
  filename: string;
  columns: string[];
  row_count: number;
  preview: Record<string, unknown>[];
  suggested_id_column: string | null;
}

interface SeparatingVar {
  field: string;
  score: number;
  kind: string;
}

interface DemoPoint {
  id: string;
  x: number;
  y: number;
  color: string;
  cluster: number;
  outlier_score: number;
  heading: string;
  metadata: Record<string, unknown>;
}

interface DemoResponse {
  summary: Record<string, unknown>;
  method: string;
  color_field: string | null;
  color_values: string[];
  separating_variables: SeparatingVar[];
  n_clusters: number;
  silhouette: number | null;
  points: DemoPoint[];
  graph: { nodes: TabularGraphNode[]; edges: { source: string; target: string; weight: number }[] };
}

const METHODS = ["umap", "pca", "tsne"] as const;

export default function TabularPage() {
  const fileRef = useRef<HTMLInputElement>(null);

  const [upload, setUpload] = useState<UploadResult | null>(null);
  const [idColumn, setIdColumn] = useState("");
  const [pkColumn, setPkColumn] = useState("");
  const [colorField, setColorField] = useState("");
  const [method, setMethod] = useState<string>("umap");
  const [sample, setSample] = useState(1200);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [demo, setDemo] = useState<DemoResponse | null>(null);
  const [hovered, setHovered] = useState<TabularGraphNode | null>(null);
  const [selected, setSelected] = useState<TabularGraphNode | null>(null);

  const handleUpload = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    setDemo(null);
    setSelected(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${API}/embeddings/tabular/upload`, { method: "POST", body: fd });
      if (!res.ok) throw new Error(await res.text());
      const data: UploadResult = await res.json();
      setUpload(data);
      setIdColumn(data.suggested_id_column || data.columns[0] || "");
      setPkColumn(data.suggested_id_column || data.columns[0] || "");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setLoading(false);
    }
  }, []);

  const runDemo = useCallback(async (opts?: { useDefault?: boolean }) => {
    setLoading(true);
    setError(null);
    try {
      const body: Record<string, unknown> = {
        method,
        sample,
        color_field: colorField || null,
        graph_k: 5,
        graph_threshold: 0.25,
      };
      if (opts?.useDefault) {
        body.use_default = true;
      } else if (upload) {
        body.session_id = upload.session_id;
        body.id_column = idColumn;
        body.pk_column = pkColumn || idColumn;
        body.rebuild = true;
      } else {
        body.use_default = true;
      }

      const res = await fetch(`${API}/embeddings/tabular/demo`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(await res.text());
      const data: DemoResponse = await res.json();
      setDemo(data);
      if (!colorField && data.color_field) setColorField(data.color_field);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Demo failed");
    } finally {
      setLoading(false);
    }
  }, [upload, idColumn, pkColumn, colorField, method, sample]);

  const recolor = useCallback(async (field: string) => {
    setColorField(field);
    if (!demo) return;
    setLoading(true);
    try {
      const body: Record<string, unknown> = {
        method,
        sample,
        color_field: field,
        graph_k: 5,
        graph_threshold: 0.25,
        rebuild: false,
      };
      if (upload) {
        body.session_id = upload.session_id;
        body.id_column = idColumn;
        body.pk_column = pkColumn || idColumn;
      } else {
        body.use_default = true;
      }
      const res = await fetch(`${API}/embeddings/tabular/demo`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(await res.text());
      setDemo(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Recolor failed");
    } finally {
      setLoading(false);
    }
  }, [demo, upload, idColumn, pkColumn, method, sample]);

  const graphNodes: TabularGraphNode[] = demo
    ? demo.graph.nodes.map((n) => {
        const pt = demo.points.find((p) => p.id === n.id);
        return { ...n, x: pt?.x, y: pt?.y, color: pt?.color ?? n.color };
      })
    : [];

  const fields = upload?.columns ?? (demo?.points[0] ? Object.keys(demo.points[0].metadata).filter((k) => !k.startsWith("_")) : []);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col">
      <header className="border-b border-zinc-800 px-4 py-3 flex items-center gap-4 shrink-0">
        <div className="flex items-center gap-2">
          <Network className="h-5 w-5 text-violet-400" />
          <h1 className="font-semibold">TabBERT Latent Explorer</h1>
        </div>
        <span className="text-xs text-zinc-500 ml-auto">Anomaly-aware tabular embeddings</span>
      </header>

      <div className="flex flex-1 min-h-0">
        {/* Left panel — upload & controls */}
        <aside className="w-80 border-r border-zinc-800 p-4 flex flex-col gap-4 overflow-y-auto shrink-0">
          <section>
            <h2 className="text-xs uppercase tracking-wider text-zinc-500 mb-2">Data</h2>
            <input
              ref={fileRef}
              type="file"
              accept=".csv,.xlsx,.xls"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleUpload(f);
              }}
            />
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              className="w-full flex items-center justify-center gap-2 rounded-lg border border-dashed border-zinc-700 px-3 py-3 text-sm hover:border-violet-500 hover:bg-zinc-900 transition"
            >
              <Upload className="h-4 w-4" />
              Upload Excel / CSV
            </button>
            <button
              type="button"
              onClick={() => runDemo({ useDefault: true })}
              className="w-full mt-2 flex items-center justify-center gap-2 rounded-lg bg-zinc-800 px-3 py-2 text-sm hover:bg-zinc-700 transition"
            >
              <Table2 className="h-4 w-4" />
              Try sample dataset
            </button>
            {upload && (
              <p className="mt-2 text-xs text-zinc-400 truncate" title={upload.filename}>
                {upload.filename} · {upload.row_count.toLocaleString()} rows
              </p>
            )}
          </section>

          {upload && (
            <section className="space-y-2">
              <h2 className="text-xs uppercase tracking-wider text-zinc-500">Keys</h2>
              <label className="block text-xs text-zinc-400">Row ID (embedding key)</label>
              <select
                value={idColumn}
                onChange={(e) => setIdColumn(e.target.value)}
                className="w-full rounded bg-zinc-900 border border-zinc-700 px-2 py-1.5 text-sm"
              >
                {upload.columns.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
              <label className="block text-xs text-zinc-400 mt-2">Display PK (hover label)</label>
              <select
                value={pkColumn}
                onChange={(e) => setPkColumn(e.target.value)}
                className="w-full rounded bg-zinc-900 border border-zinc-700 px-2 py-1.5 text-sm"
              >
                {upload.columns.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </section>
          )}

          <section className="space-y-2">
            <h2 className="text-xs uppercase tracking-wider text-zinc-500">Projection</h2>
            <div className="flex gap-1">
              {METHODS.map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setMethod(m)}
                  className={cn(
                    "flex-1 rounded px-2 py-1 text-xs uppercase",
                    method === m ? "bg-violet-600 text-white" : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700",
                  )}
                >
                  {m}
                </button>
              ))}
            </div>
            <label className="block text-xs text-zinc-400">Max rows {sample}</label>
            <input
              type="range"
              min={200}
              max={3000}
              step={100}
              value={sample}
              onChange={(e) => setSample(Number(e.target.value))}
              className="w-full"
            />
          </section>

          <button
            type="button"
            disabled={loading || (!upload && !demo)}
            onClick={() => runDemo()}
            className="flex items-center justify-center gap-2 rounded-lg bg-violet-600 px-3 py-2.5 text-sm font-medium hover:bg-violet-500 disabled:opacity-50 transition"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
            Build & visualize
          </button>

          {demo?.separating_variables && demo.separating_variables.length > 0 && (
            <section>
              <h2 className="text-xs uppercase tracking-wider text-zinc-500 mb-2 flex items-center gap-1">
                <Layers className="h-3.5 w-3.5" /> Top separating variables
              </h2>
              <ul className="space-y-1">
                {demo.separating_variables.map((v) => (
                  <li key={v.field}>
                    <button
                      type="button"
                      onClick={() => recolor(v.field)}
                      className={cn(
                        "w-full flex items-center justify-between rounded px-2 py-1.5 text-left text-xs hover:bg-zinc-800",
                        colorField === v.field && "bg-zinc-800 ring-1 ring-violet-500",
                      )}
                    >
                      <span className="truncate">{v.field}</span>
                      <span className="text-zinc-500 shrink-0 ml-1">{v.score.toFixed(3)}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {demo && (
            <section className="text-xs text-zinc-500 space-y-1">
              <p>{String(demo.summary.n_embedded)} points · {String(demo.summary.embed_backend)} embed</p>
              <p>{demo.n_clusters} clusters · silhouette {demo.silhouette ?? "—"}</p>
            </section>
          )}

          {error && (
            <p className="text-xs text-red-400 break-words">{error}</p>
          )}
        </aside>

        {/* Center — graph */}
        <main className="flex-1 flex flex-col min-w-0">
          <div className="px-4 py-2 border-b border-zinc-800 flex items-center gap-3 text-sm">
            <span className="text-zinc-500">Color by</span>
            <select
              value={colorField}
              onChange={(e) => recolor(e.target.value)}
              disabled={!demo}
              className="rounded bg-zinc-900 border border-zinc-700 px-2 py-1 text-sm disabled:opacity-50"
            >
              <option value="">—</option>
              {fields.map((f) => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
            {demo?.color_values && (
              <div className="flex flex-wrap gap-1 ml-2 max-w-md overflow-hidden">
                {demo.color_values.slice(0, 8).map((v) => (
                  <span key={v} className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 truncate max-w-[80px]" title={v}>
                    {v}
                  </span>
                ))}
                {demo.color_values.length > 8 && (
                  <span className="text-[10px] text-zinc-600">+{demo.color_values.length - 8}</span>
                )}
              </div>
            )}
            <span className="ml-auto text-xs text-zinc-600">Scroll zoom · drag pan · click detail</span>
          </div>

          <div className="flex-1 relative min-h-[400px]">
            {demo ? (
              <TabularLatentGraph
                nodes={graphNodes}
                edges={demo.graph.edges}
                colorValues={demo.color_values}
                onSelect={setSelected}
                onHover={setHovered}
              />
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-zinc-500 gap-3">
                <Network className="h-12 w-12 opacity-30" />
                <p className="text-sm">Upload a table or load the sample to explore latent space</p>
              </div>
            )}
          </div>

          {/* Cluster summary */}
          {demo && demo.n_clusters > 0 && (
            <ClusterSummary points={demo.points} nClusters={demo.n_clusters} />
          )}

          {/* Hover strip */}
          {hovered && !selected && (
            <div className="border-t border-zinc-800 px-4 py-2 text-xs flex gap-4 overflow-x-auto">
              <span className="text-violet-400 font-mono shrink-0">{hovered.heading}</span>
              {hovered.metadata && Object.entries(hovered.metadata).filter(([k]) => !k.startsWith("_")).slice(0, 6).map(([k, v]) => (
                <span key={k} className="text-zinc-400 shrink-0"><span className="text-zinc-600">{k}:</span> {String(v)}</span>
              ))}
              {(hovered.outlier_score ?? 0) > 0.35 && (
                <span className="flex items-center gap-1 text-amber-400 shrink-0">
                  <AlertTriangle className="h-3 w-3" /> outlier {hovered.outlier_score?.toFixed(3)}
                </span>
              )}
            </div>
          )}
        </main>

        {/* Right — detail panel */}
        <aside className="w-96 border-l border-zinc-800 flex flex-col shrink-0">
          <div className="px-4 py-3 border-b border-zinc-800 flex items-center gap-2">
            <ChevronRight className="h-4 w-4 text-zinc-600" />
            <h2 className="text-sm font-medium">Row detail</h2>
          </div>
          <div className="flex-1 overflow-y-auto p-4">
            {selected ? (
              <div className="space-y-3">
                <div>
                  <p className="text-xs text-zinc-500">ID</p>
                  <p className="font-mono text-sm text-violet-300 break-all">{selected.id}</p>
                </div>
                <div>
                  <p className="text-xs text-zinc-500">Label</p>
                  <p className="text-sm">{selected.heading}</p>
                </div>
                <Link
                  href={`/diff?pk=${encodeURIComponent(selected.id)}`}
                  className="inline-flex items-center gap-1.5 text-xs text-violet-400 hover:text-violet-300"
                >
                  <Search className="h-3 w-3" /> Investigate in Diff
                </Link>
                {(selected.outlier_score ?? 0) > 0.3 && (
                  <div className="rounded-lg bg-amber-950/40 border border-amber-800/50 px-3 py-2 text-xs text-amber-200 flex gap-2">
                    <AlertTriangle className="h-4 w-4 shrink-0" />
                    Anomaly score {selected.outlier_score?.toFixed(4)} — far from local neighborhood in embedding space
                  </div>
                )}
                <table className="w-full text-xs">
                  <tbody>
                    {selected.metadata && Object.entries(selected.metadata)
                      .filter(([k]) => !k.startsWith("_"))
                      .map(([k, v]) => (
                        <tr key={k} className="border-b border-zinc-800/80">
                          <td className="py-1.5 pr-2 text-zinc-500 align-top whitespace-nowrap">{k}</td>
                          <td className="py-1.5 text-zinc-200 break-all">{String(v)}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-sm text-zinc-600">Click a node to inspect the full row</p>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}

function ClusterSummary({ points, nClusters }: { points: DemoPoint[]; nClusters: number }) {
  const [open, setOpen] = useState(false);

  const clusters = useMemo(() => {
    const map = new Map<number, { count: number; outlierSum: number; colors: Map<string, number> }>();
    for (const p of points) {
      let c = map.get(p.cluster);
      if (!c) {
        c = { count: 0, outlierSum: 0, colors: new Map() };
        map.set(p.cluster, c);
      }
      c.count++;
      c.outlierSum += p.outlier_score;
      c.colors.set(p.color, (c.colors.get(p.color) ?? 0) + 1);
    }
    return Array.from(map.entries())
      .map(([id, c]) => {
        let dominant = "";
        let maxCount = 0;
        for (const [color, count] of c.colors) {
          if (count > maxCount) { dominant = color; maxCount = count; }
        }
        return {
          id,
          count: c.count,
          meanOutlier: c.outlierSum / c.count,
          dominant,
          dominantPct: c.count > 0 ? maxCount / c.count : 0,
        };
      })
      .sort((a, b) => b.count - a.count);
  }, [points]);

  return (
    <div className="border-t border-zinc-800">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full px-4 py-2 text-xs text-zinc-400 hover:text-zinc-200 flex items-center gap-2"
      >
        <ChevronRight className={cn("h-3 w-3 transition-transform", open && "rotate-90")} />
        {nClusters} clusters
      </button>
      {open && (
        <div className="px-4 pb-3">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-zinc-500">
                <th className="text-left py-1 font-medium">Cluster</th>
                <th className="text-right py-1 font-medium">Count</th>
                <th className="text-right py-1 font-medium">Outlier</th>
                <th className="text-left py-1 pl-3 font-medium">Dominant</th>
              </tr>
            </thead>
            <tbody>
              {clusters.map((c) => (
                <tr key={c.id} className="border-t border-zinc-800/60">
                  <td className="py-1 font-mono text-zinc-300">{c.id === -1 ? "noise" : c.id}</td>
                  <td className="py-1 text-right tabular-nums">{c.count}</td>
                  <td className={cn("py-1 text-right tabular-nums", c.meanOutlier > 0.35 && "text-amber-400")}>
                    {c.meanOutlier.toFixed(3)}
                  </td>
                  <td className="py-1 pl-3 truncate max-w-[120px]" title={c.dominant}>
                    {c.dominant} <span className="text-zinc-600">({(c.dominantPct * 100).toFixed(0)}%)</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
