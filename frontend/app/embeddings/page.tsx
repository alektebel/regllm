"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { ArrowLeft, Boxes, Loader2, RefreshCw, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

const Plot = dynamic(() => import("@/components/embeddings/PlotlyScatter"), {
  ssr: false,
  loading: () => (
    <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
      Loading plot engine…
    </div>
  ),
});

const API = "/api";

interface Point {
  id: string;
  path: string;
  heading: string;
  snippet: string;
  level: number;
  x: number;
  y: number;
  z?: number;
}

interface ProjectResponse {
  method: string;
  n_components: number;
  points: Point[];
}

const METHOD_LABEL: Record<string, string> = {
  pca: "PCA",
  tsne: "t-SNE",
  umap: "UMAP",
};

export default function EmbeddingsPage() {
  const [methods, setMethods] = useState<string[]>(["pca", "tsne"]);
  const [method, setMethod] = useState("pca");
  const [dims, setDims] = useState<2 | 3>(3);
  const [points, setPoints] = useState<Point[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Point | null>(null);

  useEffect(() => {
    fetch(`${API}/embeddings/methods`)
      .then((r) => r.json())
      .then((d) => d.methods && setMethods(d.methods))
      .catch(() => {});
  }, []);

  const load = useCallback(
    async (rebuild = false) => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API}/embeddings/project`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ method, n_components: dims, rebuild }),
        });
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.detail || `HTTP ${res.status}`);
        }
        const data: ProjectResponse = await res.json();
        setPoints(data.points);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setPoints([]);
      } finally {
        setLoading(false);
      }
    },
    [method, dims]
  );

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [method, dims]);

  const pathColors = useMemo(() => {
    const uniquePaths = Array.from(new Set(points.map((p) => p.path)));
    const palette = [
      "#60a5fa", "#f472b6", "#34d399", "#fbbf24",
      "#a78bfa", "#fb923c", "#22d3ee", "#f87171",
    ];
    const map = new Map<string, string>();
    uniquePaths.forEach((p, i) => map.set(p, palette[i % palette.length]));
    return map;
  }, [points]);

  return (
    <div className="flex flex-col h-screen bg-background text-foreground">
      <header className="flex items-center justify-between px-4 py-3 border-b border-border bg-card/60">
        <div className="flex items-center gap-3">
          <Boxes className="h-5 w-5 text-primary" />
          <div>
            <h1 className="text-base font-semibold">Embedding Space Visualizer</h1>
            <p className="text-xs text-muted-foreground">
              Navigate the docs/KB embedding space — PCA · t-SNE · UMAP
            </p>
          </div>
        </div>
        <Link
          href="/diff"
          className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Back to Diff Explainer
        </Link>
      </header>

      <div className="flex-1 grid grid-cols-12 gap-3 p-3 min-h-0 overflow-hidden">
        <div className="col-span-9 flex flex-col border border-border rounded bg-card/40 min-h-0">
          <div className="flex items-center gap-3 px-3 py-2 border-b border-border flex-wrap">
            <div className="flex items-center gap-1 bg-muted/40 rounded p-0.5">
              {methods.map((m) => (
                <button
                  key={m}
                  onClick={() => setMethod(m)}
                  className={cn(
                    "px-2.5 py-1 text-xs rounded transition-colors",
                    method === m
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  {METHOD_LABEL[m] || m}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-1 bg-muted/40 rounded p-0.5">
              {[2, 3].map((d) => (
                <button
                  key={d}
                  onClick={() => setDims(d as 2 | 3)}
                  className={cn(
                    "px-2.5 py-1 text-xs rounded transition-colors",
                    dims === d
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  {d}D
                </button>
              ))}
            </div>
            <button
              onClick={() => load(true)}
              disabled={loading}
              className="flex items-center gap-1.5 px-2.5 py-1 text-xs rounded bg-muted/40 hover:bg-muted/60 disabled:opacity-50"
            >
              {loading ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <RefreshCw className="h-3.5 w-3.5" />
              )}
              Recompute embeddings
            </button>
            <span className="text-xs text-muted-foreground ml-auto">
              {points.length} sections
            </span>
          </div>

          <div className="flex-1 min-h-0 relative">
            {error && (
              <div className="absolute inset-0 flex items-center justify-center text-sm text-destructive p-4 text-center">
                {error}
              </div>
            )}
            {!error && (
              <Plot
                points={points}
                dims={dims}
                pathColors={pathColors}
                onSelect={setSelected}
              />
            )}
          </div>
        </div>

        <div className="col-span-3 flex flex-col border border-border rounded bg-card/40 min-h-0 overflow-y-auto p-3">
          <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground mb-2">
            <Sparkles className="h-3.5 w-3.5" />
            Point details
          </div>
          {selected ? (
            <div className="text-xs space-y-2">
              <div className="font-mono text-[11px] text-muted-foreground break-all">
                {selected.path}
              </div>
              <div className="font-semibold">{selected.heading}</div>
              <div className="text-muted-foreground whitespace-pre-wrap">
                {selected.snippet}
              </div>
            </div>
          ) : (
            <div className="text-xs text-muted-foreground">
              Hover or click a point in the plot to see its source text.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
