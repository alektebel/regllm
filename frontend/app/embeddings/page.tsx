"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import {
  ArrowLeft,
  Boxes,
  FileSpreadsheet,
  Library,
  Loader2,
  RefreshCw,
  Sparkles,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import ColumnPicker from "@/components/embeddings/ColumnPicker";
import UploadZone from "@/components/embeddings/UploadZone";
import type { Point } from "@/components/embeddings/PlotlyScatter";

const PlotlyScatter = dynamic(
  () => import("@/components/embeddings/PlotlyScatter"),
  {
    ssr: false,
    loading: () => (
      <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
        Loading plot engine…
      </div>
    ),
  }
);

const API = "/api";

interface ColInfo {
  name: string;
  kind: "numeric" | "categorical" | "text";
  n_unique: number;
}

interface UploadResult {
  upload_id: string;
  filename: string;
  n_rows: number;
  columns: ColInfo[];
  sample: Record<string, unknown>[];
}

interface ProjectResult {
  method: string;
  n_components: number;
  points: Point[];
  id_col?: string;
}

const METHOD_LABEL: Record<string, string> = { pca: "PCA", tsne: "t-SNE", umap: "UMAP" };

// ── Controls bar (shared between tabs) ───────────────────────────────────────

function ProjectionControls({
  methods,
  method,
  dims,
  onMethod,
  onDims,
  loading,
  onCompute,
  computeLabel = "Compute",
  pointCount,
}: {
  methods: string[];
  method: string;
  dims: 2 | 3;
  onMethod: (m: string) => void;
  onDims: (d: 2 | 3) => void;
  loading: boolean;
  onCompute: () => void;
  computeLabel?: string;
  pointCount?: number;
}) {
  return (
    <div className="flex items-center gap-2 px-3 py-2 border-b border-border flex-wrap">
      <div className="flex items-center gap-1 bg-muted/40 rounded p-0.5">
        {methods.map((m) => (
          <button
            key={m}
            onClick={() => onMethod(m)}
            disabled={loading}
            className={cn(
              "px-2.5 py-1 text-xs rounded transition-colors disabled:opacity-50",
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
        {([2, 3] as const).map((d) => (
          <button
            key={d}
            onClick={() => onDims(d)}
            disabled={loading}
            className={cn(
              "px-2.5 py-1 text-xs rounded transition-colors disabled:opacity-50",
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
        onClick={onCompute}
        disabled={loading}
        className="flex items-center gap-1.5 px-2.5 py-1 text-xs rounded bg-muted/40 hover:bg-muted/60 disabled:opacity-50"
      >
        {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
        {computeLabel}
      </button>
      {pointCount !== undefined && (
        <span className="text-xs text-muted-foreground ml-auto">{pointCount} points</span>
      )}
    </div>
  );
}

// ── Detail panel ─────────────────────────────────────────────────────────────

function DetailPanel({ point }: { point: Point | null }) {
  return (
    <div className="flex flex-col border-l border-border min-h-0 overflow-y-auto p-3 w-64 shrink-0">
      <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground mb-2">
        <Sparkles className="h-3.5 w-3.5" />
        Point details
      </div>
      {point ? (
        <div className="text-xs space-y-2">
          <div className="font-mono text-[11px] text-primary break-all">{point.heading}</div>
          <div className="text-muted-foreground font-mono text-[10px]">{point.path}</div>
          {point.evolving && (
            <span className="inline-block text-[10px] px-1.5 py-0.5 rounded bg-amber-400/20 text-amber-400">
              ↺ evolving register
            </span>
          )}
          <div className="text-muted-foreground whitespace-pre-wrap leading-relaxed border-t border-border pt-2">
            {point.snippet}
          </div>
        </div>
      ) : (
        <div className="text-xs text-muted-foreground">
          Hover or click a point to see its fields.
        </div>
      )}
    </div>
  );
}

// ── Spreadsheet tab ───────────────────────────────────────────────────────────

function SpreadsheetTab({
  methods,
}: {
  methods: string[];
}) {
  const [upload, setUpload] = useState<UploadResult | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadErr, setUploadErr] = useState<string | null>(null);

  const [idCol, setIdCol] = useState("");
  const [featureCols, setFeatureCols] = useState<string[]>([]);
  const [colorCol, setColorCol] = useState("");

  const [method, setMethod] = useState("pca");
  const [dims, setDims] = useState<2 | 3>(3);
  const [result, setResult] = useState<ProjectResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Point | null>(null);

  const handleFile = async (file: File) => {
    setUploading(true);
    setUploadErr(null);
    setUpload(null);
    setResult(null);
    setError(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${API}/embeddings/upload-spreadsheet`, {
        method: "POST",
        body: fd,
      });
      if (!res.ok) {
        const b = await res.json().catch(() => ({}));
        throw new Error(b.detail || `HTTP ${res.status}`);
      }
      const data: UploadResult = await res.json();
      setUpload(data);
      const cols = data.columns;
      // Auto-pick id: first text column or first column
      const firstText = cols.find((c) => c.kind === "text") || cols[0];
      const newId = firstText?.name || "";
      setIdCol(newId);
      // Default feature: all numeric + categorical except id
      setFeatureCols(
        cols.filter((c) => c.name !== newId && (c.kind === "numeric" || c.kind === "categorical")).map((c) => c.name)
      );
      // Default color: first categorical (not id)
      const firstCat = cols.find((c) => c.kind === "categorical" && c.name !== newId);
      setColorCol(firstCat?.name || "");
    } catch (e) {
      setUploadErr(e instanceof Error ? e.message : String(e));
    } finally {
      setUploading(false);
    }
  };

  const compute = useCallback(async () => {
    if (!upload) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/embeddings/project-spreadsheet`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          upload_id: upload.upload_id,
          id_col: idCol,
          feature_cols: featureCols.length ? featureCols : null,
          color_col: colorCol || null,
          method,
          n_components: dims,
        }),
      });
      if (!res.ok) {
        const b = await res.json().catch(() => ({}));
        throw new Error(b.detail || `HTTP ${res.status}`);
      }
      const data: ProjectResult = await res.json();
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [upload, idCol, featureCols, method, dims]);

  // Re-project automatically when method, dims, or colorCol changes (if ready)
  const prevRef = useRef({ method: "", dims: 0, colorCol: "" });
  useEffect(() => {
    if (!upload || !idCol || !result) return;
    const prev = prevRef.current;
    if (prev.method === method && prev.dims === dims && prev.colorCol === colorCol) return;
    prevRef.current = { method, dims, colorCol };
    compute();
  }, [method, dims, colorCol, upload, idCol, result, compute]);

  const coloredPoints = useMemo(() => {
    if (!result) return [];
    // Server already injects colorGroup when color_col was requested
    return result.points;
  }, [result]);

  const pathColors = useMemo(() => {
    const palette = [
      "#60a5fa", "#f472b6", "#34d399", "#fbbf24",
      "#a78bfa", "#fb923c", "#22d3ee", "#f87171",
    ];
    const groups = Array.from(new Set(coloredPoints.map((p) => p.colorGroup ?? p.path)));
    const map = new Map<string, string>();
    groups.forEach((g, i) => map.set(g, palette[i % palette.length]));
    return map;
  }, [coloredPoints]);

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {!upload && !uploading && (
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="w-full max-w-md">
            {uploadErr && (
              <div className="mb-4 text-sm text-destructive border border-destructive/30 rounded p-3">
                {uploadErr}
              </div>
            )}
            <UploadZone onFile={handleFile} loading={uploading} />
          </div>
        </div>
      )}

      {uploading && (
        <div className="flex-1 flex items-center justify-center gap-3 text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" />
          <span className="text-sm">Reading file…</span>
        </div>
      )}

      {upload && !uploading && (
        <div className="flex-1 flex min-h-0">
          {/* Sidebar */}
          <div className="w-64 shrink-0 flex flex-col border-r border-border p-3 overflow-y-auto gap-4">
            <div className="flex items-center justify-between">
              <div className="text-xs">
                <div className="font-medium text-foreground truncate">{upload.filename}</div>
                <div className="text-muted-foreground">{upload.n_rows} rows · {upload.columns.length} cols</div>
              </div>
              <button
                onClick={() => { setUpload(null); setResult(null); }}
                className="text-muted-foreground hover:text-foreground"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <ColumnPicker
              columns={upload.columns}
              idCol={idCol}
              onIdColChange={(c) => { setIdCol(c); setResult(null); }}
              featureCols={featureCols}
              onFeatureColsChange={(cols) => { setFeatureCols(cols); setResult(null); }}
              colorCol={colorCol}
              onColorColChange={setColorCol}
            />

            <button
              onClick={compute}
              disabled={loading || !featureCols.length}
              className="flex items-center justify-center gap-2 py-2 text-sm rounded bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
            >
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Boxes className="h-4 w-4" />}
              Generate embeddings
            </button>

            {error && (
              <div className="text-xs text-destructive border border-destructive/30 rounded p-2">
                {error}
              </div>
            )}
          </div>

          {/* Main area */}
          <div className="flex-1 flex flex-col min-h-0">
            {result && (
              <ProjectionControls
                methods={methods}
                method={method}
                dims={dims}
                onMethod={setMethod}
                onDims={setDims}
                loading={loading}
                onCompute={compute}
                computeLabel="Recompute"
                pointCount={result.points.length}
              />
            )}

            <div className="flex flex-1 min-h-0">
              <div className="flex-1 min-h-0 relative">
                {loading && (
                  <div className="absolute inset-0 z-10 flex items-center justify-center bg-background/60 gap-3 text-muted-foreground text-sm">
                    <Loader2 className="h-5 w-5 animate-spin" />
                    Computing {method.toUpperCase()} projection…
                  </div>
                )}
                {result && !error ? (
                  <PlotlyScatter
                    points={coloredPoints}
                    dims={dims}
                    pathColors={pathColors}
                    onSelect={setSelected}
                  />
                ) : !loading ? (
                  <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground text-sm gap-2 h-full">
                    <FileSpreadsheet className="h-8 w-8 opacity-40" />
                    Configure columns and press Generate
                  </div>
                ) : null}
              </div>
              <DetailPanel point={selected} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Docs KB tab ───────────────────────────────────────────────────────────────

function DocsTab({ methods }: { methods: string[] }) {
  const [method, setMethod] = useState("pca");
  const [dims, setDims] = useState<2 | 3>(3);
  const [result, setResult] = useState<ProjectResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Point | null>(null);

  const load = useCallback(async (rebuild = false) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/embeddings/project`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ method, n_components: dims, rebuild }),
      });
      if (!res.ok) {
        const b = await res.json().catch(() => ({}));
        throw new Error(b.detail || `HTTP ${res.status}`);
      }
      setResult(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [method, dims]);

  useEffect(() => { load(); }, [load]);

  const pathColors = useMemo(() => {
    const palette = [
      "#60a5fa", "#f472b6", "#34d399", "#fbbf24",
      "#a78bfa", "#fb923c", "#22d3ee", "#f87171",
    ];
    const groups = Array.from(new Set((result?.points ?? []).map((p) => p.path)));
    const map = new Map<string, string>();
    groups.forEach((g, i) => map.set(g, palette[i % palette.length]));
    return map;
  }, [result]);

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <ProjectionControls
        methods={methods}
        method={method}
        dims={dims}
        onMethod={setMethod}
        onDims={setDims}
        loading={loading}
        onCompute={() => load(true)}
        computeLabel="Reindex & recompute"
        pointCount={result?.points.length}
      />
      <div className="flex flex-1 min-h-0">
        <div className="flex-1 min-h-0 relative">
          {loading && (
            <div className="absolute inset-0 z-10 flex items-center justify-center bg-background/60 gap-3 text-muted-foreground text-sm">
              <Loader2 className="h-5 w-5 animate-spin" />
              Computing {method.toUpperCase()} projection…
            </div>
          )}
          {error && (
            <div className="absolute inset-0 flex items-center justify-center text-sm text-destructive">
              {error}
            </div>
          )}
          {result && !error && (
            <PlotlyScatter
              points={result.points}
              dims={dims}
              pathColors={pathColors}
              onSelect={setSelected}
            />
          )}
        </div>
        <DetailPanel point={selected} />
      </div>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function EmbeddingsPage() {
  const [methods, setMethods] = useState<string[]>(["pca", "tsne"]);
  const [tab, setTab] = useState<"spreadsheet" | "docs">("spreadsheet");

  useEffect(() => {
    fetch(`${API}/embeddings/methods`)
      .then((r) => r.json())
      .then((d) => d.methods && setMethods(d.methods))
      .catch(() => {});
  }, []);

  return (
    <div className="flex flex-col h-screen bg-background text-foreground">
      <header className="flex items-center justify-between px-4 py-3 border-b border-border bg-card/60 shrink-0">
        <div className="flex items-center gap-3">
          <Boxes className="h-5 w-5 text-primary" />
          <div>
            <h1 className="text-base font-semibold">Embedding Space Visualizer</h1>
            <p className="text-xs text-muted-foreground">
              Upload a spreadsheet · pick the ID field · explore in 2D / 3D
            </p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1 bg-muted/40 rounded p-0.5">
            <button
              onClick={() => setTab("spreadsheet")}
              className={cn(
                "flex items-center gap-1.5 px-2.5 py-1 text-xs rounded transition-colors",
                tab === "spreadsheet" ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"
              )}
            >
              <FileSpreadsheet className="h-3.5 w-3.5" />
              Spreadsheet
            </button>
            <button
              onClick={() => setTab("docs")}
              className={cn(
                "flex items-center gap-1.5 px-2.5 py-1 text-xs rounded transition-colors",
                tab === "docs" ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"
              )}
            >
              <Library className="h-3.5 w-3.5" />
              Docs KB
            </button>
          </div>
          <Link
            href="/diff"
            className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Diff Explainer
          </Link>
        </div>
      </header>

      {tab === "spreadsheet" ? (
        <SpreadsheetTab methods={methods} />
      ) : (
        <DocsTab methods={methods} />
      )}
    </div>
  );
}
