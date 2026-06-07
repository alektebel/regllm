"use client";

import { Database, Code2, Sigma, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";

export interface CodeVsData {
  available: boolean;
  reason?: string;
  y_v2_under_v2: number | null;
  y_v3_under_v3: number | null;
  y_v2_under_v3: number | null;
  delta_total?: number;
  delta_data?: number;
  delta_code?: number;
  share_data?: number;
  share_code?: number;
}

function fmt(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  const sign = v >= 0 ? "" : "";
  return sign + v.toFixed(digits).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
}

function fmtSigned(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return (v >= 0 ? "+" : "") + v.toFixed(digits).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
}

function pct(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return `${Math.round(v * 100)}%`;
}

interface Props {
  cvd: CodeVsData | null | undefined;
  target: string;
}

export default function CodeVsDataDelta({ cvd, target }: Props) {
  if (!cvd) return null;

  if (!cvd.available) {
    return (
      <div className="rounded-lg border border-amber-500/40 bg-amber-500/5 p-3 text-xs">
        <div className="flex items-center gap-2 font-semibold text-amber-300">
          <AlertCircle className="h-3.5 w-3.5" />
          Code-vs-data decomposition unavailable
        </div>
        <p className="text-muted-foreground mt-1">
          {cvd.reason || "One or more anchor evaluations returned no value."}{" "}
          The lineage attribution below remains valid: it tells you the
          Δ <span className="font-mono">{target}</span> contributors
          assuming both rows had run under V3 SAS.
        </p>
      </div>
    );
  }

  const total = cvd.delta_total ?? 0;
  const data = cvd.delta_data ?? 0;
  const code = cvd.delta_code ?? 0;
  // Use absolute values for the bar geometry; show signed numbers in labels.
  const absTotal = Math.abs(data) + Math.abs(code);
  const dataPct = absTotal > 0 ? (Math.abs(data) / absTotal) * 100 : 0;
  const codePct = absTotal > 0 ? (Math.abs(code) / absTotal) * 100 : 0;

  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-xs font-semibold flex items-center gap-1.5">
          <Sigma className="h-3.5 w-3.5 text-primary" />
          Δ <span className="font-mono">{target}</span> = data Δ + code Δ
        </h3>
        <span className="text-[11px] text-muted-foreground tabular-nums" title="Δ total = Y(row_v3, code_v3) − Y(row_v2, code_v2)">
          total {fmtSigned(total, 3)}
        </span>
      </div>

      {/* Stacked bar */}
      <div className="h-3 w-full bg-muted/40 rounded-full overflow-hidden flex">
        <div
          className="h-full bg-sky-500/80"
          style={{ width: `${dataPct}%` }}
          title={`Data-driven Δ ≈ ${fmtSigned(data, 3)}`}
        />
        <div
          className="h-full bg-fuchsia-500/80"
          style={{ width: `${codePct}%` }}
          title={`Code-driven Δ ≈ ${fmtSigned(code, 3)}`}
        />
      </div>

      <div className="mt-2 grid grid-cols-2 gap-2 text-[11px]">
        <Card
          icon={<Database className="h-3 w-3 text-sky-400" />}
          label="Data Δ"
          delta={fmtSigned(data, 3)}
          share={pct(cvd.share_data)}
          formula={
            <>
              <span className="font-mono">Y(row_v3, code_v3) − Y(row_v2, code_v3)</span>
              <br />
              <span className="text-muted-foreground">→ how much input fields shifted under V3 code</span>
            </>
          }
          accent="sky"
        />
        <Card
          icon={<Code2 className="h-3 w-3 text-fuchsia-400" />}
          label="Code Δ"
          delta={fmtSigned(code, 3)}
          share={pct(cvd.share_code)}
          formula={
            <>
              <span className="font-mono">Y(row_v2, code_v3) − Y(row_v2, code_v2)</span>
              <br />
              <span className="text-muted-foreground">→ how much V2→V3 SAS changes alone shifted the same row</span>
            </>
          }
          accent="fuchsia"
        />
      </div>

      <div className="mt-2 text-[10px] text-muted-foreground/80 font-mono pt-1 border-t border-border/40">
        Y(v2,v2)={fmt(cvd.y_v2_under_v2, 3)} · Y(v2,v3)={fmt(cvd.y_v2_under_v3, 3)} · Y(v3,v3)={fmt(cvd.y_v3_under_v3, 3)}
      </div>
    </div>
  );
}

interface CardProps {
  icon: React.ReactNode;
  label: string;
  delta: string;
  share: string;
  formula: React.ReactNode;
  accent: "sky" | "fuchsia";
}

function Card({ icon, label, delta, share, formula, accent }: CardProps) {
  const ring = accent === "sky" ? "ring-sky-500/30 hover:ring-sky-500/60" : "ring-fuchsia-500/30 hover:ring-fuchsia-500/60";
  return (
    <details className={cn("rounded border bg-muted/20 p-2 group cursor-pointer ring-1", ring)}>
      <summary className="list-none flex items-baseline justify-between gap-2 select-none">
        <span className="flex items-center gap-1.5 font-medium">
          {icon} {label}
        </span>
        <span className="tabular-nums">
          <span className="font-semibold">{delta}</span>
          <span className="ml-1 text-muted-foreground text-[10px]">({share})</span>
        </span>
      </summary>
      <div className="mt-1.5 text-[10px] text-muted-foreground leading-relaxed">
        {formula}
      </div>
    </details>
  );
}
