"use client";

import { useMemo } from "react";
import { ChevronRight, ArrowUpRight } from "lucide-react";
import { cn } from "@/lib/utils";

export interface TraceLayer {
  depth: number;
  fields: string[];
}

export interface TraceEdge {
  source: string;
  target: string;
  kind: string;
  data_step: string;
  expr: string;
}

export interface LineageTrace {
  target: string;
  found: boolean;
  ancestors: string[];
  ancestor_count: number;
  direct_predecessors: TraceEdge[];
  layers: TraceLayer[];
  depth: Record<string, number>;
  edges?: TraceEdge[];
}

interface Props {
  trace: LineageTrace | null;
  activeTarget: string;
  onTraceField: (field: string) => void;
  className?: string;
}

export default function LineageTracePanel({
  trace,
  activeTarget,
  onTraceField,
  className,
}: Props) {
  const predsByTarget = useMemo(() => {
    const out: Record<string, TraceEdge[]> = {};
    if (!trace) return out;
    const allEdges = trace.edges?.length ? trace.edges : trace.direct_predecessors;
    for (const e of allEdges) {
      const t = e.target.toUpperCase();
      out[t] ??= [];
      out[t].push(e);
    }
    return out;
  }, [trace]);

  if (!trace) {
    return (
      <div className={cn("rounded-lg border border-dashed border-border p-3 text-xs text-muted-foreground", className)}>
        Select a target field to trace its dependencies backwards.
      </div>
    );
  }

  if (!trace.found) {
    return (
      <div className={cn("rounded-lg border border-amber-500/40 bg-amber-500/5 p-3 text-xs", className)}>
        <span className="font-mono text-amber-300">{activeTarget}</span> is not
        produced or referenced in the current V3 SAS pipeline.
      </div>
    );
  }

  const layers = trace.layers ?? [];

  return (
    <div className={cn("rounded-lg border border-border bg-card p-3 flex flex-col min-h-0", className)}>
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-xs font-semibold flex items-center gap-1.5">
          <ArrowUpRight className="h-3.5 w-3.5 text-primary rotate-[-135deg]" />
          Backward trace from{" "}
          <span className="font-mono text-sky-300">{trace.target}</span>
        </h3>
        <span className="text-[10px] text-muted-foreground tabular-nums">
          {trace.ancestor_count} upstream field{trace.ancestor_count === 1 ? "" : "s"}
        </span>
      </div>

      <p className="text-[10px] text-muted-foreground mb-2">
        Click any field to re-root the trace. Depth 0 is the target; each step
        back is one more hop toward raw inputs.
      </p>

      <div className="flex-1 overflow-y-auto space-y-2 pr-1 max-h-[280px]">
        {layers.map((layer) => (
          <div key={layer.depth}>
            <div className="text-[9px] uppercase tracking-wider text-muted-foreground mb-1">
              {layer.depth === 0 ? "Target" : `Depth ${layer.depth}`}
              {layer.depth === 1 && " · direct inputs"}
            </div>
            <ul className="space-y-1">
              {layer.fields.map((field) => {
                const fu = field.toUpperCase();
                const isTarget = fu === trace.target.toUpperCase();
                const preds = predsByTarget[fu] ?? [];
                return (
                  <li key={field}>
                    <FieldRow
                      field={field}
                      isTarget={isTarget}
                      isActive={fu === activeTarget.toUpperCase()}
                      predecessors={preds}
                      onTraceField={onTraceField}
                    />
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}

function FieldRow({
  field,
  isTarget,
  isActive,
  predecessors,
  onTraceField,
}: {
  field: string;
  isTarget: boolean;
  isActive: boolean;
  predecessors: TraceEdge[];
  onTraceField: (field: string) => void;
}) {
  const hasPreds = predecessors.length > 0;
  return (
    <details className={cn(
      "rounded border text-xs",
      isTarget ? "border-sky-500/40 bg-sky-500/5" : "border-border/60 bg-muted/10",
      isActive && "ring-1 ring-primary/50",
    )}>
      <summary
        className="list-none cursor-pointer select-none px-2 py-1.5 flex items-center gap-2 hover:bg-muted/30"
        onClick={(e) => {
          if ((e.target as HTMLElement).closest("[data-trace-btn]")) return;
          if (!hasPreds) {
            e.preventDefault();
            onTraceField(field);
          }
        }}
      >
        <button
          type="button"
          data-trace-btn
          onClick={(e) => { e.stopPropagation(); onTraceField(field); }}
          className={cn(
            "font-mono text-left hover:text-primary transition-colors",
            isTarget ? "text-sky-300 font-semibold" : "text-foreground",
          )}
          title="Trace dependencies from this field"
        >
          {field}
        </button>
        {hasPreds && (
          <span className="text-[10px] text-muted-foreground ml-auto">
            {predecessors.length} input{predecessors.length === 1 ? "" : "s"}
          </span>
        )}
        {hasPreds && <ChevronRight className="h-3 w-3 text-muted-foreground shrink-0" />}
      </summary>
      {hasPreds && (
        <ul className="px-2 pb-2 space-y-1 border-t border-border/40">
          {predecessors.map((p, i) => (
            <li key={`${p.source}-${i}`} className="pt-1.5 text-[10px] font-mono">
              <div className="flex items-baseline gap-1.5 flex-wrap">
                <span className="text-muted-foreground">←</span>
                <button
                  type="button"
                  onClick={() => onTraceField(p.source)}
                  className="text-emerald-400 hover:underline"
                >
                  {p.source}
                </button>
                <span className="text-muted-foreground">
                  via {p.kind}
                  {p.data_step ? ` · ${p.data_step}` : ""}
                </span>
              </div>
              {p.expr && (
                <div className="ml-3 mt-0.5 text-muted-foreground/80 truncate" title={p.expr}>
                  {p.expr}
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
    </details>
  );
}
