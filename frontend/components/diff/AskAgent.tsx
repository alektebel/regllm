"use client";

import { useEffect, useState } from "react";
import {
  Send,
  Loader2,
  Sparkles,
  Wrench,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  FileText,
  GitBranch,
  CornerDownRight,
  Square,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useSSEAgent } from "@/hooks/useSSEAgent";
import type { ToolStep, FinalPayload } from "@/hooks/useSSEAgent";

export type { Citation } from "@/hooks/useSSEAgent";

// ── Component ────────────────────────────────────────────────────────────────

interface Props {
  onLineageHighlight?: (fields: string[]) => void;
  onCitations?: (cs: Citation[]) => void;
  /** Pre-filled question (e.g. from a click on the row picker). */
  initialQuestion?: string;
  className?: string;
}

type Citation = import("@/hooks/useSSEAgent").Citation;

export default function AskAgent({
  onLineageHighlight,
  onCitations,
  initialQuestion = "",
  className,
}: Props) {
  const [question, setQuestion] = useState(initialQuestion);
  const { running, steps, status, final, error, ask: doAsk, cancel } = useSSEAgent({
    onLineageHighlight,
    onCitations,
  });

  useEffect(() => {
    if (initialQuestion && initialQuestion !== question && !running) {
      setQuestion(initialQuestion);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialQuestion]);

  function ask() {
    doAsk(question);
  }

  return (
    <div className={cn("flex flex-col gap-3 h-full min-h-0", className)}>
      {/* Header */}
      <div className="flex items-center gap-2">
        <Sparkles className="h-4 w-4 text-primary" />
        <h2 className="text-sm font-semibold">Ask the agent</h2>
        {status?.backend && (
          <span className="ml-auto text-[10px] text-muted-foreground font-mono">
            {status.backend} · {status.model}
          </span>
        )}
      </div>

      {/* Input */}
      <div className="flex flex-col gap-2">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
              e.preventDefault();
              ask();
            }
          }}
          placeholder="e.g. Why does CIC_00031 have a different ECL in V3 versus V2?"
          rows={3}
          disabled={running}
          className="w-full bg-muted/30 border border-border rounded p-2 text-sm font-sans resize-none focus:outline-none focus:ring-1 focus:ring-primary"
        />
        <div className="flex items-center gap-2">
          {running ? (
            <button
              onClick={cancel}
              className="inline-flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs rounded bg-rose-500/80 hover:bg-rose-500 text-white"
            >
              <Square className="h-3 w-3" /> Stop
            </button>
          ) : (
            <button
              onClick={ask}
              disabled={!question.trim()}
              className="inline-flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs rounded bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
            >
              <Send className="h-3 w-3" /> Ask
            </button>
          )}
          <span className="text-[10px] text-muted-foreground">
            Ctrl/⌘ + Enter
          </span>
          {steps.length > 0 && (
            <span className="ml-auto text-[10px] text-muted-foreground font-mono">
              {steps.length} tool call{steps.length === 1 ? "" : "s"}
            </span>
          )}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="rounded border border-destructive/40 bg-destructive/10 p-2 text-xs text-destructive flex items-center gap-2">
          <AlertTriangle className="h-3 w-3" />
          {error}
        </div>
      )}

      {/* Trace + final answer (scrollable) */}
      <div className="flex-1 min-h-0 overflow-y-auto pr-1 space-y-2">
        {/* Status */}
        {status && status.stage === "started" && steps.length === 0 && !final && (
          <div className="text-[11px] text-muted-foreground italic flex items-center gap-1">
            <Loader2 className="h-3 w-3 animate-spin" />
            agent thinking…
          </div>
        )}

        {/* Tool steps */}
        {steps.map((s, i) => (
          <ToolStepCard key={`${s.call.id}-${i}`} step={s} />
        ))}

        {/* Final answer */}
        {final && (
          <FinalAnswer final={final} />
        )}
      </div>
    </div>
  );
}

// ── Tool step card ───────────────────────────────────────────────────────────

function ToolStepCard({ step }: { step: ToolStep }) {
  const [open, setOpen] = useState(false);
  const { call, result, pending } = step;
  const summary = result ? summariseResult(call.tool, result.result) : "running…";
  return (
    <div className="rounded border border-border bg-card/40 text-[11px]">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-1.5 px-2 py-1 text-left hover:bg-muted/30"
      >
        {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        {pending
          ? <Loader2 className="h-3 w-3 animate-spin text-amber-400" />
          : <Wrench className="h-3 w-3 text-emerald-400" />}
        <span className="font-mono">{call.tool}</span>
        <span className="text-muted-foreground truncate">{argSummary(call.args)}</span>
        <span className="ml-auto text-muted-foreground/80 truncate max-w-[40%]">{summary}</span>
      </button>
      {open && (
        <div className="border-t border-border/50 px-2 py-1.5 bg-muted/20 space-y-1.5">
          <div>
            <div className="text-muted-foreground uppercase tracking-wider text-[9px]">args</div>
            <pre className="font-mono whitespace-pre-wrap break-words text-[10px]">{JSON.stringify(call.args, null, 2)}</pre>
          </div>
          {result && (
            <div>
              <div className="text-muted-foreground uppercase tracking-wider text-[9px]">result</div>
              <pre className="font-mono whitespace-pre-wrap break-words text-[10px] max-h-72 overflow-y-auto">
                {JSON.stringify(result.result, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function argSummary(args: Record<string, unknown>): string {
  const entries = Object.entries(args).slice(0, 3).map(([k, v]) => {
    const sv = typeof v === "string" ? `"${v}"` : JSON.stringify(v);
    return `${k}=${sv}`;
  });
  return entries.join(", ");
}

function summariseResult(tool: string, res: Record<string, unknown> | { preview: string }): string {
  if (!res || typeof res !== "object") return "";
  if ("error" in res) return `error: ${String((res as { error: unknown }).error).slice(0, 60)}`;
  if ("preview" in res) return "(truncated)";
  switch (tool) {
    case "find_row":
      return (res as { found?: boolean }).found ? "row found" : "not found";
    case "find_rows_by_field_value": {
      const m = (res as { matches?: unknown[] }).matches ?? [];
      return `${m.length} match${m.length === 1 ? "" : "es"}`;
    }
    case "inspect_lineage": {
      const a = (res as { ancestors?: string[] }).ancestors ?? [];
      return `${a.length} ancestors`;
    }
    case "compute_attribution": {
      const dy = (res as { delta_y?: number | null }).delta_y;
      const sus = (res as { suspects?: string[] }).suspects ?? [];
      return `Δ=${dy !== null && dy !== undefined ? dy.toFixed?.(3) ?? dy : "—"}, top=${sus.slice(0, 2).join(", ")}`;
    }
    case "compare_sas_versions": {
      const a = (res as { added_steps?: unknown[] }).added_steps ?? [];
      const m = (res as { modified_steps?: unknown[] }).modified_steps ?? [];
      const r = (res as { removed_steps?: unknown[] }).removed_steps ?? [];
      return `${a.length} added · ${m.length} modified · ${r.length} removed`;
    }
    case "search_docs": {
      const h = (res as { hits?: unknown[] }).hits ?? [];
      return `${h.length} hits`;
    }
    case "get_field_definition": {
      const r = (res as { result?: { found?: boolean; heading?: string } }).result;
      return r?.found ? `def: ${r.heading?.slice(0, 32)}` : "no definition";
    }
    case "search_changelog": {
      const ev = (res as { evidence?: unknown[] }).evidence ?? [];
      return `${ev.length} evidence`;
    }
    default:
      return "ok";
  }
}

// ── Final answer ─────────────────────────────────────────────────────────────

function FinalAnswer({ final }: { final: FinalPayload }) {
  return (
    <div className="rounded border border-primary/40 bg-primary/5 p-3 space-y-2">
      <div className="flex items-center gap-1.5 text-xs font-semibold text-primary">
        <Sparkles className="h-3 w-3" /> Answer
      </div>
      <SimpleMarkdown text={final.answer} />
      {final.lineage_highlight && final.lineage_highlight.length > 0 && (
        <div className="text-[10px] text-muted-foreground pt-1 border-t border-border/40">
          <span className="uppercase tracking-wider">graph highlight: </span>
          <span className="font-mono">{final.lineage_highlight.join(", ")}</span>
        </div>
      )}
      {final.citations && final.citations.length > 0 && (
        <div className="flex flex-wrap gap-1 pt-1 border-t border-border/40">
          {final.citations.map((c, i) => (
            <CitationChip key={i} c={c} />
          ))}
        </div>
      )}
    </div>
  );
}

function CitationChip({ c }: { c: Citation }) {
  const Icon = c.kind === "code" ? GitBranch : c.kind === "changelog" ? CornerDownRight : FileText;
  const label =
    c.path ? `${c.path}${c.heading ? ` · ${c.heading}` : ""}`
    : c.step ? `${c.step}${c.version ? ` (${c.version})` : ""}`
    : c.heading || JSON.stringify(c).slice(0, 60);
  return (
    <span
      className="inline-flex items-center gap-1 rounded border border-border/60 bg-card/60 px-1.5 py-0.5 text-[10px] font-mono"
      title={c.quote || ""}
    >
      <Icon className="h-2.5 w-2.5 text-muted-foreground" />
      {label}
    </span>
  );
}

// ── Tiny markdown subset (headings, bold, code, lists) ──────────────────────

function SimpleMarkdown({ text }: { text: string }) {
  // Lightweight renderer — no deps. Splits on code fences first.
  const parts = text.split(/```([\s\S]*?)```/);
  return (
    <div className="text-xs leading-relaxed space-y-1.5">
      {parts.map((p, i) => {
        if (i % 2 === 1) {
          // Inside a code fence (drop optional language tag on first line)
          const lines = p.split("\n");
          const langMatch = lines[0].match(/^[a-zA-Z0-9_-]+$/);
          const code = (langMatch ? lines.slice(1) : lines).join("\n");
          return (
            <pre
              key={i}
              className="bg-muted/40 border border-border rounded p-2 overflow-x-auto whitespace-pre-wrap break-words font-mono text-[10.5px]"
            >
              {code}
            </pre>
          );
        }
        return <ProseChunk key={i} text={p} />;
      })}
    </div>
  );
}

function ProseChunk({ text }: { text: string }) {
  const lines = text.split("\n");
  const out: React.ReactNode[] = [];
  let listBuf: string[] = [];
  const flushList = (key: string) => {
    if (!listBuf.length) return;
    out.push(
      <ul key={`ul-${key}`} className="list-disc pl-5 space-y-0.5">
        {listBuf.map((li, j) => (
          <li key={j} dangerouslySetInnerHTML={{ __html: inlineMd(li) }} />
        ))}
      </ul>,
    );
    listBuf = [];
  };
  lines.forEach((raw, i) => {
    const line = raw.trimEnd();
    if (!line.trim()) {
      flushList(`${i}-blank`);
      return;
    }
    const li = line.match(/^\s*[-*]\s+(.+)$/);
    if (li) {
      listBuf.push(li[1]);
      return;
    }
    flushList(`${i}-flush`);
    const h = line.match(/^(#{1,6})\s+(.+)$/);
    if (h) {
      const lvl = Math.min(h[1].length, 4);
      const cls = lvl === 1 ? "text-sm font-semibold mt-2"
                : lvl === 2 ? "text-xs font-semibold mt-1.5"
                : "text-xs font-medium";
      out.push(<div key={i} className={cls} dangerouslySetInnerHTML={{ __html: inlineMd(h[2]) }} />);
      return;
    }
    out.push(<p key={i} dangerouslySetInnerHTML={{ __html: inlineMd(line) }} />);
  });
  flushList("final");
  return <>{out}</>;
}

function inlineMd(s: string): string {
  // Escape, then bold + code + italics
  const esc = s
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
  return esc
    .replace(/`([^`]+)`/g, '<code class="px-1 py-0.5 rounded bg-muted/50 font-mono text-[10.5px]">$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, '<em>$1</em>');
}
