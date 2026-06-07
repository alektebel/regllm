"use client";

import { useMemo, useState } from "react";
import { GitBranch, FileCode2, GitCompare, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface ModifiedStep {
  output: string;
  body_v2?: string;
  body_v3?: string;
  writes_v2?: string[];
  writes_v3?: string[];
  reads_v2?: string[];
  reads_v3?: string[];
  writes_added?: string[];
  writes_removed?: string[];
}

interface AddedRemovedStep {
  output: string;
  body_text?: string;
  writes?: string[];
  reads?: string[];
}

export interface CodeDiffSummary {
  target?: string | null;
  added_steps: AddedRemovedStep[];
  removed_steps: AddedRemovedStep[];
  modified_steps: ModifiedStep[];
  unchanged_steps: string[];
  unified_diff: string;
}

type Tab = "v3" | "v2" | "diff";

interface Props {
  v3: string;
  v2: string;
  onV3Change: (s: string) => void;
  onV2Change: (s: string) => void;
  codeDiff?: CodeDiffSummary | null;     // populated after Explain runs
  fieldsCount: number;
  edgesCount: number;
  className?: string;
}

export default function SasEditor({
  v3, v2,
  onV3Change, onV2Change,
  codeDiff,
  fieldsCount, edgesCount,
  className,
}: Props) {
  const [tab, setTab] = useState<Tab>("v3");

  const linesV3 = useMemo(() => v3.split("\n").length, [v3]);
  const linesV2 = useMemo(() => v2.split("\n").length, [v2]);
  const codeDiffers = v3.trim() !== v2.trim();

  const modifiedCount = codeDiff?.modified_steps?.length ?? 0;
  const addedCount = codeDiff?.added_steps?.length ?? 0;
  const removedCount = codeDiff?.removed_steps?.length ?? 0;
  const totalDiffs = modifiedCount + addedCount + removedCount;

  return (
    <div className={cn("rounded-lg border border-border bg-card p-3 flex flex-col min-h-0", className)}>
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-sm font-semibold flex items-center gap-2">
          <GitBranch className="h-4 w-4 text-primary" /> SAS pipeline
        </h2>
        <span className="text-xs text-muted-foreground">
          {tab === "v3" ? `${linesV3} lines · V3` :
           tab === "v2" ? `${linesV2} lines · V2` :
           `${fieldsCount} fields · ${edgesCount} deps`}
        </span>
      </div>

      {/* Tab strip */}
      <div className="flex items-center gap-1 mb-2">
        <TabButton
          active={tab === "v3"}
          onClick={() => setTab("v3")}
          icon={<FileCode2 className="h-3.5 w-3.5" />}
          label="V3 code"
          tone="primary"
        />
        <TabButton
          active={tab === "v2"}
          onClick={() => setTab("v2")}
          icon={<FileCode2 className="h-3.5 w-3.5" />}
          label="V2 code"
          tone="muted"
          warning={codeDiffers}
        />
        <TabButton
          active={tab === "diff"}
          onClick={() => setTab("diff")}
          icon={<GitCompare className="h-3.5 w-3.5" />}
          label="Diff"
          tone="amber"
          badge={totalDiffs > 0 ? String(totalDiffs) : (codeDiffers ? "•" : undefined)}
        />
        {codeDiffers && (
          <span
            className="ml-auto text-[10px] text-amber-400 font-mono inline-flex items-center gap-1"
            title="V2 and V3 SAS differ. The Δ explanation combines data and code attribution."
          >
            <AlertCircle className="h-3 w-3" />
            V2 ≠ V3 code
          </span>
        )}
      </div>

      {/* Editor / diff view */}
      <div className="flex-1 min-h-0">
        {tab === "v3" && (
          <textarea
            value={v3}
            onChange={(e) => onV3Change(e.target.value)}
            className="w-full font-mono text-xs bg-muted/30 border border-border rounded p-2 h-[260px] resize-none focus:outline-none focus:ring-1 focus:ring-primary"
            spellCheck={false}
            placeholder="V3 SAS source — drives the lineage graph and attribution."
          />
        )}
        {tab === "v2" && (
          <textarea
            value={v2}
            onChange={(e) => onV2Change(e.target.value)}
            className="w-full font-mono text-xs bg-muted/30 border border-amber-500/40 rounded p-2 h-[260px] resize-none focus:outline-none focus:ring-1 focus:ring-amber-500"
            spellCheck={false}
            placeholder="V2 SAS source — used to compute the V2 vs V3 code diff."
          />
        )}
        {tab === "diff" && (
          <DiffView v2={v2} v3={v3} codeDiff={codeDiff} />
        )}
      </div>
    </div>
  );
}

// ── Helpers ─────────────────────────────────────────────────────────────────

interface TabBtnProps {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
  tone: "primary" | "muted" | "amber";
  badge?: string;
  warning?: boolean;
}

function TabButton({ active, onClick, icon, label, tone, badge, warning }: TabBtnProps) {
  const inactive = tone === "primary" ? "text-muted-foreground"
                 : tone === "amber"   ? "text-amber-400/60"
                 :                      "text-muted-foreground";
  const activeBg = tone === "primary" ? "bg-primary text-primary-foreground"
                 : tone === "amber"   ? "bg-amber-500/20 text-amber-200 ring-1 ring-amber-500/40"
                 :                      "bg-muted text-foreground ring-1 ring-border";
  return (
    <button
      onClick={onClick}
      className={cn(
        "px-2 py-1 rounded text-xs flex items-center gap-1 transition-colors",
        active ? activeBg : `${inactive} hover:text-foreground hover:bg-muted/50`,
      )}
    >
      {icon}
      {label}
      {warning && !active && (
        <span className="inline-block w-1 h-1 rounded-full bg-amber-400" />
      )}
      {badge && (
        <span className={cn(
          "ml-1 inline-flex items-center justify-center min-w-[1.25rem] h-4 px-1 rounded text-[10px] tabular-nums",
          active ? "bg-background/30" : "bg-amber-500/20 text-amber-300",
        )}>
          {badge}
        </span>
      )}
    </button>
  );
}

// ── Diff view ───────────────────────────────────────────────────────────────

interface DiffViewProps {
  v2: string;
  v3: string;
  codeDiff?: CodeDiffSummary | null;
}

function DiffView({ v2, v3, codeDiff }: DiffViewProps) {
  if (v2.trim() === v3.trim()) {
    return (
      <div className="h-[260px] flex items-center justify-center text-xs text-muted-foreground italic border border-dashed border-border rounded">
        V2 and V3 SAS are identical. Edit either tab to inspect a diff.
      </div>
    );
  }

  if (!codeDiff || (
    codeDiff.added_steps.length === 0 &&
    codeDiff.removed_steps.length === 0 &&
    codeDiff.modified_steps.length === 0 &&
    !codeDiff.unified_diff
  )) {
    return (
      <div className="h-[260px] flex items-center justify-center text-xs text-muted-foreground border border-dashed border-border rounded p-3 text-center">
        Click <span className="font-mono mx-1">Explain Δ</span> to compute the
        target-scoped V2 vs V3 code diff.
      </div>
    );
  }

  return (
    <div className="h-[260px] overflow-y-auto border border-border rounded bg-muted/10 text-[11px] font-mono">
      {/* Step-level summary */}
      {codeDiff.added_steps.length > 0 && (
        <DiffSection label="Added in V3" tone="emerald" steps={codeDiff.added_steps.map(s => s.output)} />
      )}
      {codeDiff.removed_steps.length > 0 && (
        <DiffSection label="Removed in V3" tone="rose" steps={codeDiff.removed_steps.map(s => s.output)} />
      )}
      {codeDiff.modified_steps.length > 0 && (
        <div className="px-2 py-1.5 border-b border-border/50">
          <div className="text-amber-400 uppercase tracking-wider text-[9px] mb-1">Modified data steps</div>
          {codeDiff.modified_steps.map((m) => (
            <details key={m.output} className="mt-1 group">
              <summary className="cursor-pointer text-foreground hover:text-amber-300 select-none">
                <span className="text-amber-400">●</span> {m.output}
                {m.writes_added && m.writes_added.length > 0 && (
                  <span className="ml-2 text-emerald-400">+{m.writes_added.join(", ")}</span>
                )}
                {m.writes_removed && m.writes_removed.length > 0 && (
                  <span className="ml-2 text-rose-400">−{m.writes_removed.join(", ")}</span>
                )}
              </summary>
              <div className="mt-1 grid grid-cols-2 gap-2 pl-4">
                <div>
                  <div className="text-rose-400 text-[9px] uppercase tracking-wider">V2</div>
                  <pre className="whitespace-pre-wrap break-words bg-rose-500/5 border border-rose-500/20 rounded p-1.5 max-h-48 overflow-y-auto">{m.body_v2 ?? "—"}</pre>
                </div>
                <div>
                  <div className="text-emerald-400 text-[9px] uppercase tracking-wider">V3</div>
                  <pre className="whitespace-pre-wrap break-words bg-emerald-500/5 border border-emerald-500/20 rounded p-1.5 max-h-48 overflow-y-auto">{m.body_v3 ?? "—"}</pre>
                </div>
              </div>
            </details>
          ))}
        </div>
      )}
      {codeDiff.unified_diff && (
        <details className="px-2 py-1.5">
          <summary className="cursor-pointer text-muted-foreground hover:text-foreground select-none uppercase tracking-wider text-[9px]">
            Unified diff
          </summary>
          <pre className="mt-1 whitespace-pre-wrap break-words text-[10px]">
            {colorizeUnifiedDiff(codeDiff.unified_diff)}
          </pre>
        </details>
      )}
    </div>
  );
}

function DiffSection({ label, tone, steps }: { label: string; tone: "emerald" | "rose"; steps: string[] }) {
  const cls = tone === "emerald" ? "text-emerald-400" : "text-rose-400";
  return (
    <div className="px-2 py-1.5 border-b border-border/50">
      <div className={`${cls} uppercase tracking-wider text-[9px] mb-0.5`}>{label}</div>
      <div className="flex flex-wrap gap-1">
        {steps.map(s => (
          <span key={s} className={`inline-block px-1.5 py-0.5 rounded bg-${tone}-500/10 ${cls} text-[10px]`}>
            {s}
          </span>
        ))}
      </div>
    </div>
  );
}

function colorizeUnifiedDiff(text: string): React.ReactNode[] {
  return text.split("\n").map((line, i) => {
    let cls = "";
    if (line.startsWith("+++") || line.startsWith("---")) cls = "text-muted-foreground";
    else if (line.startsWith("@@")) cls = "text-sky-400";
    else if (line.startsWith("+")) cls = "text-emerald-400";
    else if (line.startsWith("-")) cls = "text-rose-400";
    return <div key={i} className={cls}>{line || "\u00A0"}</div>;
  });
}
