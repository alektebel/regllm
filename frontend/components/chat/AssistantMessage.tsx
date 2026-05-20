"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Check, Copy, BookOpen, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Source } from "@/lib/api";

interface Props {
  content: string;
  sources?: Source[];
  streaming?: boolean;
  onShowSources?: (sources: Source[]) => void;
}

interface RegllmResponse {
  resumen: string;
  desarrollo: string;
  articulos?: string[];
  advertencias?: string | null;
}

function extractJsonField(content: string, field: string): string | null {
  const match = content.match(new RegExp(`"${field}"\\s*:\\s*"((?:[^"\\\\]|\\\\.)*)"`));
  return match ? match[1].replace(/\\n/g, "\n").replace(/\\"/g, '"') : null;
}

function extractJsonArray(content: string, field: string): string[] | null {
  const match = content.match(new RegExp(`"${field}"\\s*:\\s*\\[([^\\]]*)\\]`));
  if (!match) return null;
  return match[1].match(/"([^"]+)"/g)?.map((s) => s.replace(/"/g, "")) ?? [];
}

function parseRegllmJson(content: string): RegllmResponse | null {
  // Strip markdown code fences if present
  const cleaned = content.replace(/^```(?:json)?\n?/m, "").replace(/\n?```$/m, "").trim();
  if (!cleaned.includes('"resumen"')) return null;

  // Try strict parse first
  try {
    const parsed = JSON.parse(cleaned);
    if (parsed && typeof parsed.resumen === "string" && typeof parsed.desarrollo === "string") {
      return parsed as RegllmResponse;
    }
  } catch {
    // fall through to partial extraction
  }

  // Partial/truncated JSON — extract fields with regex
  const resumen = extractJsonField(cleaned, "resumen");
  const desarrollo = extractJsonField(cleaned, "desarrollo");
  if (!resumen || !desarrollo) return null;

  return {
    resumen,
    desarrollo,
    articulos: extractJsonArray(cleaned, "articulos") ?? [],
    advertencias: extractJsonField(cleaned, "advertencias"),
  };
}

function StructuredResponse({ data }: { data: RegllmResponse }) {
  return (
    <div className="space-y-4">
      {/* Summary */}
      <div className="rounded-lg bg-primary/5 border border-primary/20 px-4 py-3">
        <p className="text-xs font-semibold uppercase tracking-wide text-primary mb-1">Resumen</p>
        <p className="text-sm">{data.resumen}</p>
      </div>

      {/* Full explanation */}
      <div className="prose prose-sm max-w-none">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{data.desarrollo}</ReactMarkdown>
      </div>

      {/* Regulatory articles */}
      {data.articulos && data.articulos.length > 0 && (
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">
            Referencias regulatorias
          </p>
          <div className="flex flex-wrap gap-2">
            {data.articulos.map((art) => (
              <span
                key={art}
                className="inline-flex items-center rounded-md bg-muted px-2 py-1 text-xs font-medium text-muted-foreground"
              >
                {art}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Warnings */}
      {data.advertencias && (
        <div className="flex gap-2 rounded-lg border border-yellow-200 bg-yellow-50 dark:border-yellow-900 dark:bg-yellow-950/30 px-4 py-3">
          <AlertTriangle size={14} className="text-yellow-600 dark:text-yellow-500 shrink-0 mt-0.5" />
          <p className="text-xs text-yellow-800 dark:text-yellow-300">{data.advertencias}</p>
        </div>
      )}
    </div>
  );
}

export function AssistantMessage({ content, sources, streaming, onShowSources }: Props) {
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    await navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  const structured = !streaming ? parseRegllmJson(content) : null;

  return (
    <div className="group flex gap-3 py-4">
      {/* Avatar */}
      <div className="shrink-0 w-7 h-7 rounded-full bg-primary/20 flex items-center justify-center text-primary text-xs font-bold mt-0.5">
        R
      </div>

      <div className="flex-1 min-w-0 space-y-2">
        <div
          className={cn(
            "prose prose-sm max-w-none",
            streaming && !content && "streaming-cursor"
          )}
        >
          {structured ? (
            <StructuredResponse data={structured} />
          ) : (
            <>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
              {streaming && content && <span className="streaming-cursor" />}
            </>
          )}
        </div>

        {/* Action buttons */}
        {!streaming && content && (
          <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              onClick={handleCopy}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
            >
              {copied ? <Check size={12} /> : <Copy size={12} />}
              {copied ? "Copied" : "Copy"}
            </button>

            {sources && sources.length > 0 && onShowSources && (
              <button
                onClick={() => onShowSources(sources)}
                className="flex items-center gap-1 text-xs text-muted-foreground hover:text-primary"
              >
                <BookOpen size={12} />
                {sources.length} source{sources.length > 1 ? "s" : ""}
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
