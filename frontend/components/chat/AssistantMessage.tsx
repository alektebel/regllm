"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Check, Copy, BookOpen } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Source } from "@/lib/api";

interface Props {
  content: string;
  sources?: Source[];
  streaming?: boolean;
  onShowSources?: (sources: Source[]) => void;
}

export function AssistantMessage({ content, sources, streaming, onShowSources }: Props) {
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    await navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

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
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
          {streaming && content && <span className="streaming-cursor" />}
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
