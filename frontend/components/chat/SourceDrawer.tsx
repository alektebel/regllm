"use client";

import { X } from "lucide-react";
import type { Source } from "@/lib/api";

interface Props {
  sources: Source[];
  onClose: () => void;
}

export function SourceDrawer({ sources, onClose }: Props) {
  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/30 z-40"
        onClick={onClose}
      />

      {/* Drawer */}
      <aside className="fixed right-0 top-0 h-full w-96 bg-card border-l border-border z-50 flex flex-col shadow-2xl">
        <div className="flex items-center justify-between px-5 py-4 border-b border-border">
          <h2 className="font-semibold text-sm">Sources ({sources.length})</h2>
          <button
            onClick={onClose}
            className="rounded-md p-1 text-muted-foreground hover:text-foreground hover:bg-accent"
          >
            <X size={16} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {sources.map((src, i) => (
            <div
              key={i}
              className="rounded-lg border border-border bg-muted/30 p-4 space-y-2"
            >
              <div className="flex items-start gap-2">
                <span className="shrink-0 rounded bg-primary/20 text-primary text-xs font-bold px-1.5 py-0.5">
                  {i + 1}
                </span>
                <span className="text-xs font-medium text-foreground truncate">
                  {src.source}
                </span>
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed line-clamp-6">
                {src.text}
              </p>
            </div>
          ))}
        </div>
      </aside>
    </>
  );
}
