"use client";

import { useCallback, useRef, useState } from "react";

const API = "/api";

export type EventType = "status" | "thought" | "tool_call" | "tool_result" | "final" | "error" | "done";

export interface AgentEventBase {
  type: EventType;
  ts: number;
  [k: string]: unknown;
}

export interface ToolCall {
  iter: number;
  tool: string;
  args: Record<string, unknown>;
  id: string;
}

export interface ToolResult {
  iter: number;
  tool: string;
  id: string;
  result: Record<string, unknown> | { truncated: boolean; preview: string };
}

export interface FinalPayload {
  iter: number;
  answer: string;
  answer_raw?: string;
  lineage_highlight: string[];
  citations: Citation[];
}

export interface Citation {
  kind?: "doc" | "code" | "changelog" | string;
  path?: string;
  heading?: string;
  step?: string;
  version?: string;
  quote?: string;
}

export interface StatusPayload {
  stage: string;
  backend?: string;
  model?: string;
  question?: string;
  iters?: number;
  iter?: number;
  tools?: string[];
}

export interface ToolStep {
  call: ToolCall;
  result?: ToolResult;
  pending: boolean;
}

interface UseSSEAgentOptions {
  onLineageHighlight?: (fields: string[]) => void;
  onCitations?: (cs: Citation[]) => void;
}

interface UseSSEAgentReturn {
  running: boolean;
  steps: ToolStep[];
  status: StatusPayload | null;
  final: FinalPayload | null;
  error: string | null;
  ask: (question: string) => Promise<void>;
  cancel: () => void;
}

export function useSSEAgent(options: UseSSEAgentOptions = {}): UseSSEAgentReturn {
  const [running, setRunning] = useState(false);
  const [steps, setSteps] = useState<ToolStep[]>([]);
  const [status, setStatus] = useState<StatusPayload | null>(null);
  const [final, setFinal] = useState<FinalPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const optionsRef = useRef(options);
  optionsRef.current = options;

  const handleEvent = useCallback((ev: AgentEventBase) => {
    if (ev.type === "status") {
      setStatus(ev as unknown as StatusPayload);
    } else if (ev.type === "tool_call") {
      const c = ev as unknown as ToolCall;
      setSteps((prev) => [...prev, { call: c, result: undefined, pending: true }]);
    } else if (ev.type === "tool_result") {
      const res = ev as unknown as ToolResult;
      setSteps((prev) => prev.map((s) =>
        s.call.id === res.id ? { ...s, result: res, pending: false } : s,
      ));
    } else if (ev.type === "final") {
      const f = ev as unknown as FinalPayload;
      setFinal(f);
      optionsRef.current.onLineageHighlight?.(f.lineage_highlight ?? []);
      optionsRef.current.onCitations?.(f.citations ?? []);
    } else if (ev.type === "error") {
      setError((ev as unknown as { error?: string }).error || "agent error");
    }
  }, []);

  const ask = useCallback(async (question: string) => {
    const q = question.trim();
    if (!q || running) return;
    setRunning(true);
    setSteps([]);
    setStatus(null);
    setFinal(null);
    setError(null);

    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      const r = await fetch(`${API}/agent/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, max_iters: 8, temperature: 0.1 }),
        signal: ctrl.signal,
      });
      if (!r.ok || !r.body) {
        throw new Error(`HTTP ${r.status}`);
      }
      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const events = buf.split(/\n\n/);
        buf = events.pop() ?? "";
        for (const block of events) {
          const dataLine = block.split("\n").find((l) => l.startsWith("data:"));
          if (!dataLine) continue;
          const json = dataLine.slice(5).trim();
          if (!json) continue;
          try {
            const ev = JSON.parse(json) as AgentEventBase;
            handleEvent(ev);
          } catch {
            // ignore malformed line
          }
        }
      }
    } catch (e) {
      if ((e as Error).name === "AbortError") {
        setError("cancelled");
      } else {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  }, [running, handleEvent]);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return { running, steps, status, final, error, ask, cancel };
}
