"use client";

import { useEffect, useRef, useState } from "react";
import {
  Send,
  Loader2,
  Square,
  ChevronDown,
  ChevronRight,
  Wrench,
  Upload,
  PanelLeftOpen,
  Paperclip,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type {
  ChatSession,
  ChatMessage,
  ToolStep,
  DataRequest,
  Citation,
} from "@/app/page";

const API = "/api";
const DEFAULT_SAS_SESSION = "debug_lgd";

const LLM_OPTIONS = [
  { id: "peft:olmo3-7b", label: "Olmo 3 7B SFT (PEFT local)" },
  { id: "regllm-4b-sft", label: "RegLLM 4B SFT (Ollama)" },
  { id: "qwen2.5:14b-instruct-q4_K_M", label: "Qwen2.5 14B (Ollama)" },
  { id: "regllm-olmo3-7b-sft", label: "Olmo 3 7B SFT (Ollama, tras export)" },
] as const;

const DEFAULT_LLM_MODEL = LLM_OPTIONS[0].id;

const EXAMPLE_PROMPTS = [
  {
    label: "Comparar 3 ciclos fusionados",
    text: "En debug_lgd compara CIC_00042, CIC_00043 y CIC_00044 (SW_FUSION=1, FUS_7712). ¿Valores exactos en BDD de LGD_ESTIMADA, MoC, ECL y OR_EAD? ¿Por qué 00044 tiene ECL y los otros no?",
  },
  {
    label: "Investigar ECL missing",
    text: "¿Por qué algunos ciclos tienen MoC y ECL missing? Usa trace_causal_chain y query_cycle_data con valores reales de la BDD.",
  },
  {
    label: "Lineage LGD → ECL",
    text: "Traza LGD_ESTIMADA desde la tabla de entrada hasta ECL: pasos del pipeline, fórmulas (expr) y un ciclo de ejemplo con query_cycle_data.",
  },
];

interface Props {
  chat: ChatSession;
  onUpdate: (updater: (c: ChatSession) => ChatSession) => void;
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
}

export default function ChatView({ chat, onUpdate, sidebarOpen, onToggleSidebar }: Props) {
  const [input, setInput] = useState("");
  const [llmModel, setLlmModel] = useState(DEFAULT_LLM_MODEL);
  const [running, setRunning] = useState(false);
  const [streamingTools, setStreamingTools] = useState<ToolStep[]>([]);
  const [streamingText, setStreamingText] = useState("");
  const [streamingThoughts, setStreamingThoughts] = useState<string[]>([]);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chat.messages, streamingText, streamingTools, streamingThoughts]);

  // Focus input on chat switch
  useEffect(() => {
    inputRef.current?.focus();
  }, [chat.id]);

  async function send(message?: string) {
    const q = (message ?? input).trim();
    if (!q || running) return;
    if (!message) setInput("");
    setRunning(true);
    setStreamingTools([]);
    setStreamingText("");
    setStreamingThoughts([]);

    // Add user message
    const userMsg: ChatMessage = { role: "user", content: q, ts: Date.now() };
    onUpdate((c) => {
      const title = c.messages.length === 0 ? q.slice(0, 50) : c.title;
      return { ...c, title, messages: [...c.messages, userMsg] };
    });

    const ctrl = new AbortController();
    abortRef.current = ctrl;

    const tools: ToolStep[] = [];
    const thoughts: string[] = [];
    let finalText = "";
    let plan: string | undefined;
    let dataRequest: DataRequest | undefined;
    let citations: Citation[] | undefined;
    let lineageHighlight: string[] | undefined;

    try {
      const r = await fetch(`${API}/agent/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: q,
          session_id: chat.id,
          sas_session_id: DEFAULT_SAS_SESSION,
          model: llmModel,
          tool_names: [
            "trace_causal_chain",
            "query_cycle_data",
            "get_sas_formula",
            "trace_dependencies",
            "inspect_lineage",
            "search_docs",
            "search_regulation",
            "get_field_definition",
          ],
          max_iters: 15,
          temperature: 0.1,
        }),
        signal: ctrl.signal,
      });
      if (!r.ok || !r.body) throw new Error(`HTTP ${r.status}`);

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
            const ev = JSON.parse(json);
            if (ev.type === "thought") {
              thoughts.push(ev.text);
              setStreamingThoughts([...thoughts]);
            } else if (ev.type === "tool_call") {
              const step: ToolStep = {
                tool: ev.tool, args: ev.args, id: ev.id, pending: true,
              };
              tools.push(step);
              setStreamingTools([...tools]);
            } else if (ev.type === "tool_result") {
              const idx = tools.findIndex((t) => t.id === ev.id);
              if (idx >= 0) {
                tools[idx] = { ...tools[idx], result: ev.result, pending: false };
                setStreamingTools([...tools]);
              }
            } else if (ev.type === "plan") {
              plan = ev.plan;
            } else if (ev.type === "data_request") {
              dataRequest = { reason: ev.reason, extractions: ev.extractions };
            } else if (ev.type === "final") {
              finalText = ev.answer || ev.answer_raw || "";
              citations = ev.citations;
              lineageHighlight = ev.lineage_highlight;
              setStreamingText(finalText);
            } else if (ev.type === "error") {
              finalText = `Error: ${ev.error || "unknown error"}`;
              setStreamingText(finalText);
            }
          } catch { /* skip malformed */ }
        }
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        finalText = `Error: ${(e as Error).message}`;
      } else {
        finalText = "(cancelled)";
      }
    }

    // Commit assistant message
    const assistantMsg: ChatMessage = {
      role: "assistant",
      content: finalText || "(no response)",
      ts: Date.now(),
      thoughts: thoughts.length > 0 ? thoughts : undefined,
      toolSteps: tools.length > 0 ? tools : undefined,
      plan,
      dataRequest,
      citations,
      lineageHighlight,
    };
    onUpdate((c) => ({ ...c, messages: [...c.messages, assistantMsg] }));
    setRunning(false);
    setStreamingTools([]);
    setStreamingText("");
    setStreamingThoughts([]);
    abortRef.current = null;
  }

  function cancel() {
    abortRef.current?.abort();
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Top bar */}
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-border bg-card/40 shrink-0">
        {!sidebarOpen && (
          <button
            onClick={onToggleSidebar}
            className="p-1.5 rounded hover:bg-muted/40 text-muted-foreground"
            title="Open sidebar"
          >
            <PanelLeftOpen className="h-4 w-4" />
          </button>
        )}
        <span className="text-sm font-medium truncate">{chat.title}</span>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">
          {chat.messages.length === 0 && !running && (
            <div className="text-center py-8 space-y-4">
              <p className="text-sm text-muted-foreground">
                Pregunta sobre campos, normativa o código SAS del pipeline LGD.
                Verás el razonamiento y las herramientas en tiempo real.
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {EXAMPLE_PROMPTS.map((ex) => (
                  <button
                    key={ex.label}
                    onClick={() => send(ex.text)}
                    className="text-xs px-3 py-1.5 rounded-full border border-border/60 bg-card/60 hover:bg-muted/40 text-left"
                  >
                    {ex.label}
                  </button>
                ))}
              </div>
            </div>
          )}
          {chat.messages.map((msg, i) => (
            <MessageBubble key={`${msg.ts}-${i}`} msg={msg} />
          ))}

          {/* Streaming state */}
          {running && (
            <div className="flex gap-3">
              <div className="w-7 h-7 rounded-full bg-primary/20 flex items-center justify-center shrink-0 mt-0.5">
                <span className="text-xs font-bold text-primary">R</span>
              </div>
              <div className="flex-1 space-y-2 min-w-0">
                {/* Thoughts — dimmed, italic, like GPT/Claude thinking */}
                {streamingThoughts.length > 0 && (
                  <div className="space-y-1">
                    {streamingThoughts.map((t, i) => (
                      <p key={i} className="text-xs text-muted-foreground/70 italic leading-relaxed">
                        {t}
                      </p>
                    ))}
                  </div>
                )}

                {/* Tool calls */}
                {streamingTools.length > 0 && (
                  <div className="space-y-1">
                    {streamingTools.map((s, i) => (
                      <ToolStepRow key={`${s.id}-${i}`} step={s} />
                    ))}
                  </div>
                )}

                {/* Final text streaming in */}
                {streamingText ? (
                  <div className="text-sm">
                    <SimpleMarkdown text={streamingText} />
                  </div>
                ) : (
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Loader2 className="h-3 w-3 animate-spin" />
                    {streamingThoughts.length > 0 || streamingTools.length > 0
                      ? "Razonando..."
                      : "Pensando..."}
                  </div>
                )}
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input */}
      <div className="shrink-0 border-t border-border bg-card/40 px-4 py-3">
        <div className="max-w-3xl mx-auto">
          <div className="flex items-center gap-2 mb-2">
            <label htmlFor="llm-model" className="text-[10px] text-muted-foreground shrink-0">
              Modelo
            </label>
            <select
              id="llm-model"
              value={llmModel}
              disabled={running}
              onChange={(e) => setLlmModel(e.target.value as typeof DEFAULT_LLM_MODEL)}
              className="flex-1 text-xs rounded-md border border-border bg-background px-2 py-1"
            >
              {LLM_OPTIONS.map((o) => (
                <option key={o.id} value={o.id}>{o.label}</option>
              ))}
            </select>
          </div>
          <div className="flex items-end gap-2 bg-muted/30 border border-border rounded-xl px-3 py-2">
            <UploadButton sessionId={chat.id} onAutoSend={send} />
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Pregunta sobre un campo, normativa o código SAS..."
              rows={1}
              disabled={running}
              className="flex-1 bg-transparent resize-none text-sm focus:outline-none min-h-[24px] max-h-[120px]"
              style={{ height: "auto", overflow: "hidden" }}
              onInput={(e) => {
                const el = e.currentTarget;
                el.style.height = "auto";
                el.style.height = Math.min(el.scrollHeight, 120) + "px";
              }}
            />
            {running ? (
              <button onClick={cancel} className="p-1.5 rounded-lg bg-destructive/80 hover:bg-destructive text-white" title="Stop">
                <Square className="h-4 w-4" />
              </button>
            ) : (
              <button
                onClick={() => send()}
                disabled={!input.trim()}
                className="p-1.5 rounded-lg bg-primary hover:bg-primary/90 text-primary-foreground disabled:opacity-30"
                title="Send"
              >
                <Send className="h-4 w-4" />
              </button>
            )}
          </div>
          <p className="text-[10px] text-muted-foreground mt-1.5 text-center">
            LLM: {LLM_OPTIONS.find((o) => o.id === llmModel)?.label ?? llmModel} · sesión SAS: {DEFAULT_SAS_SESSION}
            {llmModel.startsWith("peft:") && (
              <> · requiere <code className="text-[9px]">serve_peft_agent.py</code></>
            )}
          </p>
        </div>
      </div>
    </div>
  );
}

// ── Message bubble ──────────────────────────────────────────────────────────

function MessageBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === "user";
  return (
    <div className={cn("flex gap-3", isUser && "justify-end")}>
      {!isUser && (
        <div className="w-7 h-7 rounded-full bg-primary/20 flex items-center justify-center shrink-0 mt-0.5">
          <span className="text-xs font-bold text-primary">R</span>
        </div>
      )}
      <div className={cn("max-w-[85%] space-y-2", isUser ? "text-right" : "")}>
        {isUser ? (
          <div className="inline-block bg-primary/15 rounded-2xl rounded-br-md px-4 py-2.5 text-sm text-left">
            {msg.content}
          </div>
        ) : (
          <div className="space-y-2">
            {/* Thoughts (collapsed by default in history) */}
            {msg.thoughts && msg.thoughts.length > 0 && (
              <ThoughtsBlock thoughts={msg.thoughts} />
            )}

            {/* Tool calls */}
            {msg.toolSteps && msg.toolSteps.length > 0 && (
              <div className="space-y-1">
                {msg.toolSteps.map((s, i) => (
                  <ToolStepRow key={`${s.id}-${i}`} step={s} />
                ))}
              </div>
            )}

            {/* Plan */}
            {msg.plan && (
              <div className="border border-primary/30 bg-primary/5 rounded-lg px-3 py-2 text-xs">
                <div className="text-[10px] uppercase tracking-wider text-primary/70 mb-1">Plan de investigación</div>
                <pre className="whitespace-pre-wrap font-mono text-[11px]">{msg.plan}</pre>
              </div>
            )}

            {/* Data request */}
            {msg.dataRequest && <DataRequestCard req={msg.dataRequest} />}

            {/* Answer */}
            <div className="text-sm">
              <SimpleMarkdown text={msg.content} />
            </div>

            {/* Citations */}
            {msg.citations && msg.citations.length > 0 && (
              <div className="flex flex-wrap gap-1 pt-1">
                {msg.citations.map((c, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center gap-1 rounded border border-border/60 bg-card/60 px-1.5 py-0.5 text-[10px] font-mono text-muted-foreground"
                  >
                    {c.kind === "regulation" ? c.article : c.path || c.step || c.heading}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
      {isUser && (
        <div className="w-7 h-7 rounded-full bg-muted flex items-center justify-center shrink-0 mt-0.5">
          <span className="text-xs font-bold text-muted-foreground">U</span>
        </div>
      )}
    </div>
  );
}

// ── Thoughts block (collapsed in history) ───────────────────────────────────

function ThoughtsBlock({ thoughts }: { thoughts: string[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="text-xs">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1.5 text-muted-foreground/60 hover:text-muted-foreground"
      >
        {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        <span className="italic">Razonamiento ({thoughts.length} paso{thoughts.length === 1 ? "" : "s"})</span>
      </button>
      {open && (
        <div className="mt-1 ml-4.5 space-y-1 border-l-2 border-border/30 pl-3">
          {thoughts.map((t, i) => (
            <p key={i} className="text-muted-foreground/60 italic leading-relaxed text-[11px]">
              {t}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Tool step ───────────────────────────────────────────────────────────────

function ToolStepRow({ step }: { step: ToolStep }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="text-[11px] border border-border/50 rounded bg-card/40">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-1.5 px-2 py-1 text-left hover:bg-muted/30"
      >
        {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        {step.pending
          ? <Loader2 className="h-3 w-3 animate-spin text-amber-400" />
          : <Wrench className="h-3 w-3 text-emerald-400" />}
        <span className="font-mono">{step.tool}</span>
        <span className="text-muted-foreground truncate">
          {Object.entries(step.args).slice(0, 2).map(([k, v]) =>
            `${k}=${typeof v === "string" ? `"${v}"` : JSON.stringify(v)}`
          ).join(", ")}
        </span>
      </button>
      {open && (
        <div className="border-t border-border/50 px-2 py-1.5 bg-muted/10 space-y-1">
          <pre className="font-mono whitespace-pre-wrap break-words text-[10px]">
            {JSON.stringify(step.args, null, 2)}
          </pre>
          {step.result && (
            <pre className="font-mono whitespace-pre-wrap break-words text-[10px] max-h-48 overflow-y-auto">
              {JSON.stringify(step.result, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

// ── Data request card ───────────────────────────────────────────────────────

function DataRequestCard({ req }: { req: DataRequest }) {
  return (
    <div className="border border-amber-500/30 bg-amber-500/5 rounded-lg px-3 py-2 text-xs space-y-2">
      <div className="flex items-center gap-2 text-amber-400">
        <Upload className="h-3.5 w-3.5" />
        <span className="font-medium">Datos necesarios</span>
      </div>
      <p className="text-muted-foreground">{req.reason}</p>
      <div className="space-y-1">
        {req.extractions.map((ext, i) => (
          <div key={i} className="border border-border/40 rounded px-2 py-1.5 bg-card/40">
            <span className="font-mono text-amber-300">{ext.table}</span>
            {ext.fields.length > 0 && (
              <span className="text-muted-foreground ml-1">
                ({ext.fields.join(", ")})
              </span>
            )}
            {ext.description && <p className="text-muted-foreground mt-0.5">{ext.description}</p>}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Upload button ───────────────────────────────────────────────────────────

function UploadButton({
  sessionId,
  onAutoSend,
}: {
  sessionId: string;
  onAutoSend?: (msg: string) => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);

  async function handleUpload(files: FileList) {
    setUploading(true);
    try {
      const fileArr = Array.from(files);
      const hasEgp = fileArr.some((f) => f.name.toLowerCase().endsWith(".egp"));
      const hasSas = fileArr.some((f) => f.name.toLowerCase().endsWith(".sas"));

      if (hasEgp || hasSas) {
        // SAS / EGP upload → session-scoped
        for (const f of fileArr) {
          const ext = f.name.toLowerCase().split(".").pop();
          if (ext !== "egp" && ext !== "sas") continue;
          const form = new FormData();
          if (ext === "egp") {
            form.append("file", f);
            form.append("session_id", sessionId);
            const r = await fetch(`${API}/agent/sas/upload-egp`, { method: "POST", body: form });
            if (!r.ok) throw new Error(`Upload failed: ${r.status}`);
            const data = await r.json();
            // Auto-send analysis message
            if (onAutoSend && data.blocks_metadata?.length) {
              const taskList = data.blocks_metadata
                .map((b: { order: number; task_name: string; block_type: string; line_count: number }) =>
                  `${b.order + 1}. ${b.task_name} (${b.block_type}, ${b.line_count} lines)`)
                .join("\n");
              onAutoSend(
                `I uploaded the SAS Enterprise Guide project "${data.source_file}" (${data.blocks_extracted} code blocks):\n${taskList}\n\nAnalyze this project: describe its structure, validate the code, and trace key field lineages.`
              );
            }
          } else {
            form.append("files", f);
            form.append("session_id", sessionId);
            const r = await fetch(`${API}/agent/sas/upload`, { method: "POST", body: form });
            if (!r.ok) throw new Error(`Upload failed: ${r.status}`);
            if (onAutoSend) {
              onAutoSend(`I uploaded the SAS file "${f.name}". Analyze the code: validate it and trace key field lineages.`);
            }
          }
        }
      } else {
        // CSV/XLSX data upload
        const form = new FormData();
        for (const f of fileArr) form.append("files", f);
        form.append("session_id", sessionId);
        const r = await fetch(`${API}/agent/data/upload`, { method: "POST", body: form });
        if (!r.ok) throw new Error(`Upload failed: ${r.status}`);
        const data = await r.json();
        alert(`Uploaded ${data.files?.length ?? 0} file(s). You can now ask the agent to analyze them.`);
      }
    } catch (e) {
      alert(`Upload error: ${(e as Error).message}`);
    } finally {
      setUploading(false);
    }
  }

  return (
    <>
      <input
        ref={fileRef}
        type="file"
        multiple
        accept=".csv,.xlsx,.xls,.egp,.sas"
        className="hidden"
        onChange={(e) => e.target.files && handleUpload(e.target.files)}
      />
      <button
        onClick={() => fileRef.current?.click()}
        disabled={uploading}
        className="p-1.5 rounded hover:bg-muted/40 text-muted-foreground hover:text-foreground disabled:opacity-50"
        title="Upload data (CSV/XLSX) or SAS code (.sas/.egp)"
      >
        {uploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Paperclip className="h-4 w-4" />}
      </button>
    </>
  );
}

// ── Simple markdown renderer ────────────────────────────────────────────────

function SimpleMarkdown({ text }: { text: string }) {
  const parts = text.split(/```([\s\S]*?)```/);
  return (
    <div className="leading-relaxed space-y-1.5">
      {parts.map((p, i) => {
        if (i % 2 === 1) {
          const lines = p.split("\n");
          const langMatch = lines[0].match(/^[a-zA-Z0-9_-]+$/);
          const code = (langMatch ? lines.slice(1) : lines).join("\n");
          return (
            <pre
              key={i}
              className="bg-muted/40 border border-border rounded-lg p-3 overflow-x-auto whitespace-pre-wrap break-words font-mono text-[11px]"
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
      <ul key={`ul-${key}`} className="list-disc pl-5 space-y-0.5 text-sm">
        {listBuf.map((li, j) => (
          <li key={j} dangerouslySetInnerHTML={{ __html: inlineMd(li) }} />
        ))}
      </ul>,
    );
    listBuf = [];
  };
  lines.forEach((raw, i) => {
    const line = raw.trimEnd();
    if (!line.trim()) { flushList(`${i}`); return; }
    const li = line.match(/^\s*[-*]\s+(.+)$/);
    if (li) { listBuf.push(li[1]); return; }
    flushList(`${i}`);
    const h = line.match(/^(#{1,6})\s+(.+)$/);
    if (h) {
      const lvl = Math.min(h[1].length, 4);
      const cls = lvl <= 2 ? "text-sm font-semibold mt-2" : "text-sm font-medium mt-1";
      out.push(<div key={i} className={cls} dangerouslySetInnerHTML={{ __html: inlineMd(h[2]) }} />);
      return;
    }
    const numbered = line.match(/^\s*\d+\.\s+(.+)$/);
    if (numbered) { listBuf.push(numbered[1]); return; }
    out.push(<p key={i} className="text-sm" dangerouslySetInnerHTML={{ __html: inlineMd(line) }} />);
  });
  flushList("final");
  return <>{out}</>;
}

function inlineMd(s: string): string {
  return s
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replace(/`([^`]+)`/g, '<code class="px-1 py-0.5 rounded bg-muted/50 font-mono text-[11px]">$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, "<em>$1</em>");
}
