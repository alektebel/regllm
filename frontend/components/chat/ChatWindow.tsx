"use client";

import { useEffect, useRef, useState } from "react";
import { Send, Square } from "lucide-react";
import { AssistantMessage } from "./AssistantMessage";
import { UserMessage } from "./UserMessage";
import { SourceDrawer } from "./SourceDrawer";
import { BackendSwitcher } from "@/components/BackendSwitcher";
import { useStream } from "@/hooks/useStream";
import { conversationsApi, type Source } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  conversationId?: number;
}

const EXAMPLES = [
  "¿Cuál es el tratamiento del riesgo de crédito contraparte bajo CRR para derivados OTC?",
  "¿Qué requisitos de capital exige Basilea IV para carteras IRBA con LGD estimada?",
  "Compara el método estándar con el IRB avanzado para el cálculo de APR.",
];

export function ChatWindow({ conversationId: initialConvId }: Props) {
  const { messages, setMessages, isStreaming, conversationId, setConversationId, sendMessage, abort, reset } =
    useStream();
  const [input, setInput] = useState("");
  const [backend, setBackend] = useState("groq");
  const [openSources, setOpenSources] = useState<Source[] | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Load conversation history when navigating to an existing conversation
  useEffect(() => {
    if (initialConvId) {
      setConversationId(initialConvId);
      conversationsApi.messages(initialConvId).then((msgs) => {
        setMessages(
          msgs.map((m, i) => ({
            id: `loaded-${i}`,
            role: m.role as "user" | "assistant",
            content: m.content,
            sources: m.sources ?? undefined,
          }))
        );
      }).catch(() => {});
    } else {
      reset();
    }
  }, [initialConvId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSubmit(e?: React.FormEvent) {
    e?.preventDefault();
    const q = input.trim();
    if (!q || isStreaming) return;
    setInput("");
    await sendMessage(q, backend);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  const isEmpty = messages.length === 0;

  return (
    <div className="flex flex-col h-full">
      {/* Message area */}
      <div className="flex-1 overflow-y-auto">
        {isEmpty ? (
          <div className="flex flex-col items-center justify-center h-full gap-8 px-4">
            <div className="text-center space-y-2">
              <h1 className="text-2xl font-semibold">RegLLM</h1>
              <p className="text-sm text-muted-foreground max-w-sm">
                Ask me anything about EBA guidelines, CRR/CRD IV, or Basel III/IV
              </p>
            </div>
            <div className="grid gap-2 w-full max-w-2xl">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  onClick={() => {
                    setInput(ex);
                    inputRef.current?.focus();
                  }}
                  className="text-left rounded-xl border border-border bg-card px-4 py-3 text-sm text-muted-foreground hover:text-foreground hover:border-primary/50 transition-colors"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto px-4 divide-y divide-border/30">
            {messages.map((msg) =>
              msg.role === "user" ? (
                <UserMessage key={msg.id} content={msg.content} />
              ) : (
                <AssistantMessage
                  key={msg.id}
                  content={msg.content}
                  sources={msg.sources}
                  streaming={msg.streaming}
                  onShowSources={setOpenSources}
                />
              )
            )}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Input area */}
      <div className="border-t border-border bg-background px-4 py-4">
        <div className="max-w-3xl mx-auto">
          <div className="flex items-end gap-2 rounded-xl border border-border bg-card px-4 py-3 focus-within:ring-1 focus-within:ring-ring">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about banking regulation…"
              rows={1}
              className="flex-1 resize-none bg-transparent text-sm outline-none placeholder:text-muted-foreground max-h-48 overflow-y-auto"
              style={{ height: "auto" }}
              onInput={(e) => {
                const el = e.currentTarget;
                el.style.height = "auto";
                el.style.height = `${el.scrollHeight}px`;
              }}
            />
            <div className="flex items-center gap-2 shrink-0">
              <BackendSwitcher value={backend} onChange={setBackend} disabled={isStreaming} />
              {isStreaming ? (
                <button
                  onClick={abort}
                  className="rounded-lg p-1.5 bg-destructive/15 text-destructive hover:bg-destructive/25"
                  title="Stop"
                >
                  <Square size={16} />
                </button>
              ) : (
                <button
                  onClick={() => handleSubmit()}
                  disabled={!input.trim()}
                  className={cn(
                    "rounded-lg p-1.5 transition-colors",
                    input.trim()
                      ? "bg-primary text-primary-foreground hover:bg-primary/90"
                      : "bg-muted text-muted-foreground cursor-not-allowed"
                  )}
                  title="Send"
                >
                  <Send size={16} />
                </button>
              )}
            </div>
          </div>
          <p className="text-center text-xs text-muted-foreground mt-2">
            RegLLM may make mistakes. Always verify with official regulatory documents.
          </p>
        </div>
      </div>

      {/* Source drawer */}
      {openSources && (
        <SourceDrawer sources={openSources} onClose={() => setOpenSources(null)} />
      )}
    </div>
  );
}
