"use client";

import { useCallback, useRef, useState } from "react";
import { type ChatRequest, type Source, streamChat } from "@/lib/api";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  streaming?: boolean;
}

export function useStream() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [conversationId, setConversationId] = useState<number | undefined>();
  const abortRef = useRef<(() => void) | null>(null);

  const sendMessage = useCallback(
    async (question: string, backend: string) => {
      if (isStreaming) return;

      const userMsg: Message = {
        id: `u-${Date.now()}`,
        role: "user",
        content: question,
      };
      const assistantMsg: Message = {
        id: `a-${Date.now()}`,
        role: "assistant",
        content: "",
        streaming: true,
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setIsStreaming(true);

      const body: ChatRequest = {
        question,
        backend,
        conversation_id: conversationId,
      };

      let cancelled = false;
      let reader: ReadableStreamDefaultReader<string> | null = null;

      abortRef.current = () => {
        cancelled = true;
        reader?.cancel();
      };

      try {
        const stream = streamChat(body);
        reader = stream.getReader();

        while (true) {
          const { done, value } = await reader.read();
          if (done || cancelled) break;

          try {
            const event = JSON.parse(value);

            if (event.type === "token") {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsg.id
                    ? { ...m, content: m.content + event.delta }
                    : m
                )
              );
            } else if (event.type === "sources") {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsg.id
                    ? { ...m, sources: event.sources }
                    : m
                )
              );
            } else if (event.type === "done") {
              setConversationId(event.conversation_id);
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsg.id ? { ...m, streaming: false } : m
                )
              );
            } else if (event.type === "error") {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsg.id
                    ? { ...m, content: `Error: ${event.message}`, streaming: false }
                    : m
                )
              );
            }
          } catch {
            // Non-JSON line — ignore
          }
        }
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : String(err);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsg.id
              ? { ...m, content: `Error: ${errMsg}`, streaming: false }
              : m
          )
        );
      } finally {
        setIsStreaming(false);
        abortRef.current = null;
      }
    },
    [isStreaming, conversationId]
  );

  const abort = useCallback(() => {
    abortRef.current?.();
  }, []);

  const reset = useCallback((newConvId?: number) => {
    setMessages([]);
    setConversationId(newConvId);
    setIsStreaming(false);
    abortRef.current?.();
  }, []);

  return {
    messages,
    setMessages,
    isStreaming,
    conversationId,
    setConversationId,
    sendMessage,
    abort,
    reset,
  };
}
