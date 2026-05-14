"use client";

import { useCallback, useEffect, useState } from "react";
import { type ConversationOut, conversationsApi } from "@/lib/api";

export function useConversations() {
  const [conversations, setConversations] = useState<ConversationOut[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const data = await conversationsApi.list();
      setConversations(data);
    } catch {
      // Not authenticated yet — silently ignore
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const createConversation = useCallback(
    async (backend: string = "groq") => {
      const conv = await conversationsApi.create("New conversation", backend);
      setConversations((prev) => [conv, ...prev]);
      return conv;
    },
    []
  );

  const deleteConversation = useCallback(async (id: number) => {
    await conversationsApi.delete(id);
    setConversations((prev) => prev.filter((c) => c.id !== id));
  }, []);

  const renameConversation = useCallback(async (id: number, title: string) => {
    const updated = await conversationsApi.rename(id, title);
    setConversations((prev) => prev.map((c) => (c.id === id ? updated : c)));
  }, []);

  const updateTitle = useCallback((id: number, title: string) => {
    setConversations((prev) =>
      prev.map((c) => (c.id === id ? { ...c, title } : c))
    );
  }, []);

  return {
    conversations,
    loading,
    refresh,
    createConversation,
    deleteConversation,
    renameConversation,
    updateTitle,
  };
}
