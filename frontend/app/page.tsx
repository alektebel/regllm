"use client";

import { useCallback, useEffect, useState } from "react";
import ChatSidebar from "@/components/ChatSidebar";
import ChatView from "@/components/ChatView";

export interface ChatSession {
  id: string;
  title: string;
  createdAt: number;
  messages: ChatMessage[];
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  ts: number;
  thoughts?: string[];
  toolSteps?: ToolStep[];
  plan?: string;
  dataRequest?: DataRequest;
  citations?: Citation[];
  lineageHighlight?: string[];
}

export interface ToolStep {
  tool: string;
  args: Record<string, unknown>;
  id: string;
  result?: Record<string, unknown>;
  pending: boolean;
}

export interface DataRequest {
  reason: string;
  extractions: {
    table: string;
    fields: string[];
    filters?: string;
    cycles?: string[];
    description?: string;
  }[];
}

export interface Citation {
  kind?: string;
  path?: string;
  heading?: string;
  step?: string;
  version?: string;
  article?: string;
}

const STORAGE_KEY = "regllm_chats";

function generateId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function loadChats(): ChatSession[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveChats(chats: ChatSession[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
}

export default function Home() {
  const [chats, setChats] = useState<ChatSession[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Load from localStorage on mount
  useEffect(() => {
    const loaded = loadChats();
    setChats(loaded);
    if (loaded.length > 0) {
      setActiveId(loaded[0].id);
    }
  }, []);

  // Persist whenever chats change
  useEffect(() => {
    if (chats.length > 0) {
      saveChats(chats);
    }
  }, [chats]);

  const activeChat = chats.find((c) => c.id === activeId) ?? null;

  const createChat = useCallback(() => {
    const session: ChatSession = {
      id: generateId(),
      title: "New chat",
      createdAt: Date.now(),
      messages: [],
    };
    setChats((prev) => [session, ...prev]);
    setActiveId(session.id);
  }, []);

  const deleteChat = useCallback((id: string) => {
    setChats((prev) => {
      const next = prev.filter((c) => c.id !== id);
      saveChats(next);
      return next;
    });
    if (activeId === id) {
      setActiveId((prev) => {
        const remaining = chats.filter((c) => c.id !== id);
        return remaining.length > 0 ? remaining[0].id : null;
      });
    }
  }, [activeId, chats]);

  const updateChat = useCallback((id: string, updater: (c: ChatSession) => ChatSession) => {
    setChats((prev) => prev.map((c) => (c.id === id ? updater(c) : c)));
  }, []);

  return (
    <div className="h-screen flex overflow-hidden bg-background text-foreground">
      {/* Sidebar */}
      <ChatSidebar
        chats={chats}
        activeId={activeId}
        open={sidebarOpen}
        onSelect={setActiveId}
        onCreate={createChat}
        onDelete={deleteChat}
        onToggle={() => setSidebarOpen((v) => !v)}
      />

      {/* Main */}
      <main className="flex-1 flex flex-col min-w-0">
        {activeChat ? (
          <ChatView
            chat={activeChat}
            onUpdate={(updater) => updateChat(activeChat.id, updater)}
            sidebarOpen={sidebarOpen}
            onToggleSidebar={() => setSidebarOpen((v) => !v)}
          />
        ) : (
          <EmptyState onCreate={createChat} sidebarOpen={sidebarOpen} onToggleSidebar={() => setSidebarOpen((v) => !v)} />
        )}
      </main>
    </div>
  );
}

function EmptyState({ onCreate, sidebarOpen, onToggleSidebar }: { onCreate: () => void; sidebarOpen: boolean; onToggleSidebar: () => void }) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-6 p-8">
      {!sidebarOpen && (
        <button
          onClick={onToggleSidebar}
          className="absolute top-4 left-4 p-2 rounded hover:bg-muted/40 text-muted-foreground"
          title="Open sidebar"
        >
          <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" /></svg>
        </button>
      )}
      <div className="text-center">
        <h1 className="text-2xl font-semibold mb-2">RegLLM</h1>
        <p className="text-sm text-muted-foreground max-w-md">
          Regulatory compliance investigator for banking reporting databases.
          Ask about field discrepancies, regulation, or SAS code lineage.
        </p>
      </div>
      <button
        onClick={onCreate}
        className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm hover:bg-primary/90"
      >
        New chat
      </button>
    </div>
  );
}
