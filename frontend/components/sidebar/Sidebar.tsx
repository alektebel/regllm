"use client";

import { useRouter, usePathname } from "next/navigation";
import { MessageSquarePlus, ShieldCheck, GitBranch, Code2, Trash2 } from "lucide-react";
import { useConversations } from "@/hooks/useConversations";
import { cn } from "@/lib/utils";

export function Sidebar() {
  const router = useRouter();
  const pathname = usePathname();
  const { conversations, loading, createConversation, deleteConversation } =
    useConversations();

  async function handleNew() {
    router.push("/chat");
  }

  async function handleDelete(e: React.MouseEvent, id: number) {
    e.preventDefault();
    e.stopPropagation();
    await deleteConversation(id);
    if (pathname === `/chat/${id}`) router.push("/chat");
  }

  return (
    <aside className="flex flex-col w-64 border-r border-border bg-card h-full shrink-0">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <span className="font-semibold text-sm tracking-wide">RegLLM</span>
        <button
          onClick={handleNew}
          className="rounded-md p-1.5 text-muted-foreground hover:text-foreground hover:bg-accent"
          title="New conversation"
        >
          <MessageSquarePlus size={16} />
        </button>
      </div>

      {/* Nav links */}
      <div className="px-2 pt-2 pb-1 space-y-0.5">
        <button
          onClick={() => router.push("/chat")}
          className={cn(
            "w-full flex items-center gap-2.5 rounded-md px-3 py-2 text-sm font-medium transition-colors",
            pathname.startsWith("/chat")
              ? "bg-accent text-foreground"
              : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
          )}
        >
          <MessageSquarePlus size={14} />
          Chat
        </button>
        <button
          onClick={() => router.push("/compliance")}
          className={cn(
            "w-full flex items-center gap-2.5 rounded-md px-3 py-2 text-sm font-medium transition-colors",
            pathname.startsWith("/compliance")
              ? "bg-accent text-foreground"
              : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
          )}
        >
          <ShieldCheck size={14} />
          Compliance
        </button>
        <button
          onClick={() => router.push("/pipeline")}
          className={cn(
            "w-full flex items-center gap-2.5 rounded-md px-3 py-2 text-sm font-medium transition-colors",
            pathname.startsWith("/pipeline")
              ? "bg-accent text-foreground"
              : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
          )}
        >
          <GitBranch size={14} />
          Audit Pipeline
        </button>
        <button
          onClick={() => router.push("/sas")}
          className={cn(
            "w-full flex items-center gap-2.5 rounded-md px-3 py-2 text-sm font-medium transition-colors",
            pathname.startsWith("/sas")
              ? "bg-accent text-foreground"
              : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
          )}
        >
          <Code2 size={14} />
          SAS Compiler
        </button>
      </div>

      <div className="h-px bg-border mx-2 mb-1" />

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto py-2 space-y-0.5 px-2">
        {loading ? (
          <div className="px-2 py-8 text-center text-xs text-muted-foreground">
            Loading…
          </div>
        ) : conversations.length === 0 ? (
          <div className="px-2 py-8 text-center text-xs text-muted-foreground">
            No conversations yet
          </div>
        ) : (
          conversations.map((conv) => {
            const isActive = pathname === `/chat/${conv.id}`;
            return (
              <div
                key={conv.id}
                onClick={() => router.push(`/chat/${conv.id}`)}
                className={cn(
                  "group flex items-center justify-between rounded-md px-3 py-2 text-sm cursor-pointer",
                  isActive
                    ? "bg-accent text-foreground"
                    : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
                )}
              >
                <span className="truncate flex-1 pr-2">{conv.title}</span>
                <button
                  onClick={(e) => handleDelete(e, conv.id)}
                  className="opacity-0 group-hover:opacity-100 rounded p-0.5 text-muted-foreground hover:text-destructive"
                  title="Delete"
                >
                  <Trash2 size={13} />
                </button>
              </div>
            );
          })
        )}
      </div>

    </aside>
  );
}
