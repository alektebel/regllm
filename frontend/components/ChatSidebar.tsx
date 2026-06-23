"use client";

import { Plus, Trash2, MessageSquare, PanelLeftClose } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ChatSession } from "@/app/page";

interface Props {
  chats: ChatSession[];
  activeId: string | null;
  open: boolean;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onDelete: (id: string) => void;
  onToggle: () => void;
}

export default function ChatSidebar({
  chats, activeId, open, onSelect, onCreate, onDelete, onToggle,
}: Props) {
  if (!open) return null;

  return (
    <aside className="w-64 shrink-0 border-r border-border bg-card/60 flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-3 border-b border-border">
        <span className="text-sm font-semibold">RegLLM</span>
        <div className="flex items-center gap-1">
          <button
            onClick={onCreate}
            className="p-1.5 rounded hover:bg-muted/40 text-muted-foreground hover:text-foreground"
            title="New chat"
          >
            <Plus className="h-4 w-4" />
          </button>
          <button
            onClick={onToggle}
            className="p-1.5 rounded hover:bg-muted/40 text-muted-foreground hover:text-foreground"
            title="Close sidebar"
          >
            <PanelLeftClose className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Chat list */}
      <div className="flex-1 overflow-y-auto py-2">
        {chats.length === 0 ? (
          <p className="text-xs text-muted-foreground px-3 py-4">No chats yet</p>
        ) : (
          chats.map((chat) => (
            <div
              key={chat.id}
              className={cn(
                "group flex items-center gap-2 px-3 py-2 mx-1 rounded cursor-pointer text-sm hover:bg-muted/40",
                activeId === chat.id && "bg-muted/60",
              )}
              onClick={() => onSelect(chat.id)}
            >
              <MessageSquare className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
              <span className="truncate flex-1">{chat.title}</span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(chat.id);
                }}
                className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-destructive/20 text-muted-foreground hover:text-destructive shrink-0"
                title="Delete chat"
              >
                <Trash2 className="h-3 w-3" />
              </button>
            </div>
          ))
        )}
      </div>

      {/* Footer */}
      <div className="px-3 py-2 border-t border-border text-[10px] text-muted-foreground">
        All data stored locally
      </div>
    </aside>
  );
}
