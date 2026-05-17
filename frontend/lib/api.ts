/**
 * Typed fetch wrapper.
 * All requests go through Next.js rewrites: /api/* → FastAPI.
 */

const BASE = "/api";

function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("token");
}

async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${BASE}${path}`, { ...options, headers });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `HTTP ${res.status}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

// ─── Auth ──────────────────────────────────────────────────────────────────────

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface UserOut {
  id: number;
  email: string;
  created_at: string;
}

export const authApi = {
  register: (email: string, password: string) =>
    request<TokenResponse>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  login: (email: string, password: string) =>
    request<TokenResponse>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  me: () => request<UserOut>("/auth/me"),
};

// ─── Conversations ─────────────────────────────────────────────────────────────

export interface ConversationOut {
  id: number;
  title: string;
  backend: string;
  created_at: string;
  updated_at: string | null;
}

export interface MessageOut {
  id: number;
  role: string;
  content: string;
  sources: Source[] | null;
  created_at: string;
}

export interface Source {
  source: string;
  text: string;
}

export const conversationsApi = {
  list: () => request<ConversationOut[]>("/conversations"),
  create: (title: string, backend: string) =>
    request<ConversationOut>("/conversations", {
      method: "POST",
      body: JSON.stringify({ title, backend }),
    }),
  messages: (id: number) =>
    request<MessageOut[]>(`/conversations/${id}/messages`),
  rename: (id: number, title: string) =>
    request<ConversationOut>(`/conversations/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    }),
  delete: (id: number) =>
    request<void>(`/conversations/${id}`, { method: "DELETE" }),
};

// ─── Chat (SSE) ───────────────────────────────────────────────────────────────

export interface ChatRequest {
  question: string;
  conversation_id?: number;
  backend: string;
}

export function streamChat(body: ChatRequest): ReadableStream<string> {
  const token = getToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  return new ReadableStream({
    async start(controller) {
      const res = await fetch(`${BASE}/chat/stream`, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
      });

      if (!res.ok || !res.body) {
        const err = await res.json().catch(() => ({ detail: "Stream error" }));
        controller.error(new Error(err.detail));
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (raw === "[DONE]") {
            controller.close();
            return;
          }
          controller.enqueue(raw);
        }
      }
      controller.close();
    },
  });
}
