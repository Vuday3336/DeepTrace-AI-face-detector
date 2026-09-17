import type { Health, HistoryPage, ModelInfo, PredictResponse, User } from "./types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") ?? "";
export const API_PREFIX = `${BASE}/api/v1`;

/** Error carrying the backend's machine-readable code (see docs/ARCHITECTURE.md §9.2). */
export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly code: string,
    message: string,
    public readonly requestId: string | null = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export async function parseError(res: Response): Promise<ApiError> {
  try {
    const body = await res.json();
    if (body?.error?.code) {
      return new ApiError(res.status, body.error.code, body.error.message, body.error.request_id ?? null);
    }
  } catch {
    // non-JSON body (proxy error page, network layer)
  }
  return new ApiError(res.status, "HTTP_ERROR", `Request failed with status ${res.status}`);
}

async function request<T>(path: string, init: RequestInit = {}, token?: string | null): Promise<T> {
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  let res: Response;
  try {
    res = await fetch(`${API_PREFIX}${path}`, { ...init, headers });
  } catch {
    throw new ApiError(0, "NETWORK_ERROR", "Cannot reach the DeepTrace API. Is the backend running?");
  }
  if (!res.ok) throw await parseError(res);
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<Health>("/health"),
  modelInfo: () => request<ModelInfo>("/model-info"),

  predict: (file: File, options: { store: boolean; explain: boolean }, token?: string | null) => {
    const form = new FormData();
    form.append("file", file);
    const qs = new URLSearchParams({ store: String(options.store), explain: String(options.explain) });
    return request<PredictResponse>(`/predict?${qs}`, { method: "POST", body: form }, token);
  },

  register: (email: string, password: string) => request<User>("/auth/register", json({ email, password })),
  login: (email: string, password: string) =>
    request<{ access_token: string; expires_in_minutes: number }>("/auth/login", json({ email, password })),
  me: (token: string) => request<User>("/auth/me", {}, token),

  history: (token: string, limit = 20, offset = 0) =>
    request<HistoryPage>(`/history?limit=${limit}&offset=${offset}`, {}, token),
  deleteHistory: (token: string, id: string) => request<void>(`/history/${id}`, { method: "DELETE" }, token),
  historyImage: async (token: string, id: string): Promise<Blob> => {
    const res = await fetch(`${API_PREFIX}/history/${id}/image`, { headers: { Authorization: `Bearer ${token}` } });
    if (!res.ok) throw await parseError(res);
    return res.blob();
  },
};
