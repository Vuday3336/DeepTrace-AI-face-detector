import { useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import type { HistoryItem } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { AuthForm } from "../components/AuthForm";
import { ErrorAlert, LabelBadge, Panel, Spinner } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { formatPercent } from "../lib/verdict";

function Thumbnail({ token, item }: { token: string; item: HistoryItem }) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!item.has_image) return;
    let objectUrl: string | null = null;
    let cancelled = false;
    api
      .historyImage(token, item.id)
      .then((blob) => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setUrl(objectUrl);
      })
      .catch(() => setUrl(null));
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [token, item.id, item.has_image]);
  return (
    <div className="flex h-20 w-20 shrink-0 items-center justify-center overflow-hidden rounded-md bg-ink text-[10px] text-slate-600">
      {url ? <img src={url} alt="" className="h-full w-full object-cover" /> : item.has_image ? "…" : "image expired"}
    </div>
  );
}

function HistoryList({ token }: { token: string }) {
  const { data, error, loading, reload } = useApi(() => api.history(token, 50, 0), [token]);
  const [deleteError, setDeleteError] = useState<ApiError | null>(null);

  const remove = async (id: string) => {
    if (!window.confirm("Delete this analysis and its stored image? This cannot be undone.")) return;
    try {
      await api.deleteHistory(token, id);
      reload();
    } catch (err) {
      setDeleteError(err instanceof ApiError ? err : new ApiError(0, "UNKNOWN", String(err)));
    }
  };

  if (loading && !data) return <Spinner label="Loading history…" />;
  if (error) return <ErrorAlert error={error} />;
  if (!data?.items.length) {
    return <p className="text-sm text-slate-400">Nothing saved yet. Tick “save to my history” when analysing a photo.</p>;
  }
  return (
    <div className="space-y-3">
      {deleteError && <ErrorAlert error={deleteError} />}
      <p className="text-xs text-slate-500">{data.total} saved analyses · stored images expire automatically</p>
      {data.items.map((item) => (
        <Panel key={item.id} className="flex items-center gap-4 !p-3">
          <Thumbnail token={token} item={item} />
          <div className="min-w-0 flex-1 space-y-1">
            <div className="flex flex-wrap gap-2">
              {item.faces.map((f) => (
                <span key={f.face_index} className="flex items-center gap-2 text-xs text-slate-400">
                  <LabelBadge label={f.label} />
                  {formatPercent(f.prob_ai_generated)}
                </span>
              ))}
            </div>
            <p className="text-xs text-slate-500">
              {new Date(item.created_at).toLocaleString()} · {item.model_name} v{item.model_version}
            </p>
          </div>
          <button onClick={() => remove(item.id)} className="rounded-md border border-line px-3 py-1.5 text-xs text-slate-300 hover:border-fake hover:text-fake">
            Delete
          </button>
        </Panel>
      ))}
    </div>
  );
}

export function HistoryPage() {
  const { token, user } = useAuth();
  const health = useApi(() => api.health(), []);
  const disabled = health.data?.history === "disabled";
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold text-white">History</h1>
      {disabled ? (
        <Panel>
          <p className="text-sm text-slate-300">
            Accounts and history are turned off on this deployment, so nothing you upload is ever stored. Run the full
            version with docker-compose to enable them.
          </p>
        </Panel>
      ) : token && user ? (
        <HistoryList token={token} />
      ) : token ? (
        <Spinner label="Signing in…" />
      ) : (
        <Panel>
          <AuthForm />
        </Panel>
      )}
    </div>
  );
}
