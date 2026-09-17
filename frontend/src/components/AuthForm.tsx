import { useState, type FormEvent } from "react";
import { ApiError } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { ErrorAlert } from "./ui";

export function AuthForm() {
  const { login, register } = useAuth();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<ApiError | string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await (mode === "login" ? login(email, password) : register(email, password));
    } catch (err) {
      setError(err instanceof ApiError ? err : String(err));
    } finally {
      setBusy(false);
    }
  };

  const input = "w-full rounded-md border border-line bg-ink px-3 py-2 text-sm text-white placeholder:text-slate-600";
  return (
    <form onSubmit={submit} className="mx-auto max-w-sm space-y-4">
      <h2 className="text-lg font-semibold text-white">{mode === "login" ? "Log in" : "Create an account"}</h2>
      <p className="text-sm text-slate-400">
        An account lets you <em>choose</em> to save analyses. Nothing is saved unless you tick “save to history”.
      </p>
      <input className={input} type="email" required autoComplete="email" placeholder="Email" value={email}
        onChange={(e) => setEmail(e.target.value)} aria-label="Email" />
      <input className={input} type="password" required minLength={8} maxLength={64} placeholder="Password (min 8 characters)"
        autoComplete={mode === "login" ? "current-password" : "new-password"} value={password}
        onChange={(e) => setPassword(e.target.value)} aria-label="Password" />
      {error && <ErrorAlert error={error} />}
      <button type="submit" disabled={busy}
        className="w-full rounded-md bg-accent px-4 py-2 text-sm font-semibold text-ink hover:brightness-110 disabled:opacity-50">
        {busy ? "Please wait…" : mode === "login" ? "Log in" : "Create account"}
      </button>
      <button type="button" onClick={() => setMode(mode === "login" ? "register" : "login")}
        className="w-full text-center text-xs text-slate-400 hover:text-white">
        {mode === "login" ? "No account? Create one" : "Already have an account? Log in"}
      </button>
    </form>
  );
}
