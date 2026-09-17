import { NavLink, Outlet } from "react-router-dom";
import { api } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { useApi } from "../hooks/useApi";

const NAV = [
  { to: "/", label: "Analyze" },
  { to: "/model", label: "Model" },
  { to: "/history", label: "History" },
  { to: "/about", label: "About" },
];

function StatusDot() {
  const { data, error } = useApi(() => api.health(), []);
  const ok = data?.model_loaded && !error;
  const text = error ? "API offline" : !data ? "Checking…" : ok ? `${data.model_name} v${data.model_version}` : "Model unavailable";
  return (
    <span className="hidden items-center gap-2 text-xs text-slate-400 sm:flex" title={data?.model_error ?? error?.message}>
      <span className={`h-2 w-2 rounded-full ${ok ? "bg-real" : data || error ? "bg-fake" : "bg-slate-500"}`} />
      {text}
    </span>
  );
}

export function Layout() {
  const { user, logout } = useAuth();
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-20 border-b border-line bg-ink/85 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center gap-6 px-4 py-3">
          <NavLink to="/" className="flex items-center gap-2 font-semibold tracking-tight text-white">
            <img src="/favicon.svg" alt="" className="h-7 w-7" />
            DeepTrace
          </NavLink>
          <nav className="flex gap-1" aria-label="Main">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === "/"}
                className={({ isActive }) =>
                  `rounded-md px-3 py-1.5 text-sm transition ${isActive ? "bg-panel text-white" : "text-slate-400 hover:text-white"}`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-4">
            <StatusDot />
            {user && (
              <button onClick={logout} className="text-xs text-slate-400 hover:text-white" title={user.email}>
                Log out
              </button>
            )}
          </div>
        </div>
      </header>
      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8">
        <Outlet />
      </main>
      <footer className="border-t border-line py-6 text-center text-xs text-slate-500">
        DeepTrace is a detection aid, not proof. Never use a result as the sole basis for an accusation.
      </footer>
    </div>
  );
}
