import { Link, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { AboutPage } from "./pages/AboutPage";
import { AnalyzePage } from "./pages/AnalyzePage";
import { HistoryPage } from "./pages/HistoryPage";
import { ModelPage } from "./pages/ModelPage";

function NotFound() {
  return (
    <div className="space-y-3">
      <h1 className="text-2xl font-semibold text-white">Page not found</h1>
      <Link to="/" className="text-accent underline">Back to the analyzer</Link>
    </div>
  );
}

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<AnalyzePage />} />
        <Route path="model" element={<ModelPage />} />
        <Route path="history" element={<HistoryPage />} />
        <Route path="about" element={<AboutPage />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
