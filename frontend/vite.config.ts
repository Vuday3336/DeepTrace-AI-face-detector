/// <reference types="vitest/config" />
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the API runs on :8000 and is proxied, so the app always calls same-origin /api/...
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // 127.0.0.1, not "localhost": Node may resolve localhost to IPv6 ::1 while uvicorn listens on IPv4.
    proxy: { "/api": { target: process.env.VITE_DEV_API_TARGET ?? "http://127.0.0.1:8000", changeOrigin: true } },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
