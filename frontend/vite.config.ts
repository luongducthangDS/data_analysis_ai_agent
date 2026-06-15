import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../dist",
    emptyOutDir: true,
    commonjsOptions: { transformMixedEsModules: true },
  },
  optimizeDeps: {
    include: ["react-plotly.js", "plotly.js"],
  },
  server: {
    port: 5173,
    proxy: { "/api": "http://localhost:8000" },
  },
});
