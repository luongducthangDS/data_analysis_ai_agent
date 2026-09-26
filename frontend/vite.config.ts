import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { nodePolyfills } from "vite-plugin-node-polyfills";

export default defineConfig({
  plugins: [react(), nodePolyfills()],
  build: {
    outDir: "../dist",
    emptyOutDir: true,
    commonjsOptions: { transformMixedEsModules: true },
  },
  optimizeDeps: {
    include: ["react-plotly.js/factory", "plotly.js-basic-dist-min"],
  },
  server: {
    port: 5173,
    proxy: { "/api": "http://localhost:8000" },
  },
});
