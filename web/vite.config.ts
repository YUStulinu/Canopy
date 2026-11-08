import { defineConfig } from "vite";

// In development the map talks to the tile server (8710) and the change API (8720).
export default defineConfig({
  server: {
    port: 5180,
    proxy: {
      "/tiles": "http://127.0.0.1:8710",
      "/api": "http://127.0.0.1:8720",
    },
  },
});
