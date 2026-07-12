import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    // build determinístico e rápido o suficiente pra rodar em cada chamada do checker
    minify: false,
    sourcemap: false,
  },
});
