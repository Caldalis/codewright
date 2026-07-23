import vue from "@vitejs/plugin-vue";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

// dev 工作流:codewright serve 跑在 8765(引擎+桥),Vite 跑在 5173(热更新)。
// 代理让浏览器只面对 5173 一个源:/ws 走 WebSocket 代理,/api 走 HTTP 代理。
export default defineConfig({
  plugins: [vue(), tailwindcss()],
  optimizeDeps: {
    // Shiki 的语言/主题是运行中才动态 import 的;不预打包的话,Vite 会在
    // 第一次渲染代码块时发现新依赖并强制整页刷新,把进行中的对话状态冲掉。
    include: [
      "markdown-it",
      "shiki/core",
      "shiki/engine/oniguruma",
      "shiki/wasm",
      "@shikijs/themes/vitesse-dark",
      "@shikijs/langs/typescript",
      "@shikijs/langs/javascript",
      "@shikijs/langs/python",
      "@shikijs/langs/bash",
      "@shikijs/langs/json",
      "@shikijs/langs/html",
      "@shikijs/langs/css",
      "@shikijs/langs/vue",
      "@shikijs/langs/diff",
      "@shikijs/langs/markdown",
      "@shikijs/langs/toml",
      "@shikijs/langs/yaml",
    ],
  },
  server: {
    host: "127.0.0.1",
    proxy: {
      "/ws": { target: "ws://127.0.0.1:8765", ws: true },
      "/api": { target: "http://127.0.0.1:8765", changeOrigin: true },
    },
  },
});
