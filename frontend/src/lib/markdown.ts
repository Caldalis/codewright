/**
 * Markdown 渲染:markdown-it(禁 raw HTML,天然防注入)+ Shiki 代码高亮。
 * 流式期间渲染成本高且半截代码块会闪,所以约定:流式中显示纯文本,
 * 消息完成后调用 renderMarkdown 一次性出 HTML(见 conversation store)。
 *
 * Shiki 用 fine-grained core:只打包下面显式列出的语言,且全部是动态
 * import —— 首屏不背高亮引擎的体积,第一次渲染代码块时才加载。
 */

import MarkdownIt from "markdown-it";
import { createHighlighterCore, type HighlighterCore } from "shiki/core";
import { createOnigurumaEngine } from "shiki/engine/oniguruma";

const LANG_LOADERS = {
  typescript: () => import("@shikijs/langs/typescript"),
  javascript: () => import("@shikijs/langs/javascript"),
  python: () => import("@shikijs/langs/python"),
  bash: () => import("@shikijs/langs/bash"),
  json: () => import("@shikijs/langs/json"),
  html: () => import("@shikijs/langs/html"),
  css: () => import("@shikijs/langs/css"),
  vue: () => import("@shikijs/langs/vue"),
  diff: () => import("@shikijs/langs/diff"),
  markdown: () => import("@shikijs/langs/markdown"),
  toml: () => import("@shikijs/langs/toml"),
  yaml: () => import("@shikijs/langs/yaml"),
} as const;

const LANG_ALIASES: Record<string, keyof typeof LANG_LOADERS> = {
  ts: "typescript",
  js: "javascript",
  py: "python",
  sh: "bash",
  shell: "bash",
  zsh: "bash",
  jsonc: "json",
  yml: "yaml",
  md: "markdown",
};

let highlighterPromise: Promise<HighlighterCore> | null = null;

function getHighlighter(): Promise<HighlighterCore> {
  highlighterPromise ??= createHighlighterCore({
    themes: [import("@shikijs/themes/vitesse-dark")],
    langs: Object.values(LANG_LOADERS).map((load) => load()),
    engine: createOnigurumaEngine(import("shiki/wasm")),
  });
  return highlighterPromise;
}

let mdInstance: MarkdownIt | null = null;

async function getMd(): Promise<MarkdownIt> {
  if (mdInstance !== null) return mdInstance;
  const highlighter = await getHighlighter();
  mdInstance = new MarkdownIt({
    html: false, // 关键:不透传原始 HTML
    linkify: true,
    highlight(code, lang) {
      const resolved =
        lang in LANG_LOADERS
          ? lang
          : (LANG_ALIASES[lang] ?? null);
      try {
        return highlighter.codeToHtml(code, {
          lang: resolved ?? "text",
          theme: "vitesse-dark",
        });
      } catch {
        return ""; // 回退到 markdown-it 默认的 <pre><code> 转义输出
      }
    },
  });
  return mdInstance;
}

export async function renderMarkdown(text: string): Promise<string> {
  const md = await getMd();
  return md.render(text);
}
