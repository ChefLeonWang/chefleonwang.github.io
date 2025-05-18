import { defineConfig } from "astro/config";
import tailwindcss from "@tailwindcss/vite";
import sitemap from "@astrojs/sitemap";
import remarkToc from "remark-toc";
import remarkCollapse from "remark-collapse";
import { SITE } from "./src/config";
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// https://astro.build/config
export default defineConfig({
  site: "https://chefleonwang.github.io",
  base: "/",
  output: "static",
  integrations: [
    
    sitemap({
      filter: page => SITE.showArchives || !page.endsWith("/archives"),
    }),

  ],
  markdown: {
    remarkPlugins: [remarkToc, remarkMath,[remarkCollapse, { test: "Table of contents" }]],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      // For more themes, visit https://shiki.style/themes
      themes: { light: "min-light", dark: "night-owl" },
      wrap: true,
    },
  },
  vite: {
    plugins: [tailwindcss()],
    optimizeDeps: {
      exclude: ["@resvg/resvg-js"],
    },
  },

});
