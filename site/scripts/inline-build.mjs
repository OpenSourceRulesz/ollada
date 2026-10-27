// Build a single-file page from the vite output, matching the live site's
// current form (one index.html, no /assets paths to break on the /ollada/
// project page). Reads dist/index.html + dist/assets/*, inlines the JS
// bundle as a module script and the CSS as a <style> block, and writes
// dist/index.single.html. Run via: npm run build:single
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const dist = join(root, "dist");
let html = readFileSync(join(dist, "index.html"), "utf8");
const assets = readdirSync(join(dist, "assets"));
const js = assets.find((f) => f.endsWith(".js"));
const css = assets.find((f) => f.endsWith(".css"));
if (!js || !css) throw new Error("vite output missing JS/CSS bundle in dist/assets");
const jsText = readFileSync(join(dist, "assets", js), "utf8").replaceAll("</script", "<\\/script");
const cssText = readFileSync(join(dist, "assets", css), "utf8");
html = html.replace(/<script type="module"[^>]*src="[^"]*\.js"[^>]*><\/script>/, () => `<script type="module">${jsText}</script>`);
html = html.replace(/<link rel="stylesheet"[^>]*href="[^"]*\.css"[^>]*>/, () => `<style>${cssText}</style>`);
if (html.includes("/assets/")) throw new Error("inlined output still references /assets/");
writeFileSync(join(dist, "index.single.html"), html);
console.log(`single-file written: dist/index.single.html (${html.length} bytes)`);
