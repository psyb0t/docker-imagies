#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const token = process.env.IMAGIES_AUTH_TOKEN;
let url;
try {
  url = new URL(process.env.IMAGIES_URL);
  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password ||
      url.search || url.hash || (token && /[\r\n]/.test(token))) throw new Error("invalid config");
} catch {
  console.error("Configure IMAGIES_URL for your trusted Imagies server.");
  process.exit(1);
}
url.pathname = url.pathname.replace(/\/+$/, "") + "/v1/mcp";
const proxy = require.resolve("mcp-remote/dist/proxy.js");
const args = [proxy, url.href, "--transport", "http-only"];
if (token) args.push("--header", "Authorization: Bearer ${IMAGIES_AUTH_TOKEN}");
const result = spawnSync(process.execPath, args, { stdio: "inherit" });
process.exit(result.status ?? 1);
