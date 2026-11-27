#!/usr/bin/env node
// Headless check of a replay page with the verified lab01 GPU flags:
//   node scripts/verify_replay.cjs results/replay/PAGE.html [--updates 60]
// Confirms Chrome uses the NVIDIA renderer with GPU compositing and 2D canvas acceleration,
// that every board and its evidence render without console errors, measures step-all
// input-to-paint latency with every board on screen (late-game positions), and saves a
// screenshot. Writes results/replay/verify_<page>.json and .png.
const { spawn } = require("child_process");
const fs = require("fs"), os = require("os"), path = require("path");
const WebSocket = require("ws");

class CDP {
  constructor(ws) {
    this.ws = ws; this.id = 0; this.pending = new Map(); this.errors = [];
    ws.on("message", raw => {
      const m = JSON.parse(raw);
      if (m.id && this.pending.has(m.id)) {
        const { res, rej } = this.pending.get(m.id); this.pending.delete(m.id);
        m.error ? rej(new Error(m.error.message)) : res(m.result);
      } else if (m.method === "Runtime.exceptionThrown") {
        this.errors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
      } else if (m.method === "Runtime.consoleAPICalled" && m.params.type === "error") {
        this.errors.push(m.params.args.map(a => a.value ?? a.description).join(" "));
      }
    });
  }
  send(method, params = {}, sessionId) {
    const id = ++this.id, msg = { id, method, params };
    if (sessionId) msg.sessionId = sessionId;
    this.ws.send(JSON.stringify(msg));
    return new Promise((res, rej) => this.pending.set(id, { res, rej }));
  }
  async evaluate(expression, sessionId) {
    const r = await this.send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true }, sessionId);
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
    return r.result.value;
  }
}

(async () => {
  const page = path.resolve(process.argv[2] || "");
  if (!fs.existsSync(page)) throw new Error(`no such page: ${page}`);
  const updates = Number((process.argv.find(a => a.startsWith("--updates=")) || "--updates=60").split("=")[1]);
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "replay-verify-"));
  const chrome = spawn("/usr/bin/google-chrome", ["--headless=new", "--enable-gpu", "--use-angle=vulkan", "--no-first-run",
    "--no-default-browser-check", "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows", "--remote-debugging-address=127.0.0.1", "--remote-debugging-port=0",
    `--user-data-dir=${profile}`, "about:blank"], { stdio: "ignore" });
  const portFile = path.join(profile, "DevToolsActivePort"), t0 = Date.now();
  while (!fs.existsSync(portFile)) { if (Date.now() - t0 > 20000) throw new Error("Chrome did not start"); await new Promise(r => setTimeout(r, 50)); }
  const [port, wsPath] = fs.readFileSync(portFile, "utf8").trim().split("\n");
  const ws = new WebSocket(`ws://127.0.0.1:${port}${wsPath}`);
  await new Promise((res, rej) => { ws.once("open", res); ws.once("error", rej); });
  const cdp = new CDP(ws);
  try {
    const info = await cdp.send("SystemInfo.getInfo");
    const renderer = info.gpu.auxAttributes.glRenderer || "", fs_ = info.gpu.featureStatus;
    const accelerated = /NVIDIA/.test(renderer) && fs_.gpu_compositing === "enabled" && fs_["2d_canvas"] === "enabled";
    const { targetId } = await cdp.send("Target.createTarget", { url: "about:blank" });
    const { sessionId } = await cdp.send("Target.attachToTarget", { targetId, flatten: true });
    await cdp.send("Runtime.enable", {}, sessionId);
    await cdp.send("Page.enable", {}, sessionId);
    await cdp.send("Emulation.setFocusEmulationEnabled", { enabled: true }, sessionId);
    await cdp.send("Emulation.setDeviceMetricsOverride", { width: 1600, height: 1000, deviceScaleFactor: 1, mobile: false }, sessionId);
    await cdp.send("Page.navigate", { url: `file://${page}` }, sessionId);
    const start = Date.now();
    while (!(await cdp.evaluate("document.body && document.body.dataset.ready === 'true'", sessionId))) {
      if (Date.now() - start > 30000) throw new Error("page never became ready"); await new Promise(r => setTimeout(r, 100));
    }
    const boards = await cdp.evaluate("document.querySelectorAll('.card').length", sessionId);
    // every board on screen, so the latency covers all of them
    const full = await cdp.evaluate("document.documentElement.scrollHeight", sessionId);
    await cdp.send("Emulation.setDeviceMetricsOverride", { width: 1600, height: Math.min(full, 16000), deviceScaleFactor: 1, mobile: false }, sessionId);
    await cdp.evaluate("window.replay.seekAll(150); new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))", sessionId);
    const latency = await cdp.evaluate(`(async () => {
      const t = [];
      for (let i = 0; i < ${updates} + 10; i++) {
        const s = performance.now();
        window.replay.stepAll(i % 2 ? -1 : 1);
        await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
        if (i >= 10) t.push(performance.now() - s);
      }
      t.sort((a, b) => a - b);
      const q = f => Math.round(t[Math.min(t.length - 1, Math.floor(f * t.length))] * 10) / 10;
      return { updates: t.length, p50_ms: q(0.5), p95_ms: q(0.95), max_ms: q(1) };
    })()`, sessionId);
    const withEvidence = await cdp.evaluate("[...document.querySelectorAll('.ev-now')].filter(e => e.textContent && !e.textContent.includes('無證據')).length", sessionId);
    await cdp.send("Emulation.setDeviceMetricsOverride", { width: 1600, height: 1100, deviceScaleFactor: 1, mobile: false }, sessionId);
    await cdp.evaluate("window.scrollTo(0, 0); window.replay.seekAll(120); new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))", sessionId);
    const shot = await cdp.send("Page.captureScreenshot", { format: "png" }, sessionId);
    const stem = path.join(path.dirname(page), "verify_" + path.basename(page, ".html"));
    fs.writeFileSync(stem + ".png", Buffer.from(shot.data, "base64"));
    const report = { page: path.basename(page), renderer, accelerated, features: { gpu_compositing: fs_.gpu_compositing, "2d_canvas": fs_["2d_canvas"], rasterization: fs_.rasterization },
      boards, boards_with_evidence: withEvidence, latency_all_boards_on_screen: latency, console_errors: cdp.errors, screenshot: stem + ".png" };
    fs.writeFileSync(stem + ".json", JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report));
    if (!accelerated || cdp.errors.length || boards === 0) process.exitCode = 1;
  } finally {
    try { await cdp.send("Browser.close"); } catch { /* already gone */ }
    ws.close();
    if (chrome.exitCode === null) {
      await new Promise(res => { chrome.once("exit", res); setTimeout(() => { chrome.kill("SIGKILL"); res(); }, 5000); });
    }
    try { fs.rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 }); } catch { /* temp dir; ignore */ }
  }
})().catch(e => { console.error(e.stack || String(e)); process.exit(1); });
