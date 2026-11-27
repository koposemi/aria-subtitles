// ==UserScript==
// @name         ARIA 영상 자막 (PC 서버 연동)
// @namespace    aria-subtitles
// @version      3.3.0
// @description  재생 중인 영상의 음성을 PC 자막 서버가 일본어 인식·한국어 번역하고, 재생 위치에 맞춰 영상 위에 표시합니다. PC·모바일 공용.
// @match        https://missav123.com/*
// @match        https://m.youtube.com/*
// @match        https://www.youtube.com/*
// @match        https://www.njav.com/*
// @match        https://commons.wikimedia.org/*
// @match        https://commons.m.wikimedia.org/*
// @run-at       document-start
// @grant        GM_xmlhttpRequest
// @grant        GM.xmlHttpRequest
// @grant        GM_getValue
// @grant        GM_setValue
// @grant        GM_registerMenuCommand
// @connect      127.0.0.1
// @connect      localhost
// @connect      __ARIA_CONNECT_HOST__
// ==/UserScript==
//
// 다른 영상 사이트에서도 쓰려면 위에 "// @match https://사이트주소/*" 줄을 추가하세요.
// 모든 사이트에서 켜려면 "// @match *://*/*" 한 줄이면 됩니다.
// 스크립트가 플레이어가 실제로 재생 중인 스트림 주소(m3u8/mp4)를 찾아 PC 서버에 보내므로
// yt-dlp가 모르는 사이트도 동작합니다. DRM 보호 영상(넷플릭스 등)은 불가능합니다.
// 플레이어가 다른 주소의 iframe 안에 있는 사이트는 그 iframe 주소도 @match에 추가하세요.

(function () {
  "use strict";

  // ------------------------------------------------------------------ settings
  // The server fills these in when the script is installed from /aria.user.js.
  const INSTALLED_SERVER = "__ARIA_SERVER__";
  const INSTALLED_TOKEN = "__ARIA_TOKEN__";

  const store = {
    get(k, d) { try { return typeof GM_getValue === "function" ? GM_getValue(k, d) : (JSON.parse(localStorage.getItem("aria_" + k)) ?? d); } catch (e) { return d; } },
    set(k, v) { try { if (typeof GM_setValue === "function") GM_setValue(k, v); else localStorage.setItem("aria_" + k, JSON.stringify(v)); } catch (e) { /* ignore */ } },
  };
  const placeholder = (v) => !v || v.startsWith("__ARIA_");
  let SERVER = store.get("server", "") || (placeholder(INSTALLED_SERVER) ? "http://127.0.0.1:8765" : INSTALLED_SERVER);
  let TOKEN = store.get("token", "") || (placeholder(INSTALLED_TOKEN) ? "" : INSTALLED_TOKEN);
  // Display mode: 0 = 한국어, 1 = 일본어+한국어. On/off is the round button on the video.
  let mode = store.get("mode", 0);
  let enabled = store.get("enabled", true);
  if (mode === 2) { mode = 0; enabled = false; store.set("mode", 0); store.set("enabled", false); }
  let scale = store.get("scale", 1.0);

  if (typeof GM_registerMenuCommand === "function") {
    GM_registerMenuCommand("자막 서버 주소 설정", () => {
      const v = prompt("PC 자막 서버 주소 (예: http://192.168.0.10:8765)", SERVER);
      if (v) { SERVER = v.trim().replace(/\/+$/, ""); store.set("server", SERVER); }
      const t = prompt("접속 토큰 (USB/PC면 비워두세요)", TOKEN);
      if (t !== null) { TOKEN = t.trim(); store.set("token", TOKEN); }
    });
    GM_registerMenuCommand("자막 크게", () => { scale = Math.min(2.5, scale + 0.15); store.set("scale", scale); });
    GM_registerMenuCommand("자막 작게", () => { scale = Math.max(0.5, scale - 0.15); store.set("scale", scale); });
  }

  // ------------------------------------------------------------------ HTTP
  const gmRequest = (typeof GM !== "undefined" && GM && GM.xmlHttpRequest) ? GM.xmlHttpRequest.bind(GM)
    : (typeof GM_xmlhttpRequest === "function" ? GM_xmlhttpRequest : null);

  // Servers to try: the one the script was installed from (Wi-Fi address when installed over
  // Wi-Fi) and this device's 127.0.0.1 (the PC over a USB cable). The one that answers is used.
  const USB_SERVER = "http://127.0.0.1:8765";
  function serverList() { return [...new Set([SERVER, USB_SERVER])]; }
  let serverIdx = 0;

  async function api(query) {
    const list = serverList();
    let lastError;
    for (let i = 0; i < list.length; i++) {
      const idx = (serverIdx + i) % list.length;
      try {
        const d = await getJSON(`${list[idx]}/subs?${query}`);
        if (idx !== serverIdx) serverIdx = idx;
        return d;
      } catch (e) {
        lastError = e;
      }
    }
    throw lastError || new Error("offline");
  }

  function getJSON(url) {
    return new Promise((resolve, reject) => {
      if (!gmRequest) {
        fetch(url, { cache: "no-store" }).then((r) => r.json()).then(resolve, reject);
        return;
      }
      gmRequest({
        method: "GET", url, timeout: 4000,   // short: an unreachable Wi-Fi server falls back to USB quickly
        onload: (r) => { try { resolve(JSON.parse(r.responseText)); } catch (e) { reject(e); } },
        onerror: () => reject(new Error("network")),
        ontimeout: () => reject(new Error("timeout")),
      });
    });
  }

  // ------------------------------------------------------------------ page helpers
  const isYouTube = /(^|\.)youtube\.com$/.test(location.hostname);

  function findVideo() {
    if (isYouTube) {
      const v = document.querySelector("#movie_player video, video.html5-main-video, .html5-video-player video");
      if (v) return v;
    }
    let best = null, area = 0;
    for (const v of document.querySelectorAll("video")) {
      const r = v.getBoundingClientRect();
      if (r.width * r.height > area && (v.duration || 0) > 20) { best = v; area = r.width * r.height; }
    }
    return best;
  }

  function pageKeyUrl() {
    if (isYouTube) {
      const u = new URL(location.href);
      const id = u.searchParams.get("v") || (u.pathname.match(/^\/(?:shorts|live|embed)\/([A-Za-z0-9_-]{11})/) || [])[1];
      return id ? `https://www.youtube.com/watch?v=${id}` : null;
    }
    return location.href.split("#")[0];
  }

  function adPlaying() {
    return !!document.querySelector(".ad-showing, .ad-interrupting");
  }

  // ------------------------------------------------------------------ stream detection
  // Sites yt-dlp doesn't know still work if the PC gets the stream the player is playing.
  // The script starts at document-start so it sees the player's first requests.
  const HLS_RE = /\.m3u8(?:[?#]|$)/i;
  const FILE_RE = /\.(?:mp4|m4v|webm|mov|mkv|m4a|mp3|aac|ogg|opus)(?:[?#]|$)/i;
  const seen = [];   // {url, at, hls}
  function noteMedia(url) {
    if (isYouTube || typeof url !== "string" || !/^https?:\/\//i.test(url)) return;
    const hls = HLS_RE.test(url);
    if (!hls && !FILE_RE.test(url)) return;
    if (seen.some((m) => m.url === url)) return;
    seen.push({ url, at: performance.now(), hls });
    if (seen.length > 200) seen.shift();
  }
  if (!isYouTube) {
    try { performance.setResourceTimingBufferSize(5000); } catch (e) { /* ignore */ }
    try {
      new PerformanceObserver((list) => list.getEntries().forEach((e) => noteMedia(e.name)))
        .observe({ type: "resource", buffered: true });
    } catch (e) { /* older browser: scanned in mediaUrl() */ }
  }
  let mediaSince = 0;   // only streams loaded for the current page/video

  function mediaUrl(video) {
    if (isYouTube || !video) return "";
    const src = video.currentSrc || video.src || "";
    if (/^https?:\/\//i.test(src)) return src;   // plain file, or native HLS
    try { performance.getEntriesByType("resource").forEach((e) => noteMedia(e.name)); } catch (e) { /* ignore */ }
    const fresh = seen.filter((m) => m.at >= mediaSince);
    // A player loads its master playlist first, then variant/segment playlists.
    const hls = fresh.filter((m) => m.hls);
    const named = hls.find((m) => /master|playlist|index|manifest/i.test(m.url));
    if (named) return named.url;
    if (hls.length) return hls[0].url;
    const files = fresh.filter((m) => !m.hls);
    return files.length ? files[files.length - 1].url : "";
  }

  // ------------------------------------------------------------------ overlay
  const root = document.createElement("div");
  root.id = "aria-subs-root";
  root.style.cssText = "position:fixed;left:0;top:0;width:0;height:0;pointer-events:none;z-index:2147483647;";
  const line = document.createElement("div");
  line.style.cssText = [
    // width:max-content: an absolutely placed box at left:50% would otherwise shrink to
    // the right half of the video and wrap every few words.
    "position:absolute;left:50%;transform:translateX(-50%);width:max-content;max-width:92%;text-align:center;",
    "color:#fff;font-weight:600;line-height:1.35;white-space:pre-wrap;word-break:keep-all;",
    "text-shadow:0 0 3px #000,0 0 3px #000,1px 1px 2px #000;",
    "background:rgba(0,0,0,.45);border-radius:6px;padding:.15em .5em;display:none;",
    "font-family:'Noto Sans KR','Apple SD Gothic Neo','Malgun Gothic',sans-serif;",
  ].join("");
  const ja = document.createElement("div");
  ja.style.cssText = "font-size:.72em;opacity:.85;font-weight:500;";
  const ko = document.createElement("div");
  line.append(ja, ko);
  const badge = document.createElement("div");
  badge.style.cssText = [
    "position:absolute;left:8px;top:8px;pointer-events:auto;cursor:pointer;user-select:none;",
    "font:600 12px/1.3 sans-serif;color:#fff;background:rgba(0,0,0,.6);border-radius:12px;",
    "padding:4px 9px;max-width:80%;-webkit-tap-highlight-color:transparent;",
  ].join("");
  // ON/OFF button: right edge of the video, away from the player's own controls.
  const toggle = document.createElement("div");
  toggle.style.cssText = [
    "position:absolute;right:6px;top:50%;transform:translateY(-50%);width:46px;height:46px;",
    "border-radius:50%;display:flex;align-items:center;justify-content:center;text-align:center;",
    "font:700 11px/1.15 sans-serif;color:#fff;pointer-events:auto;cursor:pointer;user-select:none;",
    "-webkit-tap-highlight-color:transparent;opacity:.6;border:2px solid rgba(255,255,255,.7);",
  ].join("");
  function paintToggle() {
    toggle.textContent = "";
    toggle.append("자막", document.createElement("br"), enabled ? "ON" : "OFF");
    toggle.style.background = enabled ? "rgba(30,140,60,.75)" : "rgba(60,60,60,.75)";
  }
  paintToggle();
  root.append(line, badge, toggle);

  let status = { status: "idle", message: "" };
  let badgeHideAt = 0;

  badge.addEventListener("click", (ev) => {
    ev.stopPropagation();
    ev.preventDefault();
    if (status.status === "error") { retryNext = true; badgeText("다시 시도 중…"); return; }
    mode = (mode + 1) % 2;
    store.set("mode", mode);
    badgeText(["자막: 한국어", "자막: 일본어+한국어"][mode], 2500);
  }, true);

  // Keep taps on the button from reaching the player (which would pause/seek).
  for (const type of ["pointerdown", "pointerup", "touchstart", "touchend", "mousedown", "mouseup", "dblclick"]) {
    toggle.addEventListener(type, (ev) => { ev.stopPropagation(); }, true);
  }
  toggle.addEventListener("click", (ev) => {
    ev.stopPropagation();
    ev.preventDefault();
    enabled = !enabled;
    store.set("enabled", enabled);
    paintToggle();
    if (enabled) {
      lastPoll = 0;   // fetch subtitles right away
      badgeText("자막 켬", 1500);
    } else {
      stopServerWork();
      badgeText("자막 끔 (서버 작업 중지)", 1500);
    }
  }, true);

  // Tell the server to stop working on this video now (it would otherwise go on
  // for up to a minute after the last request).
  function stopServerWork() {
    const key = pageKeyUrl();
    if (!key) return;
    const q = new URLSearchParams({ url: key, off: "1" });
    if (TOKEN) q.set("k", TOKEN);
    if (navigator.maxTouchPoints > 0 || "ontouchstart" in window) q.set("touch", "1");
    api(q.toString()).catch(() => {});
  }

  function badgeText(text, holdMs) {
    badge.textContent = "ARIA · " + text;
    badge.style.display = "block";
    badgeHideAt = holdMs ? performance.now() + holdMs : 0;
  }

  function mountPoint(video) {
    const fs = document.fullscreenElement || document.webkitFullscreenElement;
    if (fs && fs !== video && fs.contains(video)) return fs;
    return document.documentElement;
  }

  // ------------------------------------------------------------------ cues
  let cues = [];          // sorted by s
  let cueWindow = null;   // [lo, hi] the server sent
  let currentKey = null;
  let retryNext = false;

  function cueAt(t) {
    let lo = 0, hi = cues.length - 1, found = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (cues[mid].s <= t) { found = mid; lo = mid + 1; } else hi = mid - 1;
    }
    if (found < 0) return null;
    const c = cues[found];
    return t < c.e + 0.35 ? c : null;   // hold briefly so short gaps don't flicker
  }

  function render() {
    const video = findVideo();
    // "disabled": the server does not serve this browser (e.g. PC): show nothing at all.
    if (!video || status.status === "disabled") { root.style.display = "none"; return; }
    if (!pageKeyUrl()) { root.style.display = "none"; return; }   // not a video page
    const parent = mountPoint(video);
    if (root.parentNode !== parent) parent.appendChild(root);
    const r = video.getBoundingClientRect();
    if (r.width < 50 || r.height < 30) { root.style.display = "none"; return; }
    root.style.display = "block";
    root.style.left = r.left + "px";
    root.style.top = r.top + "px";
    root.style.width = r.width + "px";
    root.style.height = r.height + "px";

    // The picture inside the element (players often letterbox it, e.g. a tall portrait
    // player with black bars): size and place the text by the picture, not the element.
    let ph = r.height, below = 0;
    if (video.videoWidth && video.videoHeight) {
      const s = Math.min(r.width / video.videoWidth, r.height / video.videoHeight);
      ph = video.videoHeight * s;
      below = (r.height - ph) / 2;   // black bar under the picture
    }
    const size = Math.max(15, Math.min(40, ph * 0.065)) * scale;
    line.style.fontSize = size + "px";
    if (below > size * 3.2) {
      // A tall bar under the picture: put the text there, clear of burned-in captions.
      line.style.bottom = Math.round(below - size * 2.9) + "px";
    } else {
      line.style.bottom = Math.round(below + ph * (isYouTube && !document.fullscreenElement ? 0.12 : 0.08)) + "px";
    }

    // Turned off (or nothing fetched yet): only the ON/OFF button stays on screen.
    toggle.style.display = adPlaying() ? "none" : "flex";
    if (!enabled || !currentKey) {
      line.style.display = "none";
      if (!(badgeHideAt && performance.now() < badgeHideAt)) badge.style.display = "none";
      return;
    }

    const c = adPlaying() ? null : cueAt(video.currentTime);
    if (c && (c.ko || c.ja)) {
      ko.textContent = c.ko || c.ja;
      ja.textContent = mode === 1 && c.ko ? c.ja : "";
      ja.style.display = ja.textContent ? "block" : "none";
      line.style.display = "block";
    } else {
      line.style.display = "none";
    }

    // Status badge: shown while subtitles are not ready at the current position.
    const now = performance.now();
    if (badgeHideAt && now < badgeHideAt) return;
    const t = video.currentTime;
    const ready = (status.covered || []).some(([a, b]) => a <= t + 0.5 && t < b);
    if (status.status === "error") {
      badgeText("⚠ " + (status.message || "오류") + " (탭하여 재시도)");
    } else if (status.status === "offline") {
      badgeText("PC 자막 서버에 연결할 수 없음");
    } else if (!ready && status.status !== "done") {
      const p = status.progress || {};
      let msg = status.message || "자막 준비 중…";
      if (p.working && p.audio_received != null) msg += ` (${Math.round(p.audio_received)}s)`;
      badgeText(msg);
    } else {
      badge.style.display = "none";
    }
  }

  // ------------------------------------------------------------------ polling
  let polling = false;
  let lastPoll = 0;

  let videoSeenAt = 0;

  async function poll() {
    const video = findVideo();
    const key = pageKeyUrl();
    if (!video || !key || adPlaying()) return;
    if (key !== currentKey) {
      const firstPage = currentKey === null;
      currentKey = key; cues = []; cueWindow = null; status = { status: "idle" };
      videoSeenAt = performance.now();
      // A new video in a single-page site: forget the previous video's streams. The
      // first page keeps everything loaded since the page opened.
      if (!firstPage) mediaSince = Math.max(0, videoSeenAt - 2000);
    }
    const media = mediaUrl(video);
    // Off YouTube, give the player a few seconds to request its stream, so the server
    // doesn't first try (and possibly fail) to download the page with yt-dlp.
    if (!isYouTube && !media && performance.now() - videoSeenAt < 5000) return;
    const playing = !video.paused && document.visibilityState === "visible";
    const t = video.currentTime || 0;
    const q = new URLSearchParams({ url: key, t: t.toFixed(2), playing: playing ? "1" : "0", win: "1" });
    if (media) q.set("media", media);
    // Phones/tablets are served even in "desktop site" mode (desktop User-Agent).
    if (navigator.maxTouchPoints > 0 || "ontouchstart" in window) q.set("touch", "1");
    if (TOKEN) q.set("k", TOKEN);
    if (retryNext) { q.set("retry", "1"); retryNext = false; }
    try {
      const d = await api(q.toString());
      if (key !== currentKey) return;
      if (d.error === "forbidden") { status = { status: "error", message: d.message }; return; }
      if (d.error) { status = { status: "error", message: d.error }; return; }
      status = d;
      cues = (d.cues || []).slice().sort((a, b) => a.s - b.s);
      cueWindow = d.window || null;
    } catch (e) {
      status = { status: "offline" };
    }
  }

  async function tick() {
    if (polling || !enabled) return;   // off: no requests, so no quota is used
    const video = findVideo();
    const now = performance.now();
    const active = video && !video.paused && document.visibilityState === "visible";
    // 1 s while playing (the server follows the viewer's position), 5 s otherwise.
    const interval = status.status === "disabled" ? 60000
      : status.status === "offline" ? 4000 : (active ? 1000 : 5000);
    const done = status.status === "done" && cueWindow && video &&
      video.currentTime > cueWindow[0] + 5 && video.currentTime < cueWindow[1] - 60;
    if (now - lastPoll < (done ? 10000 : interval)) return;
    lastPoll = now;
    polling = true;
    try { await poll(); } finally { polling = false; }
  }

  // Seeking should be answered immediately, not on the next interval.
  document.addEventListener("seeked", () => { lastPoll = 0; }, true);
  document.addEventListener("play", () => { lastPoll = 0; }, true);

  setInterval(tick, 250);
  (function loop() { try { render(); } catch (e) { /* keep running */ } requestAnimationFrame(loop); })();
})();
