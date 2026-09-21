// Service worker for the Slang Half-Life dashboard: makes it work offline.
//
// - The app shell (page, manifest, icons) is served from the cache first, so the
//   app opens instantly and offline. Each time it's served, a fresh copy is
//   fetched in the background, so a new version of the page shows up on the
//   next launch without waiting for a new service worker.
// - The data (latest.json) is fetched from the network first, so the weekly
//   update appears as soon as there's a connection. Offline, the last copy is
//   served with an X-Slang-Offline header so the page can say it's showing
//   older data.
// - Caches carry a version. Bump VERSION when the shell list changes; the
//   activate step deletes every cache from older versions.
//
// All paths are relative to this file, which lives in the site's folder
// (/slang-half-life/), so the worker's scope is exactly the app.

const VERSION = "v1";
const SHELL_CACHE = `slang-shell-${VERSION}`;
const DATA_CACHE = `slang-data-${VERSION}`;
const SHELL = [
  "./",
  "index.html",
  "manifest.webmanifest",
  "icons/favicon-32.png",
  "icons/icon-192.png",
  "icons/icon-512.png",
  "icons/icon-maskable-512.png",
  "icons/apple-touch-icon.png",
];
const DATA = "latest.json";

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE)
      .then((cache) => cache.addAll(SHELL))
      .then(() => fetch(DATA, { cache: "no-store" }))
      .then((res) => res.ok && caches.open(DATA_CACHE).then((c) => c.put(DATA, res)))
      .catch(() => {})  // the shell is what matters; data gets cached on first use
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  const keep = new Set([SHELL_CACHE, DATA_CACHE]);
  event.waitUntil(
    caches.keys()
      .then((names) => Promise.all(names.filter((n) => n.startsWith("slang-") && !keep.has(n)).map((n) => caches.delete(n))))
      .then(() => self.clients.claim())
  );
});

async function dataNetworkFirst(request) {
  const cache = await caches.open(DATA_CACHE);
  try {
    const res = await fetch(request, { cache: "no-store" });
    if (res.ok) await cache.put(DATA, res.clone());
    return res;
  } catch (err) {
    const cached = await cache.match(DATA);
    if (!cached) throw err;
    const headers = new Headers(cached.headers);
    headers.set("X-Slang-Offline", "1");
    return new Response(await cached.blob(), { status: 200, headers });
  }
}

async function shellCacheFirst(event) {
  const request = event.request;
  const cache = await caches.open(SHELL_CACHE);
  const key = request.mode === "navigate" ? "./" : request;
  const cached = await cache.match(key, { ignoreSearch: request.mode === "navigate" });
  const refresh = fetch(request)
    .then((res) => { if (res.ok) cache.put(key, res.clone()); return res; })
    .catch(() => undefined);
  if (cached) {
    event.waitUntil(refresh);
    return cached;
  }
  const res = await refresh;
  return res || Response.error();
}

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin || !url.pathname.startsWith(new URL("./", self.location).pathname)) return;
  if (url.pathname.endsWith("/" + DATA)) {
    event.respondWith(dataNetworkFirst(request));
  } else {
    event.respondWith(shellCacheFirst(event));
  }
});
