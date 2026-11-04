const $ = (id) => document.getElementById(id);

function bytes(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = Number(value);
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  const digits = index === 0 ? 0 : 1;
  return `${size.toFixed(digits)} ${units[index]}`;
}

async function getJson(path) {
  const response = await fetch(path, { credentials: "same-origin" });
  if (response.status === 401 || response.status === 503) {
    window.location.assign("/auth/login");
    throw new Error("Sign-in required");
  }
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return response.json();
}

function showError(target, error) {
  target.textContent = error.message || String(error);
  target.classList.add("error");
}

async function loadOverview() {
  const status = await getJson("/api/admin/status");
  const stats = [
    ["Documents", status.documents.count],
    ["Chunks", status.documents.chunk_count_total],
    ["Vectors", status.vector_store.points_count ?? "—"],
    ["Volume", bytes(status.volume.size_bytes)],
    ["MCP tools", status.mcp.tool_count],
  ];
  $("overview-stats").innerHTML = stats
    .map(
      ([label, value]) =>
        `<div class="stat"><span class="muted">${label}</span><strong>${value}</strong></div>`,
    )
    .join("");
  $("overview-serve").textContent = JSON.stringify(
    {
      rag_data_root: status.rag_data_root,
      embeddings: status.embeddings,
      vector_store: status.vector_store,
      mcp: status.mcp,
    },
    null,
    2,
  );
  $("footer-note").textContent = `${status.service} · ${status.rag_data_root}`;
}

async function loadDocuments() {
  const catalog = await getJson("/api/admin/documents");
  const list = $("doc-list");
  const detail = $("doc-detail");
  if (!catalog.documents.length) {
    list.innerHTML = `<div class="item muted">No indexed documents.</div>`;
    detail.textContent = "Select a document.";
    detail.className = "muted";
    return;
  }
  list.innerHTML = catalog.documents
    .map(
      (doc) =>
        `<button type="button" data-doc="${doc.doc_id}">${doc.filename || doc.doc_id}<br><span class="muted">${doc.doc_id}</span></button>`,
    )
    .join("");
  list.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", async () => {
      list.querySelectorAll("button").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      detail.className = "";
      detail.textContent = "Loading…";
      try {
        const body = await getJson(`/api/admin/documents/${button.dataset.doc}`);
        detail.innerHTML = `<pre class="code">${escapeHtml(JSON.stringify(body, null, 2))}</pre>`;
      } catch (error) {
        showError(detail, error);
      }
    });
  });
}

function treeSection(title, entries) {
  const items = (entries || [])
    .map((entry) => `<li>${entry.is_dir ? "dir" : "file"} ${entry.name} · ${bytes(entry.size_bytes)}</li>`)
    .join("");
  return `<details open><summary>${title}</summary><ul>${items || "<li>empty</li>"}</ul></details>`;
}

async function loadVolume() {
  const volume = await getJson("/api/admin/volume");
  $("volume-root").textContent = `${volume.root}\nexists=${volume.exists} size=${bytes(volume.size_bytes)}`;
  const entries = volume.entries || {};
  $("volume-tree").innerHTML = [
    treeSection("originals", entries.originals),
    treeSection("processed", entries.processed),
    treeSection("qdrant", entries.qdrant),
    `<details open><summary>index</summary><ul><li>${entries.index?.path || "index.json"} · ${entries.index?.exists ? bytes(entries.index.size_bytes) : "missing"}</li></ul></details>`,
  ].join("");
}

async function loadMcp() {
  const mcp = await getJson("/api/admin/mcp");
  $("mcp-status").textContent = JSON.stringify(
    {
      mounted: mcp.mounted,
      mount_path: mcp.mount_path,
      server_name: mcp.server_name,
      transport: mcp.transport,
    },
    null,
    2,
  );
  $("mcp-tools").innerHTML = (mcp.tools || [])
    .map(
      (tool) =>
        `<div class="item"><strong>${escapeHtml(tool.name)}</strong><br><span class="muted">${escapeHtml(tool.description || "")}</span></div>`,
    )
    .join("") || `<div class="item muted">No tools registered.</div>`;
}

async function loadLogs() {
  const level = $("log-level").value;
  const query = level ? `?level=${encodeURIComponent(level)}&limit=200` : "?limit=200";
  const logs = await getJson(`/api/admin/logs${query}`);
  $("log-view").textContent = (logs.entries || [])
    .map((entry) => {
      const time = entry.ts || entry.timestamp || "";
      const levelName = entry.level || "";
      const message = entry.message || entry.msg || JSON.stringify(entry);
      return `${time} ${levelName} ${message}`.trim();
    })
    .join("\n");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

async function refresh() {
  $("footer-note").textContent = "Loading…";
  try {
    await Promise.all([loadOverview(), loadDocuments(), loadVolume(), loadMcp(), loadLogs()]);
  } catch (error) {
    if (error.message !== "Sign-in required") {
      $("footer-note").textContent = error.message;
    }
  }
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((panel) => panel.classList.remove("active"));
    tab.classList.add("active");
    $(`panel-${tab.dataset.tab}`).classList.add("active");
  });
});

$("refresh").addEventListener("click", refresh);
$("log-level").addEventListener("change", () => {
  loadLogs().catch((error) => showError($("log-view"), error));
});

refresh();
