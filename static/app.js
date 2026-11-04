const $ = (id) => document.getElementById(id)

let selectedAccountDate = null

function bytes (value) {
  if (value == null || Number.isNaN(Number(value))) return '—'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let size = Number(value)
  let index = 0
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024
    index += 1
  }
  const digits = index === 0 ? 0 : 1
  return `${size.toFixed(digits)} ${units[index]}`
}

function number (value) {
  if (value == null || Number.isNaN(Number(value))) return '0'
  return Number(value).toLocaleString()
}

async function getJson (path) {
  const response = await fetch(path, { credentials: 'same-origin' })
  if (response.status === 401 || response.status === 503) {
    window.location.assign('/auth/login')
    throw new Error('Sign-in required')
  }
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `Request failed (${response.status})`)
  }
  return response.json()
}

function showError (target, error) {
  target.textContent = error.message || String(error)
  target.classList.add('error')
}

async function loadOverview () {
  const status = await getJson('/api/admin/status')
  const stats = [
    ['Documents', status.documents.count],
    ['Chunks', status.documents.chunk_count_total],
    ['Vectors', status.vector_store.points_count ?? '—'],
    ['Volume', bytes(status.volume.size_bytes)],
    ['MCP tools', status.mcp.tool_count]
  ]
  $('overview-stats').innerHTML = stats
    .map(
      ([label, value]) =>
        `<div class="stat"><span class="muted">${label}</span><strong>${value}</strong></div>`
    )
    .join('')
  $('overview-serve').textContent = JSON.stringify(
    {
      rag_data_root: status.rag_data_root,
      embeddings: status.embeddings,
      vector_store: status.vector_store,
      mcp: status.mcp,
      accnt_root: status.volume?.accnt_root
    },
    null,
    2
  )
  $('footer-note').textContent = `${status.service} · ${status.rag_data_root}`
}

async function loadDocuments () {
  const catalog = await getJson('/api/admin/documents')
  const list = $('doc-list')
  const detail = $('doc-detail')
  if (!catalog.documents.length) {
    list.innerHTML = '<div class="item muted">No indexed documents.</div>'
    detail.textContent = 'Select a document.'
    detail.className = 'muted'
    return
  }
  list.innerHTML = catalog.documents
    .map(
      (doc) =>
        `<button type="button" data-doc="${doc.doc_id}">${doc.filename || doc.doc_id}<br><span class="muted">${doc.doc_id}</span></button>`
    )
    .join('')
  list.querySelectorAll('button').forEach((button) => {
    button.addEventListener('click', async () => {
      list.querySelectorAll('button').forEach((item) => item.classList.remove('active'))
      button.classList.add('active')
      detail.className = ''
      detail.textContent = 'Loading…'
      try {
        const body = await getJson(`/api/admin/documents/${button.dataset.doc}`)
        detail.innerHTML = `<pre class="code">${escapeHtml(JSON.stringify(body, null, 2))}</pre>`
      } catch (error) {
        showError(detail, error)
      }
    })
  })
}

function treeSection (title, entries) {
  const items = (entries || [])
    .map((entry) => `<li>${entry.is_dir ? 'dir' : 'file'} ${entry.name} · ${bytes(entry.size_bytes)}</li>`)
    .join('')
  return `<details open><summary>${title}</summary><ul>${items || '<li>empty</li>'}</ul></details>`
}

async function loadVolume () {
  const volume = await getJson('/api/admin/volume')
  $('volume-root').textContent = `${volume.root}\nexists=${volume.exists} size=${bytes(volume.size_bytes)}`
  const entries = volume.entries || {}
  $('volume-tree').innerHTML = [
    treeSection('originals', entries.originals),
    treeSection('processed', entries.processed),
    treeSection('qdrant', entries.qdrant),
    `<details open><summary>index</summary><ul><li>${entries.index?.path || 'index.json'} · ${entries.index?.exists ? bytes(entries.index.size_bytes) : 'missing'}</li></ul></details>`
  ].join('')
}

async function loadMcp () {
  const mcp = await getJson('/api/admin/mcp')
  $('mcp-status').textContent = JSON.stringify(
    {
      mounted: mcp.mounted,
      mount_path: mcp.mount_path,
      server_name: mcp.server_name,
      transport: mcp.transport
    },
    null,
    2
  )
  $('mcp-tools').innerHTML = (mcp.tools || [])
    .map(
      (tool) =>
        `<div class="item"><strong>${escapeHtml(tool.name)}</strong><br><span class="muted">${escapeHtml(tool.description || '')}</span></div>`
    )
    .join('') || '<div class="item muted">No tools registered.</div>'
}

function avgLatencyMs (outboundEmbed) {
  const calls = Number(outboundEmbed?.calls || 0)
  const sum = Number(outboundEmbed?.latency_ms_sum || 0)
  if (!calls) return null
  return Math.round(sum / calls)
}

function renderEmbeddingSourceTable (bySource) {
  const sources = Object.keys(bySource || {})
  if (!sources.length) {
    return '<p class="muted">No embedding activity for this day.</p>'
  }
  const rows = sources
    .map((source) => {
      const row = bySource[source] || {}
      return `<tr>
        <td>${escapeHtml(source)}</td>
        <td class="num">${number(row.calls)}</td>
        <td class="num">${number(row.errors)}</td>
        <td class="num">${number(row.prompt_tokens)}</td>
        <td class="num">${number(row.total_tokens)}</td>
        <td class="num">${number(row.input_chars)}</td>
        <td class="num">${number(row.tokens_from_api)}</td>
        <td class="num">${number(row.tokens_estimated)}</td>
      </tr>`
    })
    .join('')
  return `<table class="accounts-table">
    <thead><tr>
      <th>Source</th><th>Calls</th><th>Errors</th><th>Prompt tok</th><th>Total tok</th>
      <th>Chars</th><th>API usage</th><th>No usage</th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table>`
}

function renderEmbeddingCallsTable (events, meta = {}) {
  const embeddings = (events || []).filter((event) => event.kind === 'embedding')
  if (!embeddings.length) {
    return '<p class="muted">No embedding API calls recorded for this day yet.</p>'
  }
  const rows = embeddings
    .map((event) => {
      const ok = event.ok !== false
      const fromApi = Boolean(event.tokens_from_api)
      const prompt = number(event.prompt_tokens)
      const total = number(event.total_tokens)
      const tokenClass = fromApi ? 'token-yes' : ''
      const usageLabel = fromApi ? 'API' : (ok ? 'missing' : '—')
      const error = event.error ? escapeHtml(String(event.error)) : ''
      return `<tr class="${ok ? '' : 'fail'}">
        <td>${escapeHtml(event.ts || '')}</td>
        <td>${escapeHtml(event.source || 'other')}</td>
        <td>${ok ? 'ok' : 'error'}</td>
        <td class="num ${tokenClass}">${prompt}</td>
        <td class="num ${tokenClass}">${total}</td>
        <td>${usageLabel}</td>
        <td class="num">${number(event.chars)}</td>
        <td class="num">${number(event.latency_ms)} ms</td>
        <td>${escapeHtml(event.model || '')}</td>
        <td class="clip" title="${error}">${error}</td>
      </tr>`
    })
    .join('')
  const shown = embeddings.length
  const limit = meta.event_limit
  const note = limit && shown >= limit
    ? `<p class="muted">Showing newest ${number(shown)} embedding calls (limit ${number(limit)}).</p>`
    : `<p class="muted">${number(shown)} embedding call${shown === 1 ? '' : 's'}.</p>`
  return `${note}<table class="accounts-table">
    <thead><tr>
      <th>Time (UTC)</th><th>Source</th><th>Status</th>
      <th>Prompt tokens</th><th>Total tokens</th><th>Usage</th>
      <th>Chars</th><th>Latency</th><th>Model</th><th>Error</th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table>`
}

function renderAccountDay (day, meta = {}) {
  const emb = day.embeddings || {}
  const mcp = day.mcp || {}
  const outboundEmbed = day.outbound?.novita_embeddings || {}
  const latency = avgLatencyMs(outboundEmbed)
  const stats = [
    ['Embed calls', number(emb.calls)],
    ['Prompt tokens', number(emb.prompt_tokens)],
    ['Total tokens', number(emb.total_tokens)],
    ['API usage calls', number(emb.tokens_from_api)],
    ['No usage calls', number(emb.tokens_estimated)],
    ['Embed errors', number(emb.errors)],
    ['Input chars', number(emb.input_chars)],
    ['Avg latency', latency == null ? '—' : `${number(latency)} ms`],
    ['MCP calls', number(mcp.calls)],
    ['MCP errors', number(mcp.errors)]
  ]
  $('accounts-stats').innerHTML = stats
    .map(
      ([label, value]) =>
        `<div class="stat"><span class="muted">${label}</span><strong>${value}</strong></div>`
    )
    .join('')
  $('accounts-embeddings').className = 'table-wrap'
  $('accounts-embeddings').innerHTML = renderEmbeddingSourceTable(emb.by_source || {})
  $('accounts-embedding-calls').className = 'table-wrap'
  $('accounts-embedding-calls').innerHTML = renderEmbeddingCallsTable(meta.events || [], meta)
  $('accounts-mcp').textContent = JSON.stringify(mcp.by_tool || mcp, null, 2)
  $('accounts-comms').textContent = JSON.stringify(
    {
      http: day.http || {},
      outbound: day.outbound || {}
    },
    null,
    2
  )
  const detail = {
    date: day.date || meta.date,
    updated_at: day.updated_at,
    exists: meta.exists,
    embeddings: {
      calls: emb.calls,
      errors: emb.errors,
      prompt_tokens: emb.prompt_tokens,
      total_tokens: emb.total_tokens,
      input_chars: emb.input_chars,
      tokens_from_api: emb.tokens_from_api,
      tokens_estimated: emb.tokens_estimated,
      by_source: emb.by_source || {}
    },
    embedding_events_shown: (meta.events || []).filter((event) => event.kind === 'embedding').length,
    mcp: {
      calls: mcp.calls,
      errors: mcp.errors,
      request_chars: mcp.request_chars,
      response_chars: mcp.response_chars
    },
    outbound_novita: outboundEmbed
  }
  $('accounts-detail').className = ''
  $('accounts-detail').innerHTML = `<pre class="code">${escapeHtml(JSON.stringify(detail, null, 2))}</pre>`
}

async function selectAccountDay (date) {
  selectedAccountDate = date
  const detail = $('accounts-detail')
  detail.className = 'muted'
  detail.textContent = 'Loading…'
  $('accounts-embedding-calls').className = 'table-wrap muted'
  $('accounts-embedding-calls').textContent = 'Loading…'
  const params = new URLSearchParams({
    include_events: '1',
    event_kind: 'embedding',
    event_limit: '500'
  })
  const body = await getJson(
    `/api/admin/accounts/${encodeURIComponent(date)}?${params.toString()}`
  )
  renderAccountDay(body.day, {
    date: body.date,
    exists: body.exists,
    events: body.events || [],
    event_limit: body.event_limit,
    event_kind: body.event_kind
  })
  document.querySelectorAll('#accounts-days button').forEach((button) => {
    button.classList.toggle('active', button.dataset.date === date)
  })
}

async function loadAccounts () {
  const summary = await getJson('/api/admin/accounts?days=30')
  const totals = summary.totals || {}
  const days = summary.days || []
  const today = summary.today
  const list = $('accounts-days')

  if (!days.length) {
    list.innerHTML = '<div class="item muted">No accounting days yet. Ingest or MCP calls will create UTC day files under /data/accnt.</div>'
    renderAccountDay(
      {
        date: today,
        embeddings: totals.embeddings || { calls: 0, by_source: {} },
        mcp: totals.mcp || { calls: 0, by_tool: {} },
        http: totals.http || {},
        outbound: totals.outbound || {}
      },
      { date: today, exists: false, events: [] }
    )
    return
  }

  list.innerHTML = days
    .map((day) => {
      const embCalls = day.embeddings?.calls ?? 0
      const promptTokens = day.embeddings?.prompt_tokens ?? 0
      const mcpCalls = day.mcp?.calls ?? 0
      const label = day.date === today ? `${day.date} (today)` : day.date
      return `<button type="button" data-date="${day.date}">${label}<br><span class="muted">embed ${number(embCalls)} · ${number(promptTokens)} tok · mcp ${number(mcpCalls)}</span></button>`
    })
    .join('')

  list.querySelectorAll('button').forEach((button) => {
    button.addEventListener('click', () => {
      selectAccountDay(button.dataset.date).catch((error) =>
        showError($('accounts-detail'), error)
      )
    })
  })

  const initial =
    selectedAccountDate && days.some((day) => day.date === selectedAccountDate)
      ? selectedAccountDate
      : days[0].date
  await selectAccountDay(initial)
}

async function loadLogs () {
  const level = $('log-level').value
  const query = level ? `?level=${encodeURIComponent(level)}&limit=200` : '?limit=200'
  const logs = await getJson(`/api/admin/logs${query}`)
  $('log-view').textContent = (logs.entries || [])
    .map((entry) => {
      const time = entry.ts || entry.timestamp || ''
      const levelName = entry.level || ''
      const message = entry.message || entry.msg || JSON.stringify(entry)
      return `${time} ${levelName} ${message}`.trim()
    })
    .join('\n')
}

function escapeHtml (value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
}

function activeTabName () {
  const active = document.querySelector('.tab.active')
  return active?.dataset?.tab || 'overview'
}

async function refresh () {
  $('footer-note').textContent = 'Loading…'
  try {
    const tab = activeTabName()
    if (tab === 'accounts') {
      await loadAccounts()
    } else if (tab === 'documents') {
      await loadDocuments()
    } else if (tab === 'volume') {
      await loadVolume()
    } else if (tab === 'mcp') {
      await loadMcp()
    } else if (tab === 'logs') {
      await loadLogs()
    } else {
      await loadOverview()
    }
    // Keep overview footer path fresh even when another tab is active.
    if (tab !== 'overview') {
      const status = await getJson('/api/admin/status')
      $('footer-note').textContent = `${status.service} · ${status.rag_data_root}`
    }
  } catch (error) {
    if (error.message !== 'Sign-in required') {
      $('footer-note').textContent = error.message
    }
  }
}

document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((item) => item.classList.remove('active'))
    document.querySelectorAll('.panel').forEach((panel) => panel.classList.remove('active'))
    tab.classList.add('active')
    $(`panel-${tab.dataset.tab}`).classList.add('active')
    refresh()
  })
})

$('refresh').addEventListener('click', refresh)
$('log-level').addEventListener('change', () => {
  loadLogs().catch((error) => showError($('log-view'), error))
})

refresh()
