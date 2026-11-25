# OpenRouter and development helpers (Cursor / Grok agents)

This is the briefing for **Cursor Cloud Agent Grok** (and other agents that only
see this git repository). It is not a contact channel to Marek's named Grok
bots on the operator box. Those IDs live in [AI_AGENTS.md](./AI_AGENTS.md).

OpenRouter is **optional development chat** plus an **opt-in figure-summary
LLM**. It is **not** the ingest or query embedding path.

Related: [inference roles ADR](./adr/inference-roles.md),
[NVIDIA NIM section in AI_AGENTS.md](./AI_AGENTS.md#nvidia-nim-cloud-agent--ingester-development).

## Do / do not

Do:

- Import helpers from ``rag.openrouter`` and ``rag.nim``. Do not put keys on
  argv or in logs.
- Use ``OPENROUTER_API_KEY`` for ``GET /api/v1/models`` and
  ``POST /api/v1/chat/completions``.
- Use ``OPENROUTER_MANAGEMENT_KEY`` only for BYOK (``GET/POST /api/v1/byok``).
- Pin BYOK with ``provider.only`` set to ``novita`` or ``nvidia``.
- Keep production text embeddings on Novita ``baai/bge-m3`` or FastEmbed
  (1024-dim). Same family for query.

Do not:

- Add ``openrouter`` as ``EMBEDDING_PROVIDER_INGEST`` or
  ``EMBEDDING_PROVIDER_QUERY``.
- Call chat with the management key (it 401s).
- Paste keys into chat, PRs, or docs.
- Copy Cursor environment secrets onto Railway through MCP unless Marek asks.
- Turn ``FIGURE_SUMMARY_LLM=openrouter`` on Railway unless Marek enables it.
- Replace Tesseract OCR with a vision model on this path.

## Secrets (names only)

| Name | Use |
| --- | --- |
| ``OPENROUTER_API_KEY`` | Inference: list models, chat completions |
| ``OPENROUTER_MANAGEMENT_KEY`` | BYOK attach only |
| ``OPENROUTER_MODEL`` | Optional default catalog model id |
| ``NVIDIA_NIM_API`` | Direct NVIDIA NIM (preferred name) |
| ``NVIDIA_NIM_APU`` | Same NIM key (this environment's injected spelling) |
| ``NVIDIA_API_KEY`` | NVIDIA catalog fallback name |
| ``NVIDIA_NIM_MODEL`` | Optional NIM catalog model id |
| ``BGE_M3_NOVITA_API_KEY`` | Prepaid Novita (embeddings and OpenRouter Novita BYOK) |
| ``NOVITA_API_KEY`` | Novita fallback if the BGE-named var is empty |

``GET /health`` reports ``openrouter.configured``,
``openrouter.management_configured``, and ``nvidia_nim.configured`` as
booleans only.

A typical Cursor Cloud Agent environment for this repo has the **management**
key and Novita/NIM provider keys, and may **lack** ``OPENROUTER_API_KEY``.
Chat helpers then return ``None``. Ask Marek to add the inference key as a
runtime secret; do not invent one.

## Python helpers

Module: [rag/openrouter.py](../rag/openrouter.py). Tests:
[tests/test_openrouter.py](../tests/test_openrouter.py).

| Helper | Needs | Returns / does |
| --- | --- | --- |
| ``openrouter_status()`` | nothing | Public flags, no secrets |
| ``openrouter_configured()`` | ``OPENROUTER_API_KEY`` | bool |
| ``openrouter_management_configured()`` | ``OPENROUTER_MANAGEMENT_KEY`` | bool |
| ``resolve_openrouter_model()`` | ``OPENROUTER_MODEL`` | catalog id or ``None`` |
| ``list_openrouter_models()`` | inference key | list of catalog ids; raises if key missing |
| ``openrouter_chat_completion(messages, model=..., provider_only=...)`` | inference key | assistant text or ``None`` (missing key, HTTP error, empty body) |
| ``list_openrouter_byok_providers()`` | management key | enabled slugs (``nvidia``, ``novita``, …) |
| ``ensure_openrouter_byok_credential(provider, key=..., name=...)`` | management key | ``created`` or ``exists`` |
| ``sync_openrouter_nvidia_novita_byok()`` | management + provider keys | per-provider status, no secret values |

CLI (same BYOK sync, JSON on stdout, no keys):

```text
python scripts/openrouter_byok_sync.py
```

NVIDIA NIM (direct, not OpenRouter): [rag/nim.py](../rag/nim.py)

| Helper | Needs | Returns / does |
| --- | --- | --- |
| ``nvidia_nim_status()`` | nothing | Public flags |
| ``nvidia_nim_configured()`` | ``NVIDIA_NIM_API`` / ``NVIDIA_NIM_APU`` / ``NVIDIA_API_KEY`` | bool |
| ``list_nim_models()`` | NIM key | catalog ids; base ``https://integrate.api.nvidia.com/v1`` |

## Chat call shape

``openrouter_chat_completion`` posts to
``https://openrouter.ai/api/v1/chat/completions`` with
``Authorization: Bearer $OPENROUTER_API_KEY``. When ``provider_only`` is
``novita`` or ``nvidia``, the body includes
``{"provider": {"only": ["novita"]}}`` (or ``nvidia``).

Default figure-summary model id: ``z-ai/glm-4.6``
(``OPENROUTER_FIGURE_MODEL``). Novita served that id on this account. Override
with ``FIGURE_SUMMARY_LLM_MODEL`` or ``OPENROUTER_MODEL``.

BYOK still bills the upstream provider plus OpenRouter's surcharge. Direct
Novita chat remains cheaper if you never need NVIDIA or failover.

## Optional figure LLM (ingester only)

[rag/figure_summary.py](../rag/figure_summary.py) stays **extractive** unless:

- ``FIGURE_SUMMARY_LLM=openrouter`` (or ``novita`` for direct Novita chat)
- ``FIGURE_SUMMARY_LLM_PROVIDER=novita`` (default) or ``nvidia``
- a per-document LLM budget remains
- the inference key is present and the reply parses as grounded JSON

Unknown ``FIGURE_SUMMARY_LLM`` values fail closed to ``off``. Failures fall
back to extractive text. Summaries are metadata on ``pdf_figure`` records; they
do not write vectors.

Default on Railway: leave the flag off.

## What is already attached

``nvidia`` and ``novita`` BYOK credentials were registered on the OpenRouter
account (PRs #66, #67, #68). Re-run ``python scripts/openrouter_byok_sync.py``
if a new environment has the management key and the provider keys; the helper
is idempotent (``exists`` if already enabled).
