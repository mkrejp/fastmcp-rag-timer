"""Fetch remote JSON/GeoJSON bytes with basic SSRF guards."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import requests

from rag.config import MAX_REMOTE_REDIRECTS, resolve_max_remote_json_bytes, resolve_remote_fetch_timeout

_USER_AGENT = "fastmcp-rag-timer/ingest-json"


class UnsafeRemoteUrlError(ValueError):
    """Raised when a URL is not a public http(s) target."""


class RemoteTooLargeError(ValueError):
    """Raised when the remote body exceeds the configured size limit."""


class RemoteFetchError(RuntimeError):
    """Raised when the remote host cannot be fetched."""


def normalize_download_url(url: str) -> str:
    """Turn GitHub blob pages into raw file URLs; leave other http(s) URLs as-is.

    Examples::

        https://github.com/siwekm/czech-geojson/blob/master/kraje.json
          -> https://raw.githubusercontent.com/siwekm/czech-geojson/master/kraje.json
    """
    stripped = url.strip()
    parsed = urlparse(stripped)
    if parsed.scheme in {"http", "https"} and parsed.netloc.lower() == "github.com":
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 5 and parts[2] == "blob":
            owner, repo, _blob, ref = parts[0], parts[1], parts[2], parts[3]
            rest = "/".join(parts[4:])
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{rest}"
        if len(parts) >= 5 and parts[2] == "raw":
            owner, repo, _raw, ref = parts[0], parts[1], parts[2], parts[3]
            rest = "/".join(parts[4:])
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{rest}"
    return stripped


def filename_from_url(url: str) -> str:
    """Return a filesystem-safe basename from ``url``, defaulting to ``remote.json``."""
    path = urlparse(url).path
    name = path.rsplit("/", 1)[-1] if path else ""
    name = name.split("?")[0].strip()
    if not name:
        return "remote.json"
    if "." not in name:
        return f"{name}.json"
    return name


def _assert_public_http_url(url: str) -> None:
    """Reject non-http(s) URLs and hosts that resolve to private addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise UnsafeRemoteUrlError("Only http and https URLs are allowed")
    if parsed.username or parsed.password:
        raise UnsafeRemoteUrlError("URLs with embedded credentials are not allowed")
    host = parsed.hostname
    if not host:
        raise UnsafeRemoteUrlError("URL host is required")
    lowered = host.lower().rstrip(".")
    if lowered in {"localhost", "metadata.google.internal", "metadata.google.com"}:
        raise UnsafeRemoteUrlError("URL host is not allowed")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UnsafeRemoteUrlError(f"Could not resolve host: {host}") from exc
    if not infos:
        raise UnsafeRemoteUrlError(f"Could not resolve host: {host}")
    for info in infos:
        ip_str = str(info[4][0])
        addr = ipaddress.ip_address(ip_str)
        if (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_multicast
            or addr.is_reserved
            or addr.is_unspecified
        ):
            raise UnsafeRemoteUrlError("URL resolves to a private or reserved address")


def fetch_remote_bytes(url: str) -> tuple[bytes, str]:
    """GET ``url`` (following a few redirects) and return ``(body, final_url)``."""
    current = normalize_download_url(url)
    timeout = resolve_remote_fetch_timeout()
    max_bytes = resolve_max_remote_json_bytes()
    last_error: Exception | None = None
    for _ in range(MAX_REMOTE_REDIRECTS + 1):
        _assert_public_http_url(current)
        try:
            response = requests.get(
                current,
                stream=True,
                timeout=timeout,
                allow_redirects=False,
                headers={"User-Agent": _USER_AGENT, "Accept": "application/json, application/geo+json, */*"},
            )
        except requests.RequestException as exc:
            raise RemoteFetchError(f"Failed to fetch URL: {exc}") from exc
        if response.is_redirect or response.status_code in {301, 302, 303, 307, 308}:
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise RemoteFetchError("Redirect without Location header")
            current = urljoin(current, location)
            continue
        try:
            if response.status_code >= 400:
                raise RemoteFetchError(f"Remote host returned HTTP {response.status_code}")
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    declared = int(content_length)
                except ValueError:
                    declared = 0
                if declared > max_bytes:
                    raise RemoteTooLargeError(
                        f"Remote JSON exceeds the {max_bytes} byte limit"
                    )
            chunks: list[bytes] = []
            total = 0
            for piece in response.iter_content(chunk_size=64 * 1024):
                if not piece:
                    continue
                total += len(piece)
                if total > max_bytes:
                    raise RemoteTooLargeError(
                        f"Remote JSON exceeds the {max_bytes} byte limit"
                    )
                chunks.append(piece)
            return b"".join(chunks), current
        finally:
            response.close()
        last_error = RemoteFetchError("Too many redirects")
    raise last_error or RemoteFetchError("Too many redirects")
