"""Pack and unpack sync push chunks (target 32 MiB).

Chunk on-disk / wire format::

    <json-header>\\n\\n<raw-payload>

The header includes ``release_id``, ``stage``, ``seq``, ``sha256`` (of the
payload), and ``entries`` describing how to reassemble files or JSONL bytes.
A single oversized blob is split across multiple chunks via ``part``/``parts``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rag.publish.sync_constants import (
    BLOBS_STAGE,
    CHUNK_HARD_MAX_BYTES,
    CHUNK_SOFT_MAX_BYTES,
    CHUNK_TARGET_BYTES,
    POINTS_STAGE,
    STAGE_NAMES,
)

HEADER_SEP = b"\n\n"


@dataclass
class ChunkEntry:
    """One logical slice inside a chunk payload."""

    path: str
    offset: int
    length: int
    sha256: str
    part: int = 0
    parts: int = 1
    total_size: int = 0
    file_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the chunk header."""
        return {
            "path": self.path,
            "offset": self.offset,
            "length": self.length,
            "sha256": self.sha256,
            "part": self.part,
            "parts": self.parts,
            "total_size": self.total_size or self.length,
            "file_sha256": self.file_sha256 or self.sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChunkEntry:
        """Deserialize from a header entry."""
        return cls(
            path=str(data["path"]),
            offset=int(data["offset"]),
            length=int(data["length"]),
            sha256=str(data["sha256"]),
            part=int(data.get("part") or 0),
            parts=int(data.get("parts") or 1),
            total_size=int(data.get("total_size") or data.get("length") or 0),
            file_sha256=str(data.get("file_sha256") or data.get("sha256") or ""),
        )


@dataclass
class PackedChunk:
    """One ready-to-upload sync chunk."""

    release_id: str
    stage: str
    seq: int
    payload: bytes
    entries: list[ChunkEntry] = field(default_factory=list)
    sha256: str = ""

    def __post_init__(self) -> None:
        """Fill sha256 from payload when omitted."""
        if not self.sha256:
            self.sha256 = hashlib.sha256(self.payload).hexdigest()

    def header(self) -> dict[str, Any]:
        """Return the JSON header object."""
        return {
            "release_id": self.release_id,
            "stage": self.stage,
            "seq": self.seq,
            "sha256": self.sha256,
            "bytes": len(self.payload),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def encode(self) -> bytes:
        """Serialize header + payload for PUT / disk storage."""
        header = json.dumps(self.header(), ensure_ascii=False, separators=(",", ":"))
        body = header.encode("utf-8") + HEADER_SEP + self.payload
        if len(body) > CHUNK_HARD_MAX_BYTES:
            raise ValueError(
                f"chunk {self.stage}:{self.seq} exceeds hard max "
                f"({len(body)} > {CHUNK_HARD_MAX_BYTES})"
            )
        return body


def decode_chunk(raw: bytes) -> PackedChunk:
    """Parse a wire/disk chunk; validate payload sha256."""
    if len(raw) > CHUNK_HARD_MAX_BYTES:
        raise ValueError(
            f"chunk body exceeds hard max ({len(raw)} > {CHUNK_HARD_MAX_BYTES})"
        )
    sep = raw.find(HEADER_SEP)
    if sep < 0:
        raise ValueError("chunk missing header separator")
    try:
        header = json.loads(raw[:sep].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid chunk header: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("chunk header must be an object")
    stage = str(header.get("stage") or "")
    if stage not in STAGE_NAMES:
        raise ValueError(f"unknown chunk stage: {stage!r}")
    payload = raw[sep + len(HEADER_SEP) :]
    digest = hashlib.sha256(payload).hexdigest()
    expected = str(header.get("sha256") or "")
    if expected and expected != digest:
        raise ValueError(
            f"chunk sha256 mismatch for {stage}:{header.get('seq')}: "
            f"header={expected} actual={digest}"
        )
    entries = [
        ChunkEntry.from_dict(item)
        for item in (header.get("entries") or [])
        if isinstance(item, dict)
    ]
    return PackedChunk(
        release_id=str(header.get("release_id") or ""),
        stage=stage,
        seq=int(header.get("seq") or 0),
        payload=payload,
        entries=entries,
        sha256=digest,
    )


def _sha256_bytes(data: bytes) -> str:
    """Return hex digest for ``data``."""
    return hashlib.sha256(data).hexdigest()


def _split_blob_parts(
    data: bytes,
    *,
    path: str,
    file_sha256: str,
    max_part: int = CHUNK_TARGET_BYTES,
) -> list[tuple[bytes, ChunkEntry]]:
    """Split one blob into part payloads with entry metadata."""
    total = len(data)
    if total == 0:
        entry = ChunkEntry(
            path=path,
            offset=0,
            length=0,
            sha256=_sha256_bytes(b""),
            part=0,
            parts=1,
            total_size=0,
            file_sha256=file_sha256,
        )
        return [(b"", entry)]
    parts = max(1, (total + max_part - 1) // max_part)
    out: list[tuple[bytes, ChunkEntry]] = []
    for part in range(parts):
        start = part * max_part
        chunk = data[start : start + max_part]
        entry = ChunkEntry(
            path=path,
            offset=0,  # filled when packed into a chunk payload
            length=len(chunk),
            sha256=_sha256_bytes(chunk),
            part=part,
            parts=parts,
            total_size=total,
            file_sha256=file_sha256,
        )
        out.append((chunk, entry))
    return out


def pack_blob_chunks(
    release_id: str,
    files: Sequence[tuple[str, bytes, str]],
    *,
    target_bytes: int = CHUNK_TARGET_BYTES,
    soft_max_bytes: int = CHUNK_SOFT_MAX_BYTES,
) -> list[PackedChunk]:
    """Pack ``(rel_path, data, file_sha256)`` into ~target-sized blob chunks.

    Small files share a chunk; a single oversized file is split across parts.
    Soft max is preferred when packing; hard max is enforced by
    :meth:`PackedChunk.encode`.
    """
    if target_bytes <= 0:
        raise ValueError("target_bytes must be positive")
    # Flatten to part slices first.
    slices: list[tuple[bytes, ChunkEntry]] = []
    for path, data, file_sha in files:
        slices.extend(
            _split_blob_parts(
                data, path=path, file_sha256=file_sha, max_part=target_bytes
            )
        )

    chunks: list[PackedChunk] = []
    seq = 0
    buf = bytearray()
    entries: list[ChunkEntry] = []

    def flush() -> None:
        nonlocal seq, buf, entries
        if not entries and not buf:
            return
        chunks.append(
            PackedChunk(
                release_id=release_id,
                stage=BLOBS_STAGE,
                seq=seq,
                payload=bytes(buf),
                entries=list(entries),
            )
        )
        seq += 1
        buf = bytearray()
        entries = []

    for data, entry in slices:
        # Start a new chunk when adding would exceed soft max (unless empty).
        if buf and len(buf) + len(data) > soft_max_bytes:
            flush()
        elif buf and len(buf) + len(data) > target_bytes and len(buf) >= target_bytes:
            flush()
        placed = ChunkEntry(
            path=entry.path,
            offset=len(buf),
            length=len(data),
            sha256=entry.sha256,
            part=entry.part,
            parts=entry.parts,
            total_size=entry.total_size,
            file_sha256=entry.file_sha256,
        )
        buf.extend(data)
        entries.append(placed)
        # If a single part already exceeds target, flush immediately.
        if len(buf) >= target_bytes:
            flush()
    flush()
    return chunks


def pack_points_chunks(
    release_id: str,
    records: Iterable[Mapping[str, Any]],
    *,
    target_bytes: int = CHUNK_TARGET_BYTES,
    soft_max_bytes: int = CHUNK_SOFT_MAX_BYTES,
) -> list[PackedChunk]:
    """Pack point records as JSONL into ~target-sized ``points`` chunks."""
    lines: list[bytes] = []
    for record in records:
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        if len(line) > soft_max_bytes:
            # Extremely rare; still allow up to hard max as its own chunk.
            if len(line) > CHUNK_HARD_MAX_BYTES - 4096:
                raise ValueError(
                    f"single point record exceeds hard chunk budget ({len(line)} bytes)"
                )
        lines.append(line)

    chunks: list[PackedChunk] = []
    seq = 0
    buf = bytearray()

    def flush() -> None:
        nonlocal seq, buf
        if not buf:
            return
        digest = _sha256_bytes(bytes(buf))
        entry = ChunkEntry(
            path="qdrant/points.jsonl",
            offset=0,
            length=len(buf),
            sha256=digest,
            part=seq,
            parts=0,  # filled after packing
            total_size=0,
            file_sha256="",
        )
        chunks.append(
            PackedChunk(
                release_id=release_id,
                stage=POINTS_STAGE,
                seq=seq,
                payload=bytes(buf),
                entries=[entry],
                sha256=digest,
            )
        )
        seq += 1
        buf = bytearray()

    for line in lines:
        if buf and len(buf) + len(line) > soft_max_bytes:
            flush()
        elif buf and len(buf) + len(line) > target_bytes:
            flush()
        buf.extend(line)
    flush()

    total_parts = len(chunks)
    total_size = sum(len(c.payload) for c in chunks)
    file_sha = _sha256_bytes(b"".join(c.payload for c in chunks)) if chunks else ""
    for chunk in chunks:
        for entry in chunk.entries:
            entry.parts = total_parts or 1
            entry.total_size = total_size
            entry.file_sha256 = file_sha
    return chunks


def write_chunk_file(path: Path | str, chunk: PackedChunk) -> Path:
    """Write an encoded chunk to ``path``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(chunk.encode())
    return target


def read_chunk_file(path: Path | str) -> PackedChunk:
    """Load and validate a chunk from disk."""
    return decode_chunk(Path(path).read_bytes())


def assemble_blob_files(
    chunks: Sequence[PackedChunk],
    dest_root: Path | str,
) -> list[str]:
    """Reassemble blob chunks into ``dest_root/<path>``. Returns written paths."""
    root = Path(dest_root)
    root.mkdir(parents=True, exist_ok=True)
    # Collect parts per path: part_index -> bytes
    by_path: dict[str, dict[int, tuple[bytes, ChunkEntry]]] = {}
    for chunk in chunks:
        if chunk.stage != BLOBS_STAGE:
            continue
        for entry in chunk.entries:
            slice_bytes = chunk.payload[entry.offset : entry.offset + entry.length]
            if len(slice_bytes) != entry.length:
                raise ValueError(f"short payload for {entry.path} part {entry.part}")
            actual = _sha256_bytes(slice_bytes)
            if actual != entry.sha256:
                raise ValueError(
                    f"part sha256 mismatch for {entry.path} part {entry.part}"
                )
            by_path.setdefault(entry.path, {})[entry.part] = (slice_bytes, entry)

    written: list[str] = []
    for path, parts in sorted(by_path.items()):
        if not parts:
            continue
        sample = next(iter(parts.values()))[1]
        expected_parts = sample.parts
        if set(parts) != set(range(expected_parts)):
            raise ValueError(
                f"incomplete parts for {path}: have {sorted(parts)} "
                f"want 0..{expected_parts - 1}"
            )
        data = b"".join(parts[i][0] for i in range(expected_parts))
        if sample.file_sha256 and _sha256_bytes(data) != sample.file_sha256:
            raise ValueError(f"reassembled sha256 mismatch for {path}")
        if sample.total_size and len(data) != sample.total_size:
            raise ValueError(
                f"reassembled size mismatch for {path}: "
                f"{len(data)} != {sample.total_size}"
            )
        dest = root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        written.append(path)
    return written


def assemble_points_jsonl(
    chunks: Sequence[PackedChunk],
    dest: Path | str,
) -> Path | None:
    """Concatenate ``points`` chunk payloads into ``dest``. Returns path or None."""
    ordered = sorted(
        (c for c in chunks if c.stage == POINTS_STAGE),
        key=lambda c: c.seq,
    )
    if not ordered:
        return None
    target = Path(dest)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        for chunk in ordered:
            handle.write(chunk.payload)
    return target
