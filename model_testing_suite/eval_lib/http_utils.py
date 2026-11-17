"""Download helpers for the evaluation suite.

Three building blocks, used by the per-dataset modules in eval_lib/datasets/:

1. ``simple_download``      - a plain, idempotent, retrying HTTP(S) download.
2. ``GoogleDriveDownloader`` - handles Google Drive's "file too large to scan
   for viruses" confirmation flow, which several of the required datasets
   route through (CrowdHuman images, Le2i raw videos). Falls back to a clear
   manual-download message instead of raising if Google Drive blocks it.
3. ``RemoteZipFile`` + ``extract_from_remote_zip`` - a file-like object backed
   by HTTP Range requests, so a handful of entries can be pulled out of a
   multi-gigabyte remote zip (e.g. MOT20.zip, 4.7GB) WITHOUT downloading the
   whole archive. Only used when the server advertises ``Accept-Ranges: bytes``
   (verified for motchallenge.net during research); falls back gracefully if
   the server does not support it.

Every function returns a small ``Result`` namedtuple (ok, status, message)
rather than raising, so a single dataset's network trouble never aborts the
whole evaluation run — this mirrors the "SKIPPED: <reason> / PARTIAL: <reason>"
status contract the rest of the suite relies on.
"""
from __future__ import annotations

import io
import os
import re
import sys
import time
import zipfile
from collections import namedtuple
from datetime import datetime

import requests

Result = namedtuple("Result", ["ok", "status", "message"])

DEFAULT_TIMEOUT = 30
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) VisionGuard-Eval/1.0 "
    "(research evaluation script; see README in external_datasets/)"
)


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def human_bytes(n: int) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def simple_download(url: str, dest_path: str, timeout: int = DEFAULT_TIMEOUT,
                     max_bytes: int | None = None, retries: int = 2,
                     force: bool = False, session: requests.Session | None = None) -> Result:
    """Idempotent streaming download. Skips if dest_path already exists and
    is non-empty, unless force=True. Truncates at max_bytes if given (used to
    cap default-subset downloads of files we only partially need)."""
    if not force and os.path.isfile(dest_path) and os.path.getsize(dest_path) > 0:
        return Result(True, "CACHED", f"already present: {dest_path}")

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    sess = session or _session()
    last_err = None
    for attempt in range(1, retries + 2):
        try:
            with sess.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length") or 0)
                written = 0
                tmp_path = dest_path + ".part"
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if not chunk:
                            continue
                        f.write(chunk)
                        written += len(chunk)
                        if max_bytes and written >= max_bytes:
                            break
                os.replace(tmp_path, dest_path)
                size_note = f" ({human_bytes(written)}" + (
                    f" of {human_bytes(total)} total, capped)" if max_bytes and total and written < total
                    else ")")
                return Result(True, "OK", f"downloaded {dest_path}{size_note}")
        except Exception as e:  # noqa: BLE001 - deliberately broad, this must never crash the run
            last_err = e
            if attempt <= retries:
                time.sleep(2 * attempt)
                continue
    return Result(False, "FAILED", f"download failed for {url}: {last_err}")


class GoogleDriveDownloader:
    """Handles the Google Drive "can't scan this file for viruses" interstitial
    that large (>~100MB) publicly-shared files trigger. No login/API key used
    — this only works for files shared as "Anyone with the link can view".

    Two known response shapes are handled:
      - classic: a download_warning_* cookie carries a confirm token, reused
        as a query parameter on a second request.
      - newer:   the HTML response contains a hidden-form "confirm"/"uuid"
        pair pointing at drive.usercontent.google.com; both are extracted
        with a regex and used to build the real download URL.

    If neither pattern is found (Google changed the flow again, or the file
    is quota-limited that day), this returns ok=False with a clear manual
    download message instead of silently producing a corrupt file.
    """

    BASE = "https://drive.google.com/uc?id={file_id}&export=download"
    CONFIRM_HOST = "https://drive.usercontent.google.com/download"

    @staticmethod
    def download(file_id: str, dest_path: str, timeout: int = 60,
                  max_bytes: int | None = None, force: bool = False) -> Result:
        if not force and os.path.isfile(dest_path) and os.path.getsize(dest_path) > 0:
            return Result(True, "CACHED", f"already present: {dest_path}")

        sess = _session()
        url = GoogleDriveDownloader.BASE.format(file_id=file_id)
        try:
            r = sess.get(url, timeout=timeout, allow_redirects=True)
        except Exception as e:  # noqa: BLE001
            return Result(False, "FAILED",
                          f"Google Drive request failed for id={file_id}: {e}. "
                          f"Manual download: https://drive.google.com/file/d/{file_id}/view "
                          f"-> place the file at {dest_path}")

        content_type = r.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            # Small file: served directly, no confirmation needed.
            return GoogleDriveDownloader._save(r, dest_path, max_bytes)

        html = r.text
        # Classic cookie-token pattern.
        token = None
        for name, value in sess.cookies.items():
            if name.startswith("download_warning"):
                token = value
                break
        if token:
            confirm_url = f"{GoogleDriveDownloader.BASE.format(file_id=file_id)}&confirm={token}"
            try:
                r2 = sess.get(confirm_url, timeout=timeout, stream=True)
                if "text/html" not in r2.headers.get("Content-Type", ""):
                    return GoogleDriveDownloader._save(r2, dest_path, max_bytes)
            except Exception:  # noqa: BLE001
                pass

        # Newer form-based pattern: look for confirm + uuid fields.
        m_confirm = re.search(r'name="confirm"\s+value="([^"]+)"', html)
        m_uuid = re.search(r'name="uuid"\s+value="([^"]+)"', html)
        if m_confirm and m_uuid:
            params = {"id": file_id, "export": "download",
                      "confirm": m_confirm.group(1), "uuid": m_uuid.group(1)}
            try:
                r3 = sess.get(GoogleDriveDownloader.CONFIRM_HOST, params=params,
                             timeout=timeout, stream=True)
                if "text/html" not in r3.headers.get("Content-Type", ""):
                    return GoogleDriveDownloader._save(r3, dest_path, max_bytes)
            except Exception:  # noqa: BLE001
                pass

        return Result(False, "FAILED",
                      f"Google Drive blocked automated download for id={file_id} "
                      f"(virus-scan interstitial could not be resolved automatically). "
                      f"MANUAL STEP: open https://drive.google.com/file/d/{file_id}/view, "
                      f"download it yourself, and place the file at: {dest_path}")

    @staticmethod
    def resolve_direct_url(file_id: str, timeout: int = 30, session: requests.Session | None = None):
        """Resolve the confirm-token interstitial WITHOUT downloading the
        body, returning (session, direct_url) for Range-based partial
        reads (e.g. via RemoteZipFile), or (None, None) if the flow
        couldn't be resolved (caller should fall back to a plain capped
        download / manual instructions)."""
        sess = session or _session()
        url = GoogleDriveDownloader.BASE.format(file_id=file_id)
        try:
            r = sess.get(url, timeout=timeout, stream=True)
        except Exception:  # noqa: BLE001
            return None, None
        content_type = r.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            r.close()
            return sess, r.url

        html = r.text
        r.close()
        m_confirm = re.search(r'name="confirm"\s+value="([^"]+)"', html)
        m_uuid = re.search(r'name="uuid"\s+value="([^"]+)"', html)
        if m_confirm and m_uuid:
            params = {"id": file_id, "export": "download",
                      "confirm": m_confirm.group(1), "uuid": m_uuid.group(1)}
            try:
                r2 = sess.head(GoogleDriveDownloader.CONFIRM_HOST, params=params,
                               timeout=timeout, allow_redirects=True)
                if r2.status_code < 400:
                    return sess, r2.url
            except Exception:  # noqa: BLE001
                pass
        token = None
        for name, value in sess.cookies.items():
            if name.startswith("download_warning"):
                token = value
                break
        if token:
            direct = f"{url}&confirm={token}"
            try:
                r3 = sess.head(direct, timeout=timeout, allow_redirects=True)
                if r3.status_code < 400:
                    return sess, r3.url
            except Exception:  # noqa: BLE001
                pass
        return None, None

    @staticmethod
    def _save(response, dest_path: str, max_bytes: int | None) -> Result:
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        written = 0
        tmp_path = dest_path + ".part"
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 16):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                if max_bytes and written >= max_bytes:
                    break
        os.replace(tmp_path, dest_path)
        return Result(True, "OK", f"downloaded {dest_path} ({human_bytes(written)})")


class RemoteZipFile(io.RawIOBase):
    """A read-only, seekable file-like object backed by HTTP Range requests.

    Lets ``zipfile.ZipFile`` open a multi-gigabyte remote archive and extract
    only the entries we actually need, without downloading the rest. Only
    works against servers advertising ``Accept-Ranges: bytes`` (checked by
    the caller before use).
    """

    #: minimum size of each underlying HTTP Range fetch. zipfile tends to
    #: issue many small sequential reads while decompressing an entry;
    #: batching them into >=256KB requests cuts the request-count overhead
    #: substantially on a slow/high-latency link without changing the total
    #: bytes transferred for the entries actually requested.
    MIN_FETCH = 256 * 1024

    def __init__(self, url: str, session: requests.Session | None = None,
                 timeout: int = 60):
        super().__init__()
        self.url = url
        self.session = session or _session()
        self.timeout = timeout
        self._pos = 0
        self._buf_start = -1
        self._buf = b""
        head = self.session.head(url, timeout=timeout, allow_redirects=True)
        head.raise_for_status()
        if head.headers.get("Accept-Ranges", "").lower() != "bytes":
            raise RuntimeError(f"Server for {url} does not advertise Accept-Ranges: bytes")
        self.length = int(head.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self.length + offset
        else:
            raise ValueError(f"bad whence {whence}")
        return self._pos

    def readinto(self, b):
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            # Whole-remainder reads only happen for tiny files; large ones
            # would defeat the point of range-reading, so cap defensively.
            size = min(self.MIN_FETCH, self.length - self._pos)
        if self._pos >= self.length:
            return b""

        buf_end = self._buf_start + len(self._buf)
        if self._buf_start <= self._pos < buf_end:
            # Serve what we can from the buffer first.
            offset = self._pos - self._buf_start
            available = self._buf[offset:offset + size]
            self._pos += len(available)
            if len(available) >= size:
                return available
            size -= len(available)
        else:
            available = b""

        fetch_size = max(size, self.MIN_FETCH)
        end = min(self._pos + fetch_size, self.length) - 1
        if end < self._pos:
            return available
        headers = {"Range": f"bytes={self._pos}-{end}"}
        r = self.session.get(self.url, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        fresh = r.content
        self._buf_start = self._pos
        self._buf = fresh
        want = fresh[:size]
        self._pos += len(want)
        return available + want


def extract_from_remote_zip(url: str, dest_dir: str, name_predicate,
                             limit: int | None = None,
                             timeout: int = 60,
                             session: requests.Session | None = None) -> Result:
    """Extract entries whose ``name_predicate(filename) is True`` from a
    remote zip, up to ``limit`` entries, without downloading the whole file.
    Falls back with a clear FAILED Result (caller decides what to do next —
    typically: fall back to a plain/capped ``simple_download`` of the whole
    file) if the server doesn't support Range or the zip can't be opened."""
    try:
        rf = RemoteZipFile(url, session=session, timeout=timeout)
        with zipfile.ZipFile(rf) as zf:
            names = [n for n in zf.namelist() if name_predicate(n)]
            if limit:
                names = names[:limit]
            os.makedirs(dest_dir, exist_ok=True)
            extracted = []
            for n in names:
                zf.extract(n, dest_dir)
                extracted.append(os.path.join(dest_dir, n))
            return Result(True, "OK",
                          f"range-extracted {len(extracted)} file(s) from {url} "
                          f"(zip is {human_bytes(rf.length)}, full download skipped)")
    except Exception as e:  # noqa: BLE001
        return Result(False, "FAILED", f"remote-zip range extraction failed for {url}: {e}")
