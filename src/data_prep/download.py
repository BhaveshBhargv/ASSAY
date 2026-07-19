"""
download.py — fetch NHANES .XPT files from the CDC for all configured cycles.

Idempotent: skips files that already exist. Blood-pressure component base name
differs per cycle (BPX vs BPXO) and is resolved here. Missing optional files
(e.g. a component absent in one cycle) are logged and skipped, not fatal.
"""
from __future__ import annotations

import logging
from pathlib import Path

import requests

from .config import RAW_DIR, get_cycles, load_file_registry

log = logging.getLogger(__name__)

_TIMEOUT = 60
_CHUNK = 1 << 16

# Every SAS XPORT file begins with this magic. The CDC serves a soft-404 HTML
# page with HTTP 200 for missing files, so a status check is NOT enough — we
# must confirm the payload is genuinely an XPORT file.
_XPORT_MAGIC = b"HEADER RECORD*******"


def _is_xport(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(len(_XPORT_MAGIC)) == _XPORT_MAGIC
    except OSError:
        return False


def _download_one(url: str, dest: Path) -> bool:
    """Download a single file, validating it is a real XPORT payload."""
    if dest.exists() and _is_xport(dest):
        log.info("skip (valid, exists): %s", dest.name)
        return True
    try:
        with requests.get(url, stream=True, timeout=_TIMEOUT) as resp:
            if resp.status_code != 200:
                log.warning("unavailable (%s): %s", resp.status_code, url)
                return False
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(_CHUNK):
                    fh.write(chunk)
            if not _is_xport(tmp):
                tmp.unlink(missing_ok=True)
                log.error("not an XPORT file (soft-404?): %s", url)
                return False
            tmp.replace(dest)
        log.info("downloaded: %s", dest.name)
        return True
    except requests.RequestException as exc:  # network error
        log.error("failed %s: %s", url, exc)
        return False


def download_all(raw_dir: Path = RAW_DIR) -> dict[str, list[str]]:
    """
    Download every component for every cycle. Files are namespaced per cycle in
    subdirectories so identical base names across cycles never collide.
    """
    reg = load_file_registry()
    base_url = reg["base_url"].rstrip("/")
    bases: list[str] = list(reg["components"])

    obtained: dict[str, list[str]] = {}
    for cycle in get_cycles():
        cycle_dir = raw_dir / cycle.name
        cycle_dir.mkdir(parents=True, exist_ok=True)

        got: list[str] = []
        for base in bases:
            fname = cycle.filename(base)
            url = f"{base_url}/{cycle.dir}/DataFiles/{fname}"
            dest = cycle_dir / fname
            if _download_one(url, dest):
                got.append(fname)
        obtained[cycle.name] = got
        log.info("cycle %s: %d/%d files", cycle.name, len(got), len(bases))
    return obtained


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    download_all()
