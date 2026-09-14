"""Download the NHANES .XPT files from the CDC for every configured cycle.

Files that are already downloaded are skipped, and a component that doesn't
exist for a cycle is logged instead of stopping the run.
"""
from __future__ import annotations

import logging
from pathlib import Path

import requests

from .config import RAW_DIR, get_cycles, load_file_registry

log = logging.getLogger(__name__)

_TIMEOUT = 60
_CHUNK = 1 << 16

# Every XPORT file starts with this. The CDC site answers a missing file with an
# HTML page and status 200, so the status code alone isn't enough to go on.
_XPORT_MAGIC = b"HEADER RECORD*******"


def _is_xport(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(len(_XPORT_MAGIC)) == _XPORT_MAGIC
    except OSError:
        return False


def _download_one(url: str, dest: Path) -> bool:
    """Download one file and check it really is an XPORT file."""
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
    except requests.RequestException as exc:
        log.error("failed %s: %s", url, exc)
        return False


def download_all(raw_dir: Path = RAW_DIR) -> dict[str, list[str]]:
    """Download every component for every cycle, into one folder per cycle."""
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
