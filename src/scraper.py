"""Fetch sxodim.com/almaty pages as markdown via Jina Reader.

Jina Reader (https://r.jina.ai/<url>) renders a page and returns clean
markdown, which sidesteps writing CSS selectors against a site that can change
its markup at any time. Raw pages are cached under data/raw/ so re-running the
extraction step does not re-hit the network.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from .config import JINA_READER, RAW_DIR, REQUEST_TIMEOUT, SXODIM_PAGES


def fetch_markdown(url: str, timeout: int = REQUEST_TIMEOUT) -> str:
    req = urllib.request.Request(
        JINA_READER + url,
        headers={"User-Agent": "shodim-almaty-agent/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def scrape_all(pages: dict[str, str] | None = None, out_dir: Path = RAW_DIR) -> dict[str, Path]:
    """Fetch every configured page, write it to out_dir, return the paths."""
    pages = pages or SXODIM_PAGES
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for name, url in pages.items():
        print(f"fetching {name}: {url}")
        try:
            markdown = fetch_markdown(url)
        except Exception as e:
            print(f"  [warn] {name} failed: {e}")
            continue

        path = out_dir / f"{name}.md"
        path.write_text(markdown, encoding="utf-8")
        written[name] = path
        print(f"  -> {path.name} ({len(markdown):,} chars)")

    if not written:
        raise RuntimeError("every page failed to fetch — check your connection")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    written = scrape_all(out_dir=args.out)
    total = sum(p.stat().st_size for p in written.values())
    print(f"\nscraped {len(written)} pages, {total:,} bytes total")


if __name__ == "__main__":
    main()
