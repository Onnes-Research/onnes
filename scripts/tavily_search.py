"""
tavily_search.py — minimal Tavily search helper for the 2025-2026 research sweep.

Reads TAVILY_API_KEY from the environment (never printed). Uses only the stdlib
so there is nothing to install. Restricts results to recent, high-signal sources
(arXiv + GitHub by default) and biases the query toward 2025-2026 work.

Usage:
    TAVILY_API_KEY=... .venv/bin/python scripts/tavily_search.py \
        "many-shot in-context learning classification 2026" --arxiv --n 8

Or import:
    from scripts.tavily_search import search
    hits = search("multi-agent debate self-consistency 2026", arxiv=True)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import urllib.request

API = "https://api.tavily.com/search"


def search(query: str, *, n: int = 8, arxiv: bool = False, github: bool = False,
           depth: str = "advanced", recent_only: bool = True) -> list[dict]:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY not set in environment")
    domains = []
    if arxiv:
        domains.append("arxiv.org")
    if github:
        domains.append("github.com")
    payload = {
        "query": query,
        "search_depth": depth,
        "max_results": n,
        "include_answer": False,
        "include_raw_content": False,
    }
    if domains:
        payload["include_domains"] = domains
    if recent_only:
        # Tavily supports a coarse recency window; keep to the last ~2 years.
        payload["time_range"] = "year"
    req = urllib.request.Request(
        API, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=45) as r:
        body = json.loads(r.read().decode())
    return body.get("results", [])


def _cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--arxiv", action="store_true")
    ap.add_argument("--github", action="store_true")
    ap.add_argument("--json", action="store_true", help="emit raw JSON")
    args = ap.parse_args()
    try:
        hits = search(args.query, n=args.n, arxiv=args.arxiv, github=args.github)
    except Exception as exc:  # noqa: BLE001
        print(f"[tavily] ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(hits, indent=2))
        return 0
    for h in hits:
        print(f"\n• {h.get('title','')}\n  {h.get('url','')}")
        content = (h.get("content") or "").strip().replace("\n", " ")
        if content:
            print(f"  {content[:280]}")
    print(f"\n[tavily] {len(hits)} results for: {args.query}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
