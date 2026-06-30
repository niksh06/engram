#!/usr/bin/env python3
"""okf_stamp — write/ensure OKF frontmatter on a report, AT THE SOURCE.

Shared by project report-writers so machine-generated reports are OKF-native
(Engram profile — see engram/docs/ADR-0002) instead of relying on ingest-time stamping.
Missing fields are derived from the path (.../projects/<project>/<report_type>/...),
filename date, and first H1. `tags`/`description` are optional (enrichment may add them later).

Library:
    import sys; sys.path.append("/Users/nsh/Downloads/engram/tools")
    from okf_stamp import write_report, stamp
    write_report("~/Reports/projects/aleph/analysis/2026-07-01-x.md", body,
                 project="aleph", report_type="analysis", title="…",
                 tags=["cve", "vdb"], description="…")
    stamp("path/to/existing.md", project="aleph")        # add header to an existing file

CLI:
    okf_stamp.py FILE [--project P] [--report-type T] [--title T] [--date YYYY-MM-DD]
                 [--tags a,b,c] [--description D] [--type Report] [--force]
"""
import os, re, sys, json, argparse, datetime
from pathlib import Path
try:
    import yaml
except ImportError:
    yaml = None

# Recommended (free-form) report types; not enforced.
REPORT_TYPES = {"daily", "weekly", "analysis", "planning", "implementation",
                "premortem", "reference", "archive", "handoff"}
FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


def _split_fm(raw):
    m = FM_RE.match(raw)
    if m and yaml:
        try:
            meta = yaml.safe_load(m.group(1))
            if isinstance(meta, dict):
                return meta, m.group(2)
        except Exception:
            pass
    return {}, raw


def _derive(abspath):
    """(project, report_type) from .../projects/<project>/<report_type>/... or .../global/<type>/..."""
    parts = list(Path(abspath).parts)
    lower = [p.lower() for p in parts]
    project = rtype = None
    for key in ("projects", "global"):
        if key in lower:
            i = lower.index(key)
            if key == "global":
                project, base = "global", i
            else:
                if i + 1 <= len(parts) - 1:
                    project = parts[i + 1]
                base = i + 1
            if base + 1 <= len(parts) - 2:
                rtype = parts[base + 1]
            break
    return project, rtype


def _date_from_name(name):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    if m:
        return m.group(1)
    m = re.search(r"(\d{4})(\d{2})(\d{2})", name)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def _first_h1(body):
    for ln in body.splitlines():
        s = ln.strip()
        if s.startswith("# "):
            return s[2:].strip()
    return None


def build_frontmatter(abspath, body, existing=None, project=None, report_type=None,
                      title=None, report_date=None, tags=None, description=None,
                      doc_type="Report"):
    """Build the OKF metadata dict. Precedence: explicit arg > existing frontmatter > derived."""
    existing = existing or {}
    dp, dt = _derive(abspath)
    mtime = os.path.getmtime(abspath) if os.path.isfile(abspath) else None
    today = datetime.date.today().isoformat()  # only used when nothing else available
    meta = {
        "type": doc_type or existing.get("type") or "Report",
        "title": title or existing.get("title") or _first_h1(body)
                 or os.path.splitext(os.path.basename(abspath))[0],
        "project": project or existing.get("project") or dp,
        "report_type": report_type or existing.get("report_type") or dt,
        "report_date": str(report_date or existing.get("report_date")
                           or _date_from_name(os.path.basename(abspath))
                           or (datetime.date.fromtimestamp(mtime).isoformat() if mtime else today)),
        "tags": tags if tags is not None else existing.get("tags"),
        "description": description or existing.get("description"),
        "timestamp": existing.get("timestamp")
                     or (datetime.datetime.fromtimestamp(mtime).isoformat(timespec="seconds") if mtime else None),
    }
    if isinstance(meta["tags"], str):
        meta["tags"] = [t.strip() for t in meta["tags"].split(",") if t.strip()]
    return {k: v for k, v in meta.items() if v is not None}


def _render(meta, body):
    if yaml:
        fm = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip()
    else:  # degraded, still valid-ish YAML
        fm = "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in meta.items())
    return f"---\n{fm}\n---\n\n{body.strip()}\n"


def stamp(path, force=False, **meta_kw):
    """Ensure OKF frontmatter on an existing report file. No-op if already stamped and no
    overrides given (unless force). Returns the written/kept metadata."""
    p = Path(path).expanduser()
    raw = p.read_text(encoding="utf-8", errors="replace")
    existing, body = _split_fm(raw)
    has_overrides = any(v is not None for v in meta_kw.values())
    if existing and not force and not has_overrides:
        return existing
    meta = build_frontmatter(str(p), body, existing=existing, **meta_kw)
    p.write_text(_render(meta, body), encoding="utf-8")
    return meta


def write_report(path, body, **meta_kw):
    """Write a NEW report (OKF frontmatter + body) to path, creating parent dirs."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = build_frontmatter(str(p), body, **meta_kw)
    p.write_text(_render(meta, body), encoding="utf-8")
    return meta


def main():
    ap = argparse.ArgumentParser(description="Stamp OKF frontmatter on a report file.")
    ap.add_argument("file")
    ap.add_argument("--project")
    ap.add_argument("--report-type", dest="report_type")
    ap.add_argument("--title")
    ap.add_argument("--date", dest="report_date")
    ap.add_argument("--tags")
    ap.add_argument("--description")
    ap.add_argument("--type", dest="doc_type", default="Report")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    meta = stamp(a.file, force=a.force, project=a.project, report_type=a.report_type,
                 title=a.title, report_date=a.report_date,
                 tags=([t.strip() for t in a.tags.split(",")] if a.tags else None),
                 description=a.description, doc_type=a.doc_type)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
