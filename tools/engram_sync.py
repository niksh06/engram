#!/usr/bin/env python3
"""Engram sync watcher — keeps the hub + DuckDB index in sync with report sources.

Modes:
  watch          watchdog daemon: scan once for catch-up, then react to .md changes
  scan           one-shot reconcile: (re-)ingest everything changed since last run
  <file path>    sync a single file

For an *origin* file (a project repo's Reports/) it is copied into the hub under
~/Reports/projects/<project>/<report_type>/... (report_type from the source subdir),
preserving any description/tags already present in the hub copy, then ingested.
For a *hub-native* file (global/, handoffs/, directly-authored project files) it is
ingested as-is. Ingest is upsert-by-source_path, so re-runs never duplicate chunks.

Config: engram-sync.json next to this script (override via $ENGRAM_SYNC_CONFIG).
Deletions propagate on BOTH paths: live watchdog events (delete/move) and the
`scan` reconcile — files that vanished while the daemon was not watching are
removed from state and index on the next scan. Guard: a source root (or the
hub) that is currently unavailable is skipped entirely — its files merely
LOOK deleted, and reconcile must never mass-delete an unmounted corpus.
"""
import os, sys, re, json, time, datetime, urllib.parse, urllib.request, threading
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent
CFG = Path(os.environ.get("ENGRAM_SYNC_CONFIG", HERE / "engram-sync.json"))
STATE = Path(os.environ.get("ENGRAM_SYNC_STATE", Path.home() / ".engram" / "sync-state.json"))
ENRICH_QUEUE = Path.home() / ".engram" / "enrich-queue.txt"
LOG = Path.home() / ".engram" / "sync.log"

RTYPE = {"projects": "planning", "handoffs": "handoff"}  # else: source dir name as-is
RESERVED = {"readme.md", "index.md", "log.md"}
TEXT_EXT = {".md", ".markdown", ".mdx"}
FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)

_recent_writes = {}  # hub paths we just wrote (origin->hub), to skip the echo event


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def log(msg):
    line = f"{_now()} {msg}"
    print(line, flush=True)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_cfg():
    c = json.loads(CFG.read_text())
    c["hub"] = str(Path(c["hub"]).expanduser().resolve())
    for s in c["sources"]:
        s["path"] = str(Path(s["path"]).expanduser().resolve())
    return c


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(st):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st))


def strip_fm(raw):
    m = FM_RE.match(raw)
    if m:
        try:
            meta = yaml.safe_load(m.group(1))
            if isinstance(meta, dict):
                return meta, m.group(2)
        except Exception:
            pass
    return {}, raw


def first_h1(body):
    for ln in body.splitlines():
        s = ln.strip()
        if s.startswith("# "):
            return s[2:].strip()
    return None


def date_from_name(name):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    if m:
        return m.group(1)
    m = re.search(r"(\d{4})(\d{2})(\d{2})", name)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def container_path(hub, hub_file):
    return "/reports/" + str(Path(hub_file).resolve().relative_to(hub))


def api_ingest(rag_url, cpath):
    url = rag_url.rstrip("/") + "/api/ingest-report?" + urllib.parse.urlencode({"path": cpath})
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())


def api_delete(rag_url, cpath):
    url = rag_url.rstrip("/") + "/api/delete-report?" + urllib.parse.urlencode({"source_path": cpath})
    req = urllib.request.Request(url, method="DELETE")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def _queue_enrich(p):
    try:
        ENRICH_QUEUE.parent.mkdir(parents=True, exist_ok=True)
        with open(ENRICH_QUEUE, "a") as f:
            f.write(str(p) + "\n")
    except Exception:
        pass


def map_origin_to_hub(cfg, src, path):
    """(hub_target Path, report_type) for an origin file, or None to skip."""
    p = Path(path)
    if p.suffix.lower() not in TEXT_EXT or p.name.lower() in RESERVED:
        return None
    rel = p.resolve().relative_to(src["path"])
    parts = rel.parts
    if len(parts) > 1:
        rtype = RTYPE.get(parts[0], parts[0])
        rest = parts[1:]
    else:
        rtype, rest = "misc", (p.name,)
    return Path(cfg["hub"]) / "projects" / src["project"] / rtype / Path(*rest), rtype


def write_hub_copy(origin, hub_target, project, rtype):
    """Write origin body into hub_target with minimal OKF frontmatter, preserving any
    description/tags already present in the existing hub copy (enrichment survives)."""
    _, body = strip_fm(Path(origin).read_text(encoding="utf-8", errors="replace"))
    desc = tags = None
    if hub_target.exists():
        ex, _ = strip_fm(hub_target.read_text(encoding="utf-8", errors="replace"))
        desc, tags = ex.get("description"), ex.get("tags")
    mtime = os.path.getmtime(origin)
    meta = {
        "type": "Report",
        "title": first_h1(body) or Path(origin).stem.replace("_", " ").replace("-", " "),
        "project": project,
        "report_type": rtype,
        "report_date": date_from_name(Path(origin).name) or datetime.date.fromtimestamp(mtime).isoformat(),
        "timestamp": datetime.datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
        "resource": f"file://{origin}",
    }
    if desc:
        meta["description"] = desc
    if tags:
        meta["tags"] = tags
    hub_target.parent.mkdir(parents=True, exist_ok=True)
    fm = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip()
    hub_target.write_text(f"---\n{fm}\n---\n\n{body.lstrip()}\n", encoding="utf-8")
    _recent_writes[str(hub_target.resolve())] = time.time()
    return meta


def sync_path(cfg, path):
    path = str(Path(path).resolve())
    hub = cfg["hub"]
    # 1) origin file -> copy into hub, then ingest the hub copy
    for src in cfg["sources"]:
        if path == src["path"] or path.startswith(src["path"] + os.sep):
            if not os.path.isfile(path):
                return None
            mapped = map_origin_to_hub(cfg, src, path)
            if not mapped:
                return None
            hub_target, rtype = mapped
            meta = write_hub_copy(path, hub_target, src["project"], rtype)
            res = api_ingest(cfg["rag_url"], container_path(hub, hub_target))
            if not meta.get("tags"):
                _queue_enrich(hub_target)
            log(f"origin {src['project']}/{rtype} {Path(path).name} "
                f"(chunks={res.get('chunks_added')}, replaced={res.get('replaced_chunks')})")
            return res
    # 2) hub-native file -> ingest as-is
    if path == hub or path.startswith(hub + os.sep):
        p = Path(path)
        if p.suffix.lower() not in TEXT_EXT or p.name.lower() in RESERVED or not p.is_file():
            return None
        # skip the echo of an origin->hub write we just made
        t = _recent_writes.get(path)
        if t and time.time() - t < 5:
            return None
        res = api_ingest(cfg["rag_url"], container_path(hub, path))
        ex, _ = strip_fm(p.read_text(encoding="utf-8", errors="replace"))
        if not ex.get("tags"):
            _queue_enrich(p)
        log(f"hub {p.name} (chunks={res.get('chunks_added')}, replaced={res.get('replaced_chunks')})")
        return res
    return None


def delete_path(cfg, path):
    """Propagate a file deletion into the index: remove its chunks (and, for an origin
    file, the mirrored hub copy too)."""
    path = str(Path(path).resolve())
    hub = cfg["hub"]
    for src in cfg["sources"]:
        if path == src["path"] or path.startswith(src["path"] + os.sep):
            mapped = map_origin_to_hub(cfg, src, path)
            if not mapped:
                return None
            hub_target, _ = mapped
            try:
                hub_target.unlink()
            except FileNotFoundError:
                pass
            cpath = "/reports/" + str(hub_target.relative_to(hub))
            res = api_delete(cfg["rag_url"], cpath)
            log(f"delete origin {Path(path).name} -> {res.get('deleted_chunks')} chunks + hub copy")
            return res
    if path == hub or path.startswith(hub + os.sep):
        p = Path(path)
        if p.suffix.lower() not in TEXT_EXT or p.name.lower() in RESERVED:
            return None
        cpath = "/reports/" + str(p.relative_to(hub))
        res = api_delete(cfg["rag_url"], cpath)
        log(f"delete hub {p.name} -> {res.get('deleted_chunks')} chunks")
        return res
    return None


def scan(cfg):
    state = load_state()
    synced_hub, changed = set(), 0
    seen = set()  # every state-keyed path observed this run (reconcile basis)
    # origins first (write hub copies + ingest)
    for src in cfg["sources"]:
        base = Path(src["path"])
        if not base.is_dir():
            continue
        for f in base.rglob("*.md"):
            fp = str(f.resolve())
            mapped = map_origin_to_hub(cfg, src, fp)
            if not mapped:
                continue
            hub_target, rtype = mapped
            synced_hub.add(str(hub_target.resolve()))
            seen.add(fp)
            try:
                mt = os.path.getmtime(fp)
            except OSError:
                continue
            if state.get(fp) == mt:
                continue
            try:
                meta = write_hub_copy(fp, hub_target, src["project"], rtype)
                res = api_ingest(cfg["rag_url"], container_path(cfg["hub"], hub_target))
                if not meta.get("tags"):
                    _queue_enrich(hub_target)
                state[fp] = mt
                changed += 1
                log(f"scan origin {src['project']}/{rtype} {Path(fp).name} (chunks={res.get('chunks_added')})")
            except Exception as ex:
                log(f"ERROR origin {fp}: {ex}")
    # hub-native files not mirrored from an origin
    for f in Path(cfg["hub"]).rglob("*.md"):
        fp = str(f.resolve())
        seen.add(fp)
        if fp in synced_hub or f.name.lower() in RESERVED:
            continue
        try:
            mt = os.path.getmtime(fp)
        except OSError:
            continue
        if state.get(fp) == mt:
            continue
        try:
            res = api_ingest(cfg["rag_url"], container_path(cfg["hub"], fp))
            ex, _ = strip_fm(f.read_text(encoding="utf-8", errors="replace"))
            if not ex.get("tags"):
                _queue_enrich(f)
            state[fp] = mt
            changed += 1
            log(f"scan hub {f.name} (chunks={res.get('chunks_added')})")
        except Exception as ex:
            log(f"ERROR hub {fp}: {ex}")
    # reconcile deletions: files that vanished while the daemon was not
    # watching (sleep, crash window, other machine) used to stay in the
    # index forever — state only ever grew. Guard: never reconcile-delete
    # under a root that is unavailable right now; every file there merely
    # LOOKS deleted (unmounted disk would otherwise mass-delete a corpus).
    removed = 0
    hub = cfg["hub"]
    hub_available = Path(hub).is_dir()
    source_roots = {src["path"]: Path(src["path"]).is_dir() for src in cfg["sources"]}
    for fp in list(state):
        if fp in seen or os.path.exists(fp):
            continue
        root = next((r for r in source_roots
                     if fp == r or fp.startswith(r + os.sep)), None)
        if root is not None:
            if not source_roots[root]:
                continue  # root unavailable: skip, do not touch
        elif fp == hub or fp.startswith(hub + os.sep):
            if not hub_available:
                continue
        else:
            state.pop(fp)  # belongs to no configured root: drop state only
            continue
        try:
            delete_path(cfg, fp)
            state.pop(fp)
            removed += 1
            log(f"scan reconcile-delete {Path(fp).name}")
        except Exception as ex:
            log(f"ERROR reconcile-delete {fp}: {ex}")
    save_state(state)
    log(f"scan done: {changed} changed, {removed} reconciled deletions")
    return changed


def seed(cfg):
    """Record current mtimes WITHOUT ingesting — marks the already-migrated corpus as
    'known' so the watcher only acts on future changes (avoids re-mapping manually
    classified files, e.g. tparser root reports, into misc/)."""
    state = load_state()
    n = 0
    for src in cfg["sources"]:
        base = Path(src["path"])
        if not base.is_dir():
            continue
        for f in base.rglob("*.md"):
            if map_origin_to_hub(cfg, src, str(f)):
                state[str(f.resolve())] = os.path.getmtime(f)
                n += 1
    for f in Path(cfg["hub"]).rglob("*.md"):
        if f.name.lower() in RESERVED:
            continue
        state[str(f.resolve())] = os.path.getmtime(f)
        n += 1
    save_state(state)
    log(f"seeded {n} paths (no ingest)")
    return n


def watch(cfg):
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    log("startup catch-up scan...")
    try:
        scan(cfg)
    except Exception as ex:
        log(f"startup scan error: {ex}")
    pending, pending_del, lock = {}, {}, threading.Lock()

    class H(FileSystemEventHandler):
        def _on(self, path):
            if Path(path).suffix.lower() in TEXT_EXT:
                with lock:
                    pending[str(Path(path).resolve())] = time.time()

        def _del(self, path):
            if Path(path).suffix.lower() in TEXT_EXT:
                with lock:
                    pending_del[str(Path(path).resolve())] = time.time()

        def on_created(self, e):
            if not e.is_directory:
                self._on(e.src_path)

        def on_modified(self, e):
            if not e.is_directory:
                self._on(e.src_path)

        def on_deleted(self, e):
            if not e.is_directory:
                self._del(e.src_path)

        def on_moved(self, e):
            if not e.is_directory:
                self._del(e.src_path)      # old path gone
                self._on(e.dest_path)      # new path appeared

    obs = Observer()
    roots = [cfg["hub"]] + [s["path"] for s in cfg["sources"] if os.path.isdir(s["path"])]
    for r in roots:
        obs.schedule(H(), r, recursive=True)
        log(f"watching {r}")
    obs.start()
    try:
        while True:
            time.sleep(1.0)
            now, due, due_del = time.time(), [], []
            with lock:
                for p, t in list(pending.items()):
                    if now - t >= 1.5:  # debounce
                        due.append(p)
                        del pending[p]
                for p, t in list(pending_del.items()):
                    if now - t >= 1.5:
                        due_del.append(p)
                        del pending_del[p]
            for p in due:
                try:
                    sync_path(cfg, p)
                except Exception as ex:
                    log(f"ERROR sync {p}: {ex}")
            for p in due_del:
                if os.path.exists(p):  # reappeared (atomic save / rename-in) -> not a real delete
                    continue
                try:
                    delete_path(cfg, p)
                except Exception as ex:
                    log(f"ERROR delete {p}: {ex}")
    except KeyboardInterrupt:
        obs.stop()
    obs.join()


def main():
    cfg = load_cfg()
    arg = sys.argv[1] if len(sys.argv) > 1 else "watch"
    if arg == "watch":
        watch(cfg)
    elif arg == "scan":
        scan(cfg)
    elif arg == "seed":
        seed(cfg)
    else:
        print(json.dumps(sync_path(cfg, arg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
