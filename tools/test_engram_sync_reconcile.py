"""scan() reconcile-deletions: vanished files leave the index; unavailable
roots are never mass-deleted. Run: python3 tools/test_engram_sync_reconcile.py
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class ReconcileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="engram-sync-test-")
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        (root / "hub").mkdir()
        (root / "origin" / "research").mkdir(parents=True)
        os.environ["ENGRAM_SYNC_STATE"] = str(root / "state.json")
        self.addCleanup(os.environ.pop, "ENGRAM_SYNC_STATE", None)
        import importlib
        import engram_sync
        importlib.reload(engram_sync)  # pick up the env override
        self.sync = engram_sync
        # load_cfg() resolves paths in production; mirror that (macOS /var symlink)
        self.cfg = {
            "hub": str((root / "hub").resolve()),
            "rag_url": "http://test.invalid",
            "sources": [{"path": str((root / "origin").resolve()), "project": "demo"}],
        }
        self.deleted, self.ingested = [], []
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(self.sync, "api_ingest",
                          side_effect=lambda _u, c: self.ingested.append(c) or {}).start()
        mock.patch.object(self.sync, "api_delete",
                          side_effect=lambda _u, c: self.deleted.append(c) or {}).start()

    def _write(self, rel, text="# T\nbody\n"):
        p = Path(self.tmp.name) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    def test_vanished_files_are_reconciled_out(self):
        origin = self._write("origin/research/gone.md")
        hub_native = self._write("hub/global/notes.md")
        self.sync.scan(self.cfg)
        state = json.loads(Path(os.environ["ENGRAM_SYNC_STATE"]).read_text())
        self.assertIn(str(origin.resolve()), state)
        self.assertIn(str(hub_native.resolve()), state)

        origin.unlink()
        hub_native.unlink()
        # the mirrored hub copy of the origin must also disappear
        hub_copy = Path(self.cfg["hub"]) / "projects" / "demo" / "research" / "gone.md"
        self.assertTrue(hub_copy.exists())
        self.deleted.clear()
        self.sync.scan(self.cfg)

        state = json.loads(Path(os.environ["ENGRAM_SYNC_STATE"]).read_text())
        self.assertNotIn(str(origin.resolve()), state)
        self.assertNotIn(str(hub_native.resolve()), state)
        # two state keys reconcile: the origin (its delete also unlinks the
        # mirrored hub copy) and the hub-native file
        self.assertEqual(len(self.deleted), 2)
        self.assertFalse(hub_copy.exists())

    def test_unavailable_root_is_never_mass_deleted(self):
        origin = self._write("origin/research/keep.md")
        self.sync.scan(self.cfg)
        # simulate the whole source root being unmounted
        import shutil
        shutil.rmtree(Path(self.tmp.name) / "origin")
        self.deleted.clear()
        self.sync.scan(self.cfg)
        state = json.loads(Path(os.environ["ENGRAM_SYNC_STATE"]).read_text())
        self.assertIn(str(origin.resolve()), state)  # untouched
        self.assertEqual(self.deleted, [])           # and nothing deleted

    def test_stale_path_outside_all_roots_drops_state_only(self):
        stray = self._write("elsewhere/orphan.md")
        state_path = Path(os.environ["ENGRAM_SYNC_STATE"])
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({str(stray.resolve()): 1.0}))
        stray.unlink()
        self.sync.scan(self.cfg)
        state = json.loads(state_path.read_text())
        self.assertEqual(state, {})
        self.assertEqual(self.deleted, [])


if __name__ == "__main__":
    unittest.main()
