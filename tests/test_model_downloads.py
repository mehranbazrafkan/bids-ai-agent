"""Tests for model downloads: what counts as local, and the fetch job.

``is_downloaded`` reads the standard huggingface_hub cache layout -- one
``models--<org>--<name>`` folder per model, ``refs`` pointing at a
revision, ``snapshots`` holding the files, ``blobs`` holding the bytes.
A model copied in by hand must count as downloaded, and a half-fetched
download must not. ``start`` answers immediately and reports progress in
bytes; it never blocks on the fetch.
"""

import shutil
import time

import model_downloads
import pytest


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch, tmp_path):
    """The download state is module-global; every test starts idle and
    leaves nothing behind. The cache directory is pointed at a fresh
    tmp layout so no test reads or writes the real one."""
    monkeypatch.setattr(model_downloads, "_STATE", {
        "name": "", "status": "idle", "downloaded_bytes": 0,
        "total_bytes": None, "error": "", "started": 0.0, "finished": 0.0,
    })
    monkeypatch.setattr(model_downloads, "_THREAD", None)
    monkeypatch.setattr(model_downloads, "cache_dir", lambda: tmp_path / "hub")


def _make_model(tmp_path, name="Qwen/Qwen3-1.7B", *, revision="abc123",
                files=("config.json", "model.safetensors"), incomplete=False):
    """A synthetic cache folder in the standard HF layout."""
    root = tmp_path / "hub"
    repo = root / f"models--{name.replace('/', '--')}"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(revision, encoding="utf-8")
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True)
    for filename in files:
        target = snapshot / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * 10)
    if incomplete:
        blobs = repo / "blobs"
        blobs.mkdir(parents=True)
        (blobs / "deadbeef.incomplete").write_bytes(b"partial")
    return root


# ---------------------------------------------------------------------------
# What counts as local
# ---------------------------------------------------------------------------

class TestIsDownloaded:
    def test_a_complete_cache_folder_counts(self, tmp_path):
        root = _make_model(tmp_path)
        assert model_downloads.is_downloaded("Qwen/Qwen3-1.7B", root=root)

    def test_a_missing_folder_does_not(self, tmp_path):
        assert not model_downloads.is_downloaded("Qwen/Qwen3-1.7B",
                                                 root=tmp_path)

    def test_no_refs_means_not_ready(self, tmp_path):
        root = _make_model(tmp_path)
        shutil.rmtree(root / "models--Qwen--Qwen3-1.7B" / "refs")
        assert not model_downloads.is_downloaded("Qwen/Qwen3-1.7B", root=root)

    def test_an_empty_snapshot_does_not_count(self, tmp_path):
        root = _make_model(tmp_path, files=())
        assert not model_downloads.is_downloaded("Qwen/Qwen3-1.7B", root=root)

    def test_an_interrupted_download_does_not_count(self, tmp_path):
        """A half-fetched fetch leaves ``*.incomplete`` blobs behind;
        that model is not ready to use."""
        root = _make_model(tmp_path, incomplete=True)
        assert not model_downloads.is_downloaded("Qwen/Qwen3-1.7B", root=root)

    def test_the_folder_spelling_matches_the_hub_layout(self, tmp_path):
        root = _make_model(tmp_path, name="MehranBazrafkan/gemma-3-270m-it-bids-finetuned")
        assert model_downloads.is_downloaded(
            "MehranBazrafkan/gemma-3-270m-it-bids-finetuned", root=root)

    def test_downloaded_bytes_counts_the_blobs(self, tmp_path):
        root = _make_model(tmp_path)
        blobs = (root / "models--Qwen--Qwen3-1.7B" / "blobs")
        blobs.mkdir(parents=True)
        (blobs / "aaaa").write_bytes(b"z" * 1000)
        assert model_downloads.downloaded_bytes("Qwen/Qwen3-1.7B",
                                                root=root) == 1000

    def test_downloaded_bytes_of_nothing_is_zero(self, tmp_path):
        assert model_downloads.downloaded_bytes("Qwen/Qwen3-1.7B",
                                                root=tmp_path) == 0


# ---------------------------------------------------------------------------
# The fetch job
# ---------------------------------------------------------------------------

class TestStart:
    def test_start_never_blocks_and_reports_progress(self, tmp_path, monkeypatch):
        """start() must answer before the fetch finishes: the client
        polls for progress instead of waiting out a multi-GB download."""
        root = tmp_path / "hub"
        monkeypatch.setattr(model_downloads, "total_bytes", lambda name: 40)

        def fake_fetch(name):
            time.sleep(0.2)
            blob = (root / "models--Qwen--Qwen3-1.7B" / "blobs")
            blob.mkdir(parents=True, exist_ok=True)
            (blob / "bbbb").write_bytes(b"z" * 40)

        monkeypatch.setattr(model_downloads, "snapshot_download", fake_fetch)

        state = model_downloads.start("Qwen/Qwen3-1.7B")
        # Back before the fetch's 0.2s sleep is over, already downloading.
        assert state["status"] == "downloading"

        for _ in range(100):
            if model_downloads.status()["status"] != "downloading":
                break
            time.sleep(0.02)
        final = model_downloads.status()
        assert final["status"] == "done"
        assert final["downloaded_bytes"] == 40
        assert final["total_bytes"] == 40

    def test_a_failed_fetch_reports_the_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_downloads, "total_bytes", lambda name: 10)

        def failing_fetch(name):
            raise OSError("network gone")

        monkeypatch.setattr(model_downloads, "snapshot_download", failing_fetch)
        model_downloads.start("Qwen/Qwen3-1.7B")
        for _ in range(100):
            if model_downloads.status()["status"] != "downloading":
                break
            time.sleep(0.02)
        final = model_downloads.status()
        assert final["status"] == "error"
        assert "network gone" in final["error"]

    def test_a_blank_name_raises(self):
        with pytest.raises(ValueError):
            model_downloads.start("")

    def test_an_already_local_model_reports_done_without_fetching(
            self, tmp_path, monkeypatch):
        root = _make_model(tmp_path)

        def must_not_fetch(name):
            raise AssertionError("nothing to fetch")

        monkeypatch.setattr(model_downloads, "snapshot_download", must_not_fetch)
        state = model_downloads.start("Qwen/Qwen3-1.7B")
        assert state["status"] == "done"

    def test_a_running_download_is_not_raced(self, tmp_path, monkeypatch):
        """One at a time: a second repo while one is fetching answers
        with the active job rather than starting a parallel one."""
        monkeypatch.setattr(model_downloads, "total_bytes", lambda name: 10)

        def slow_fetch(name):
            time.sleep(0.3)

        monkeypatch.setattr(model_downloads, "snapshot_download", slow_fetch)
        model_downloads.start("Qwen/Qwen3-1.7B")
        second = model_downloads.start("Qwen/Qwen3-0.6B")
        assert second["name"] == "Qwen/Qwen3-1.7B"
        assert second["status"] == "downloading"
        for _ in range(100):
            if model_downloads.status()["status"] != "downloading":
                break
            time.sleep(0.02)
