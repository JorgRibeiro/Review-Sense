"""Download testado sem rede: cache, integridade e arquivos incompletos."""

import hashlib
import io

import pytest

from reviewsense import data


def test_download_verifies_checksum_and_reuses_cache(tmp_path, monkeypatch):
    content = b"review_text,overall_rating\nbom,5\n"
    monkeypatch.setattr(data, "DATASET_SHA256", hashlib.sha256(content).hexdigest())
    calls = []

    def fake_open(url, timeout):
        calls.append(url)
        return io.BytesIO(content)

    monkeypatch.setattr(data, "urlopen", fake_open)
    path = data.download_b2w(tmp_path)
    assert path.read_bytes() == content
    assert data.download_b2w(tmp_path) == path
    assert calls == [data.DATASET_URL]
    loaded = data.load_b2w(cache_dir=tmp_path)
    assert loaded.attrs["source"]["revision"] == data.DATASET_REVISION


def test_corrupt_cache_is_rejected(tmp_path):
    (tmp_path / "B2W-Reviews01.csv").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="cache divergente"):
        data.download_b2w(tmp_path)


def test_bad_download_is_not_promoted_to_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: io.BytesIO(b"corrupt"))
    with pytest.raises(ValueError, match="Checksum inesperado"):
        data.download_b2w(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_interrupted_download_cleans_temporary_file(tmp_path, monkeypatch):
    def failed_download(*args, **kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr(data, "urlopen", failed_download)
    with pytest.raises(OSError, match="network unavailable"):
        data.download_b2w(tmp_path)
    assert list(tmp_path.iterdir()) == []
