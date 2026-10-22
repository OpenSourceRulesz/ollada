"""Local measurement records — roundtrip, max-merge, and refusal to
record implausible peaks. All in a tmp dir; nothing touches ~/.ollada."""

from ollada import measurements


def test_record_roundtrip_and_max_merge(tmp_path):
    assert measurements.load_record("dream:7b", tmp_path) is None
    measurements.record_peak("dream:7b", 9.5, hardware="GPU A", directory=tmp_path)
    record = measurements.record_peak("dream:7b", 8.1, hardware="GPU A", directory=tmp_path)
    assert record["peak_vram_gb"] == 9.5  # the highest seen is kept
    assert record["observations"] == 2
    record = measurements.record_peak("dream:7b", 10.25, directory=tmp_path)
    assert record["peak_vram_gb"] == 10.25
    assert record["hardware"] == "GPU A"  # kept when a later write omits it
    assert measurements.load_record("dream:7b", tmp_path)["peak_vram_gb"] == 10.25


def test_implausible_peak_is_not_recorded(tmp_path):
    assert measurements.record_peak("dream:7b", 0.01, directory=tmp_path) is None
    assert measurements.load_record("dream:7b", tmp_path) is None


def test_all_records_and_corrupt_files(tmp_path):
    measurements.record_peak("dream:7b", 9.8, directory=tmp_path)
    measurements.record_peak("sdar:4b", 8.87, directory=tmp_path)
    (tmp_path / "broken.json").write_text("{not json")
    records = measurements.all_records(tmp_path)
    assert set(records) == {"dream:7b", "sdar:4b"}
    assert measurements.load_record("broken", tmp_path) is None
