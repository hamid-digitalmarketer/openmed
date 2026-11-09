"""Offline tests for Maple ONNX Runtime bundle integrity."""

from __future__ import annotations

import json
import zipfile

import pytest

from openmed.onnx.maple_bundle import (
    MAPLE_BUNDLE_FILENAME,
    MapleBundleError,
    build_maple_onnx_bundle,
    validate_maple_onnx_bundle,
)


def _write_fake_export(root):
    files = {
        "decoder_model.ort": b"prefill-graph",
        "decoder_with_past_model.ort": b"decode-graph",
        "tokenizer.json": b'{"model":{"type":"BPE"}}',
        "tokenizer_config.json": b'{"model_max_length":131072}',
        "config.json": b'{"architectures":["MapleForCausalLM"]}',
    }
    for name, payload in files.items():
        (root / name).write_bytes(payload)


def _rewrite_manifest(original_path, destination, rewrite):
    """Copy a valid bundle, replacing the manifest text with ``rewrite`` output."""

    with zipfile.ZipFile(original_path) as source:
        original_manifest = source.read(MAPLE_BUNDLE_FILENAME).decode("utf-8")
        payloads = [
            (entry.filename, source.read(entry)) for entry in source.infolist()[1:]
        ]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as output:
        output.writestr(MAPLE_BUNDLE_FILENAME, rewrite(original_manifest))
        for name, payload in payloads:
            output.writestr(name, payload)
    return destination


def _mutate_manifest(original_path, destination, mutate):
    """Copy a valid bundle after applying ``mutate`` to its decoded manifest."""

    def rewrite(text):
        manifest = json.loads(text)
        mutate(manifest)
        return json.dumps(manifest)

    return _rewrite_manifest(original_path, destination, rewrite)


def _valid_bundle(tmp_path):
    source = tmp_path / "export"
    source.mkdir()
    _write_fake_export(source)
    return build_maple_onnx_bundle(source, tmp_path / "maple.zip")


def test_builds_android_bundle_with_manifest_first_and_exact_integrity(tmp_path):
    source = tmp_path / "export"
    source.mkdir()
    _write_fake_export(source)

    result = build_maple_onnx_bundle(source, tmp_path / "maple.ommaple.zip")

    assert result.manifest["architecture"] == "MapleForCausalLM"
    assert result.manifest["runtime"] == "onnxruntime-mobile"
    assert result.manifest["quantization"] == "qmoe-4bit-blockwise-128"
    assert result.total_size_bytes > 0
    with zipfile.ZipFile(result.bundle_path) as archive:
        assert archive.namelist()[0] == MAPLE_BUNDLE_FILENAME
        assert all(
            entry.compress_type == zipfile.ZIP_STORED for entry in archive.infolist()
        )
    assert validate_maple_onnx_bundle(result.bundle_path).manifest == result.manifest


def test_builds_web_bundle_with_single_decoder_graph(tmp_path):
    source = tmp_path / "export"
    source.mkdir()
    _write_fake_export(source)

    result = build_maple_onnx_bundle(
        source,
        tmp_path / "maple-web.zip",
        prefill_path="decoder_model.ort",
        decode_path=None,
        runtime="onnxruntime-web",
    )

    assert result.manifest["runtime"] == "onnxruntime-web"
    assert result.manifest["graphs"]["decode_path"] is None
    assert result.manifest["cache"] is None


def test_builds_stateless_android_bundle_accepted_by_mobile_contract(tmp_path):
    source = tmp_path / "export"
    source.mkdir()
    _write_fake_export(source)

    result = build_maple_onnx_bundle(
        source,
        tmp_path / "maple-mobile-stateless.zip",
        prefill_path="decoder_model.ort",
        decode_path=None,
        runtime="onnxruntime-mobile",
    )

    assert result.manifest["runtime"] == "onnxruntime-mobile"
    assert result.manifest["graphs"]["decode_path"] is None
    assert result.manifest["cache"] is None


def test_builds_unified_cached_decoder_without_duplicate_payload(tmp_path):
    source = tmp_path / "export"
    source.mkdir()
    _write_fake_export(source)

    result = build_maple_onnx_bundle(
        source,
        tmp_path / "maple-mobile-unified.zip",
        prefill_path="decoder_model.ort",
        decode_path="decoder_model.ort",
        runtime="onnxruntime-mobile",
    )

    assert result.manifest["graphs"]["prefill_path"] == "decoder_model.ort"
    assert result.manifest["graphs"]["decode_path"] == "decoder_model.ort"
    assert result.manifest["cache"] is not None
    declared_paths = [item["path"] for item in result.manifest["files"]]
    assert declared_paths.count("decoder_model.ort") == 1
    with zipfile.ZipFile(result.bundle_path) as archive:
        assert archive.namelist().count("decoder_model.ort") == 1


def test_refuses_overwrite_and_path_traversal(tmp_path):
    source = tmp_path / "export"
    source.mkdir()
    _write_fake_export(source)
    output = tmp_path / "maple.zip"
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="overwrite"):
        build_maple_onnx_bundle(source, output)
    with pytest.raises(MapleBundleError, match="unsafe bundle path"):
        build_maple_onnx_bundle(source, tmp_path / "new.zip", prefill_path="../x.ort")


def test_rejects_tampered_payload(tmp_path):
    source = tmp_path / "export"
    source.mkdir()
    _write_fake_export(source)
    original = build_maple_onnx_bundle(source, tmp_path / "maple.zip")
    tampered = tmp_path / "tampered.zip"

    with zipfile.ZipFile(original.bundle_path) as source_archive:
        with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_STORED) as output:
            for entry in source_archive.infolist():
                payload = source_archive.read(entry)
                if entry.filename == "decoder_model.ort":
                    payload = b"tampered-data"
                output.writestr(entry.filename, payload)

    with pytest.raises(MapleBundleError, match="checksum mismatch"):
        validate_maple_onnx_bundle(tampered)


def test_rejects_undeclared_archive_entry(tmp_path):
    source = tmp_path / "export"
    source.mkdir()
    _write_fake_export(source)
    original = build_maple_onnx_bundle(source, tmp_path / "maple.zip")
    invalid = tmp_path / "invalid.zip"

    with zipfile.ZipFile(original.bundle_path) as source_archive:
        manifest = json.loads(source_archive.read(MAPLE_BUNDLE_FILENAME))
        with zipfile.ZipFile(invalid, "w", compression=zipfile.ZIP_STORED) as output:
            output.writestr(MAPLE_BUNDLE_FILENAME, json.dumps(manifest))
            for entry in source_archive.infolist()[1:]:
                output.writestr(entry.filename, source_archive.read(entry))
            output.writestr("undeclared.bin", b"not allowed")

    with pytest.raises(MapleBundleError, match="exactly match"):
        validate_maple_onnx_bundle(invalid)


def test_rejects_duplicate_manifest_key(tmp_path):
    original = _valid_bundle(tmp_path)

    def duplicate_key(text):
        return text.replace("{", '{"schema_version": 1,', 1)

    hostile = _rewrite_manifest(
        original.bundle_path, tmp_path / "duplicate-key.zip", duplicate_key
    )

    with pytest.raises(MapleBundleError) as error:
        validate_maple_onnx_bundle(hostile)
    assert "unambiguous JSON" in str(error.value)
    assert "schema_version" not in str(error.value)


def test_rejects_non_finite_manifest_number(tmp_path):
    original = _valid_bundle(tmp_path)

    def non_finite(text):
        return text.replace('"schema_version": 1', '"schema_version": NaN', 1)

    hostile = _rewrite_manifest(
        original.bundle_path, tmp_path / "non-finite.zip", non_finite
    )

    with pytest.raises(MapleBundleError) as error:
        validate_maple_onnx_bundle(hostile)
    assert "unambiguous JSON" in str(error.value)
    assert "NaN" not in str(error.value)


def test_rejects_oversized_manifest(tmp_path):
    original = _valid_bundle(tmp_path)

    def oversized(text):
        padding = "0" * (1024**2)
        return text[:-1] + ', "padding": "' + padding + '"\n}'

    hostile = _rewrite_manifest(
        original.bundle_path, tmp_path / "oversized.zip", oversized
    )

    with pytest.raises(MapleBundleError) as error:
        validate_maple_onnx_bundle(hostile)
    assert "too large" in str(error.value)


def test_rejects_unknown_runtime(tmp_path):
    original = _valid_bundle(tmp_path)
    hostile = _mutate_manifest(
        original.bundle_path,
        tmp_path / "runtime.zip",
        lambda manifest: manifest.update(runtime="onnxruntime-desktop"),
    )

    with pytest.raises(MapleBundleError, match="runtime must be"):
        validate_maple_onnx_bundle(hostile)


def test_rejects_non_hex_checksum(tmp_path):
    original = _valid_bundle(tmp_path)
    hostile = _mutate_manifest(
        original.bundle_path,
        tmp_path / "checksum.zip",
        lambda manifest: manifest["files"][0].update(sha256="z" * 64),
    )

    with pytest.raises(MapleBundleError, match="valid SHA-256"):
        validate_maple_onnx_bundle(hostile)


def test_rejects_short_source_revision(tmp_path):
    original = _valid_bundle(tmp_path)
    hostile = _mutate_manifest(
        original.bundle_path,
        tmp_path / "revision.zip",
        lambda manifest: manifest.update(source_revision="abc123"),
    )

    with pytest.raises(MapleBundleError, match="immutable lowercase commit SHA"):
        validate_maple_onnx_bundle(hostile)
