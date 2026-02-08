"""Tests for scripts/tokenize_fineweb_roberta.py.

python -m pytest tests/ -v --override-ini="addopts="
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Generator, Iterable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from transformers import RobertaTokenizerFast  # type: ignore

import scripts.tokenize_fineweb_roberta as script_mod
from scripts.tokenize_fineweb_roberta import (
    HEADER_SIZE,
    MAGIC_NUMBER,
    ROBERTA_VERSION,
    main,
    tokenize_document,
    write_shard,
)

FIXTURE_TOKENIZER = Path(__file__).parent / "fixtures" / "roberta-base"


@pytest.fixture
def tokenizer() -> RobertaTokenizerFast:
    return RobertaTokenizerFast.from_pretrained(str(FIXTURE_TOKENIZER))


class TestTokenizeDocument:
    """Tests for tokenize_document()."""

    @pytest.fixture(autouse=True)
    def _set_tokenizer(
        self, tokenizer: RobertaTokenizerFast, monkeypatch: Any
    ) -> Generator[None, None, None]:
        monkeypatch.setattr(script_mod, "_tokenizer", tokenizer)
        yield
        monkeypatch.setattr(script_mod, "_tokenizer", None)

    def test_appends_eos(self, tokenizer: RobertaTokenizerFast) -> None:
        result = tokenize_document({"text": "Hello world"})
        assert result[-1] == tokenizer.sep_token_id  # </s> = 2

    def test_no_bos_token(self, tokenizer: RobertaTokenizerFast) -> None:
        result = tokenize_document({"text": "Hello world"})
        assert result[0] != tokenizer.cls_token_id  # <s> = 0

    def test_returns_uint16(self) -> None:
        result = tokenize_document({"text": "Hello world"})
        assert result.dtype == np.uint16

    def test_assertion_without_init(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(script_mod, "_tokenizer", None)
        with pytest.raises(AssertionError, match="Tokenizer not initialized"):
            tokenize_document({"text": "Hello world"})


class TestWriteShard:
    """Tests for write_shard()."""

    def test_header_fields(
        self, tmp_path: Path, tokenizer: RobertaTokenizerFast
    ) -> None:
        tokens = np.array([100, 200, 300, 2], dtype=np.uint16)
        path = tmp_path / "test.bin"
        write_shard(path, tokens, tokenizer)

        header = np.fromfile(str(path), dtype=np.int32, count=HEADER_SIZE)
        assert header[0] == MAGIC_NUMBER  # 20240520
        assert header[1] == ROBERTA_VERSION  # 3
        assert header[2] == len(tokens)  # token_count
        assert header[3] == 50265  # vocab_size
        assert header[4] == tokenizer.pad_token_id  # 1
        assert header[5] == tokenizer.cls_token_id  # 0
        assert header[6] == tokenizer.sep_token_id  # 2
        assert header[7] == tokenizer.mask_token_id  # 50264

    def test_token_data_roundtrip(
        self, tmp_path: Path, tokenizer: RobertaTokenizerFast
    ) -> None:
        tokens = np.array([100, 200, 300, 2], dtype=np.uint16)
        path = tmp_path / "test.bin"
        write_shard(path, tokens, tokenizer)

        with open(path, "rb") as f:
            f.seek(HEADER_SIZE * 4)  # skip header (256 int32 = 1024 bytes)
            data = np.frombuffer(f.read(), dtype=np.uint16)
        np.testing.assert_array_equal(data, tokens)

    def test_file_size(self, tmp_path: Path, tokenizer: RobertaTokenizerFast) -> None:
        num_tokens = 1000
        tokens = np.arange(num_tokens, dtype=np.uint16)
        path = tmp_path / "test.bin"
        write_shard(path, tokens, tokenizer)

        expected_size = HEADER_SIZE * 4 + num_tokens * 2
        assert path.stat().st_size == expected_size

    def test_non_uint16_input_converted(
        self, tmp_path: Path, tokenizer: RobertaTokenizerFast
    ) -> None:
        tokens_int32 = np.array([100, 200, 300, 2], dtype=np.int32)
        path = tmp_path / "test.bin"
        write_shard(path, tokens_int32, tokenizer)

        with open(path, "rb") as f:
            f.seek(HEADER_SIZE * 4)
            data = np.frombuffer(f.read(), dtype=np.uint16)
        np.testing.assert_array_equal(data, tokens_int32.astype(np.uint16))


class TestMainPipeline:
    """End-to-end tests for main() with mocked dataset and multiprocessing."""

    @pytest.fixture(autouse=True)
    def _cleanup_tokenizer(self) -> Generator[None, None, None]:
        yield
        script_mod._tokenizer = None

    SAMPLE_DOCS = [
        {"text": "The quick brown fox jumps over the lazy dog."},
        {"text": "Machine learning models require large datasets."},
        {"text": "Python is a popular programming language."},
        {"text": "Natural language processing is a subfield of AI."},
        {"text": "Transformers have revolutionized deep learning."},
    ]

    def _make_mock_pool(self) -> MagicMock:
        """Create a mock multiprocessing context that runs in-process."""
        mock_ctx = MagicMock()

        def make_pool(
            num_workers: int, initializer: Callable | None = None, initargs: tuple = ()
        ) -> MagicMock:
            # Call the initializer in-process so _tokenizer is set
            if initializer is not None:
                initializer(*initargs)

            mock_pool = MagicMock()

            def fake_imap(
                fn: Callable, data: Iterable, chunksize: int = 1
            ) -> Iterable[Any]:
                return map(fn, data)

            mock_pool.imap = fake_imap
            return mock_pool

        mock_ctx.Pool = make_pool
        return mock_ctx

    def _run_main(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        extra_args: list[str] | None = None,
        docs: list[dict[str, str]] | None = None,
    ) -> Path:
        """Helper to run main() with mocked dependencies."""
        if docs is None:
            docs = self.SAMPLE_DOCS

        output_dir = tmp_path / "output"
        argv = [
            "tokenize_fineweb_roberta.py",
            "--version",
            "10B",
            "--output_dir",
            str(output_dir),
            "--shard_size",
            "50",
            "--tokenizer",
            str(FIXTURE_TOKENIZER),
            "--num_workers",
            "1",
        ]
        if extra_args:
            argv.extend(extra_args)

        monkeypatch.setattr(sys, "argv", argv)

        mock_ctx = self._make_mock_pool()

        with (
            patch.object(script_mod, "load_dataset", return_value=docs),
            patch.object(script_mod.mp, "get_context", return_value=mock_ctx),
        ):
            main()

        return output_dir

    def test_produces_shards(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        output_dir = self._run_main(monkeypatch, tmp_path)
        shards = sorted(output_dir.glob("*.bin"))
        assert len(shards) >= 1

    def test_val_shards_zero_all_train(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        output_dir = self._run_main(
            monkeypatch, tmp_path, extra_args=["--val_shards", "0"]
        )
        shards = sorted(output_dir.glob("*.bin"))
        assert len(shards) >= 1
        for s in shards:
            assert "_train_" in s.name
            assert "_val_" not in s.name

    def test_val_shards_two(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Use more docs and smaller shard_size to produce ≥3 shards
        docs = self.SAMPLE_DOCS * 5  # 25 docs
        output_dir = self._run_main(
            monkeypatch,
            tmp_path,
            extra_args=["--val_shards", "2"],
            docs=docs,
        )
        shards = sorted(output_dir.glob("*.bin"))
        assert len(shards) >= 3, f"Expected ≥3 shards, got {len(shards)}"

        val_shards = [s for s in shards if "_val_" in s.name]
        train_shards = [s for s in shards if "_train_" in s.name]
        assert len(val_shards) == 2
        assert len(train_shards) >= 1

    def test_shard_content_readable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        output_dir = self._run_main(monkeypatch, tmp_path)
        shards = sorted(output_dir.glob("*.bin"))
        assert len(shards) >= 1

        shard = shards[0]
        header = np.fromfile(str(shard), dtype=np.int32, count=HEADER_SIZE)
        assert header[0] == MAGIC_NUMBER
        assert header[1] == ROBERTA_VERSION

        token_count = header[2]
        with open(shard, "rb") as f:
            f.seek(HEADER_SIZE * 4)
            data = np.frombuffer(f.read(), dtype=np.uint16)
        assert len(data) == token_count

    def test_tail_shard_written(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Use a shard_size that won't evenly divide total tokens
        output_dir = tmp_path / "output_tail"
        argv = [
            "tokenize_fineweb_roberta.py",
            "--version",
            "10B",
            "--output_dir",
            str(output_dir),
            "--shard_size",
            "200",
            "--tokenizer",
            str(FIXTURE_TOKENIZER),
            "--num_workers",
            "1",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        mock_ctx = self._make_mock_pool()

        with (
            patch.object(script_mod, "load_dataset", return_value=self.SAMPLE_DOCS),
            patch.object(script_mod.mp, "get_context", return_value=mock_ctx),
        ):
            main()

        shards = sorted(output_dir.glob("*.bin"))
        assert len(shards) >= 2

        # The last shard should have fewer tokens than shard_size
        last_header = np.fromfile(str(shards[-1]), dtype=np.int32, count=HEADER_SIZE)
        last_token_count = last_header[2]
        assert 0 < last_token_count < 200
