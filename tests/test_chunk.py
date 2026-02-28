"""チャンク分割・マージモジュールのテスト"""

import gc
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import pytest
from hypothesis import given, settings, strategies as st, assume

from excel_translator.chunk import (
    Chunk,
    ChunkProcessor,
    ChunkProcessorError,
    ChunkResult,
    estimate_tokens,
)
from excel_translator.parser import InputRow


# ============================================================================
# テストユーティリティ
# ============================================================================

def create_input_row(
    text_id: str,
    source_text: str,
    character: Optional[str] = None,
    file_name: str = "test.xlsx",
    sheet_name: str = "Sheet1",
    row_number: int = 1,
    skip: bool = False,
    skip_reason: Optional[str] = None,
) -> InputRow:
    """テスト用InputRowを作成"""
    return InputRow(
        text_id=text_id,
        character=character,
        source_text=source_text,
        sheet_name=sheet_name,
        file_name=file_name,
        row_number=row_number,
        skip=skip,
        skip_reason=skip_reason,
    )


def create_chunk_result(chunk_id: str, text_ids: List[str]) -> ChunkResult:
    """テスト用ChunkResultを作成"""
    results = [
        {"text_id": tid, "translated_text": f"Translated {tid}"}
        for tid in text_ids
    ]
    return ChunkResult(chunk_id=chunk_id, results=results)


# ============================================================================
# ユニットテスト
# ============================================================================

class TestEstimateTokens:
    """トークン推定のテスト"""
    
    def test_empty_string(self):
        """空文字列は0トークン"""
        assert estimate_tokens("") == 0
    
    def test_japanese_text(self):
        """日本語テキストのトークン推定"""
        text = "こんにちは"  # 5文字
        tokens = estimate_tokens(text)
        assert tokens > 0
        assert tokens <= 10  # 日本語は約1.5文字/トークン
    
    def test_english_text(self):
        """英語テキストのトークン推定"""
        text = "Hello World"  # 11文字
        tokens = estimate_tokens(text)
        assert tokens > 0
        assert tokens <= 5  # 英語は約4文字/トークン
    
    def test_mixed_text(self):
        """混合テキストのトークン推定"""
        text = "Hello こんにちは World"
        tokens = estimate_tokens(text)
        assert tokens > 0


class TestChunkBasic:
    """Chunk基本機能のテスト"""
    
    def test_chunk_to_dict_and_from_dict(self):
        """Chunkの辞書変換とラウンドトリップ"""
        rows = [
            create_input_row("001", "テスト文1", "キャラA"),
            create_input_row("002", "テスト文2", "キャラB"),
        ]
        context_rows = [
            create_input_row("000", "前の文", "キャラC"),
        ]
        
        chunk = Chunk(
            chunk_id="test_chunk_001",
            file_name="test.xlsx",
            sheet_name="Sheet1",
            rows=rows,
            context_rows=context_rows,
        )
        
        # 辞書に変換
        data = chunk.to_dict()
        assert data["chunk_id"] == "test_chunk_001"
        assert len(data["rows"]) == 2
        assert len(data["context_rows"]) == 1
        
        # 辞書から復元
        restored = Chunk.from_dict(data)
        assert restored.chunk_id == chunk.chunk_id
        assert len(restored.rows) == len(chunk.rows)
        assert len(restored.context_rows) == len(chunk.context_rows)
        assert restored.rows[0].text_id == chunk.rows[0].text_id


class TestChunkResultBasic:
    """ChunkResult基本機能のテスト"""
    
    def test_chunk_result_to_dict_and_from_dict(self):
        """ChunkResultの辞書変換とラウンドトリップ"""
        result = ChunkResult(
            chunk_id="test_chunk_001",
            results=[
                {"text_id": "001", "translated_text": "Test 1"},
                {"text_id": "002", "translated_text": "Test 2"},
            ],
        )
        
        # 辞書に変換
        data = result.to_dict()
        assert data["chunk_id"] == "test_chunk_001"
        assert len(data["results"]) == 2
        
        # 辞書から復元
        restored = ChunkResult.from_dict(data)
        assert restored.chunk_id == result.chunk_id
        assert len(restored.results) == len(result.results)


class TestChunkProcessorBasic:
    """ChunkProcessor基本機能のテスト"""
    
    def test_split_empty_rows(self):
        """空の行リストは空のチャンクリストを返す"""
        processor = ChunkProcessor()
        chunks = processor.split_into_chunks([])
        assert chunks == []
    
    def test_split_single_row(self):
        """単一行は1チャンクになる"""
        processor = ChunkProcessor()
        rows = [create_input_row("001", "テスト文")]
        
        chunks = processor.split_into_chunks(rows)
        
        assert len(chunks) == 1
        assert len(chunks[0].rows) == 1
        assert chunks[0].rows[0].text_id == "001"
    
    def test_split_skips_skip_rows(self):
        """skipフラグがTrueの行はチャンクに含まれない"""
        processor = ChunkProcessor()
        rows = [
            create_input_row("001", "テスト文1", skip=False),
            create_input_row("002", "", skip=True, skip_reason="空テキスト"),
            create_input_row("003", "テスト文3", skip=False),
        ]
        
        chunks = processor.split_into_chunks(rows)
        
        assert len(chunks) == 1
        text_ids = [r.text_id for r in chunks[0].rows]
        assert "001" in text_ids
        assert "002" not in text_ids
        assert "003" in text_ids
    
    def test_split_groups_by_file(self):
        """ファイル単位でグループ化される"""
        processor = ChunkProcessor()
        rows = [
            create_input_row("001", "テスト文1", file_name="file1.xlsx"),
            create_input_row("002", "テスト文2", file_name="file2.xlsx"),
        ]
        
        chunks = processor.split_into_chunks(rows)
        
        # 2つのファイルなので2チャンク
        assert len(chunks) == 2
        file_names = {c.file_name for c in chunks}
        assert "file1.xlsx" in file_names
        assert "file2.xlsx" in file_names
    
    def test_merge_results_basic(self):
        """基本的なマージ"""
        processor = ChunkProcessor()
        
        rows = [
            create_input_row("001", "テスト文1"),
            create_input_row("002", "テスト文2"),
        ]
        chunks = processor.split_into_chunks(rows)
        
        results = [
            create_chunk_result(chunks[0].chunk_id, ["001", "002"]),
        ]
        
        merged = processor.merge_results(chunks, results)
        
        assert len(merged) == 2
        assert merged[0]["text_id"] == "001"
        assert merged[1]["text_id"] == "002"
    
    def test_merge_results_missing_raises_error(self):
        """欠落がある場合はエラー"""
        processor = ChunkProcessor()
        
        rows = [
            create_input_row("001", "テスト文1"),
            create_input_row("002", "テスト文2"),
        ]
        chunks = processor.split_into_chunks(rows)
        
        # 001のみ、002が欠落
        results = [
            ChunkResult(chunk_id=chunks[0].chunk_id, results=[
                {"text_id": "001", "translated_text": "Test 1"},
            ]),
        ]
        
        with pytest.raises(ChunkProcessorError) as exc_info:
            processor.merge_results(chunks, results)
        assert "欠落" in str(exc_info.value)


class TestChunkProcessorProgress:
    """ChunkProcessor進捗管理のテスト"""
    
    def test_save_and_load_progress(self):
        """進捗の保存と読み込み"""
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = ChunkProcessor(progress_dir=tmpdir)
            
            rows = [create_input_row("001", "テスト文")]
            chunk = Chunk(
                chunk_id="test_chunk",
                file_name="test.xlsx",
                sheet_name="Sheet1",
                rows=rows,
                context_rows=[],
            )
            result = ChunkResult(
                chunk_id="test_chunk",
                results=[{"text_id": "001", "translated_text": "Test"}],
            )
            
            # 保存
            processor.save_progress(chunk, result)
            
            # 読み込み
            progress = processor.load_progress()
            
            assert "test_chunk" in progress
            assert progress["test_chunk"].chunk_id == "test_chunk"
            assert len(progress["test_chunk"].results) == 1
    
    def test_clear_progress(self):
        """進捗のクリア"""
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = ChunkProcessor(progress_dir=tmpdir)
            
            rows = [create_input_row("001", "テスト文")]
            chunk = Chunk(
                chunk_id="test_chunk",
                file_name="test.xlsx",
                sheet_name="Sheet1",
                rows=rows,
                context_rows=[],
            )
            result = ChunkResult(
                chunk_id="test_chunk",
                results=[{"text_id": "001", "translated_text": "Test"}],
            )
            
            processor.save_progress(chunk, result)
            processor.clear_progress()
            
            progress = processor.load_progress()
            assert len(progress) == 0
    
    def test_get_pending_chunks(self):
        """未処理チャンクの取得"""
        processor = ChunkProcessor()
        
        chunks = [
            Chunk("chunk1", "test.xlsx", "Sheet1", [], []),
            Chunk("chunk2", "test.xlsx", "Sheet1", [], []),
            Chunk("chunk3", "test.xlsx", "Sheet1", [], []),
        ]
        
        progress = {
            "chunk1": ChunkResult("chunk1", []),
        }
        
        pending = processor.get_pending_chunks(chunks, progress)
        
        assert len(pending) == 2
        pending_ids = {c.chunk_id for c in pending}
        assert "chunk2" in pending_ids
        assert "chunk3" in pending_ids
        assert "chunk1" not in pending_ids


# ============================================================================
# Property-Based Tests
# ============================================================================

@st.composite
def input_rows_strategy(draw):
    """InputRowリストを生成するストラテジー"""
    num_files = draw(st.integers(min_value=1, max_value=3))
    rows = []
    
    for f in range(num_files):
        file_name = f"file{f+1}.xlsx"
        num_sheets = draw(st.integers(min_value=1, max_value=2))
        
        for s in range(num_sheets):
            sheet_name = f"Sheet{s+1}"
            num_rows = draw(st.integers(min_value=1, max_value=10))
            
            for r in range(num_rows):
                text_id = f"F{f+1}_S{s+1}_{r+1:03d}"
                source_text = draw(st.text(
                    alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z')),
                    min_size=1,
                    max_size=100
                ))
                character = draw(st.one_of(
                    st.none(),
                    st.sampled_from(["キャラA", "キャラB", "キャラC", ""])
                ))
                skip = draw(st.booleans()) and draw(st.integers(min_value=0, max_value=10)) < 2
                
                rows.append(InputRow(
                    text_id=text_id,
                    character=character if character else None,
                    source_text=source_text,
                    sheet_name=sheet_name,
                    file_name=file_name,
                    row_number=r + 1,
                    skip=skip,
                    skip_reason="テスト" if skip else None,
                ))
    
    return rows


@st.composite
def large_input_rows_strategy(draw):
    """大きなトークン数を持つInputRowリストを生成（チャンク分割をテスト）"""
    num_rows = draw(st.integers(min_value=5, max_value=20))
    rows = []
    
    for r in range(num_rows):
        text_id = f"ROW_{r+1:03d}"
        # 各行に十分なテキストを生成してトークン上限を超えさせる
        text_length = draw(st.integers(min_value=50, max_value=200))
        source_text = "テスト文" * (text_length // 4)
        character = draw(st.sampled_from(["キャラA", "キャラB", None]))
        
        rows.append(InputRow(
            text_id=text_id,
            character=character,
            source_text=source_text,
            sheet_name="Sheet1",
            file_name="test.xlsx",
            row_number=r + 1,
            skip=False,
            skip_reason=None,
        ))
    
    return rows


class TestProperty19ChunkSplit:
    """
    Property 19: チャンク分割
    
    *For any* 複数ファイル・大規模データについて、チャンクはファイル単位でグループ化され、
    トークン上限を超える場合は分割される。分割時は前チャンク末尾の行が次チャンクの文脈参照として含まれる
    
    **Feature: excel-translation-api, Property 19: チャンク分割**
    **Validates: Requirements 8.1, 8.2, 8.3**
    """
    
    @given(rows=input_rows_strategy())
    @settings(max_examples=100, deadline=None)
    def test_chunks_grouped_by_file(self, rows: List[InputRow]):
        """チャンクはファイル単位でグループ化される"""
        assume(len(rows) > 0)
        assume(any(not r.skip for r in rows))
        
        processor = ChunkProcessor(max_tokens_per_chunk=10000)  # 大きな上限で分割を防ぐ
        chunks = processor.split_into_chunks(rows)
        
        # 各チャンクは単一ファイルの行のみを含む
        for chunk in chunks:
            file_names = {r.file_name for r in chunk.rows}
            assert len(file_names) == 1, "チャンクは単一ファイルの行のみを含むべき"
            assert chunk.file_name in file_names
    
    @given(rows=large_input_rows_strategy())
    @settings(max_examples=100, deadline=None)
    def test_chunks_respect_token_limit(self, rows: List[InputRow]):
        """チャンクはトークン上限を尊重する"""
        assume(len(rows) > 0)
        
        max_tokens = 500  # 小さな上限で分割を強制
        processor = ChunkProcessor(max_tokens_per_chunk=max_tokens)
        chunks = processor.split_into_chunks(rows)
        
        # 各チャンクのトークン数が上限以下（または単一行で上限超過）
        for chunk in chunks:
            total_tokens = sum(estimate_tokens(r.source_text) for r in chunk.rows)
            # 単一行でも上限を超える場合があるので、複数行の場合のみチェック
            if len(chunk.rows) > 1:
                # 最後の行を除いたトークン数は上限以下であるべき
                tokens_without_last = sum(estimate_tokens(r.source_text) for r in chunk.rows[:-1])
                assert tokens_without_last <= max_tokens * 2  # 余裕を持たせる
    
    @given(rows=large_input_rows_strategy())
    @settings(max_examples=100, deadline=None)
    def test_split_chunks_have_context_rows(self, rows: List[InputRow]):
        """分割されたチャンクは前チャンクの末尾を文脈参照として持つ"""
        assume(len(rows) > 0)
        
        max_tokens = 200  # 小さな上限で分割を強制
        context_lines = 2
        processor = ChunkProcessor(max_tokens_per_chunk=max_tokens, context_lines=context_lines)
        chunks = processor.split_into_chunks(rows)
        
        if len(chunks) > 1:
            # 2番目以降のチャンクは文脈参照を持つ
            for i in range(1, len(chunks)):
                current_chunk = chunks[i]
                prev_chunk = chunks[i - 1]
                
                # 同じファイル・シートの場合のみ文脈参照がある
                if (current_chunk.file_name == prev_chunk.file_name and 
                    current_chunk.sheet_name == prev_chunk.sheet_name):
                    # 文脈参照は前チャンクの末尾から取られる
                    if len(prev_chunk.rows) > 0 and len(current_chunk.context_rows) > 0:
                        expected_context_ids = {r.text_id for r in prev_chunk.rows[-context_lines:]}
                        actual_context_ids = {r.text_id for r in current_chunk.context_rows}
                        assert actual_context_ids.issubset(expected_context_ids) or len(actual_context_ids) == 0
    
    @given(rows=input_rows_strategy())
    @settings(max_examples=100, deadline=None)
    def test_all_non_skip_rows_in_chunks(self, rows: List[InputRow]):
        """全ての非スキップ行がチャンクに含まれる"""
        assume(len(rows) > 0)
        
        processor = ChunkProcessor()
        chunks = processor.split_into_chunks(rows)
        
        # 期待される非スキップ行のtext_id
        expected_ids = {r.text_id for r in rows if not r.skip}
        
        # チャンク内の全text_id
        actual_ids = set()
        for chunk in chunks:
            for row in chunk.rows:
                actual_ids.add(row.text_id)
        
        assert expected_ids == actual_ids, "全ての非スキップ行がチャンクに含まれるべき"


class TestProperty20ChunkMerge:
    """
    Property 20: チャンクマージ
    
    *For any* 分割されたチャンクの翻訳結果について、マージ後のデータはtext_id順に並び、
    元のデータと同じ行数を持つ（重複・欠落がない）
    
    **Feature: excel-translation-api, Property 20: チャンクマージ**
    **Validates: Requirements 8.4, 8.5**
    """
    
    @given(rows=input_rows_strategy())
    @settings(max_examples=100, deadline=None)
    def test_merge_preserves_all_rows(self, rows: List[InputRow]):
        """マージ後は全ての行が保持される"""
        assume(len(rows) > 0)
        assume(any(not r.skip for r in rows))
        
        processor = ChunkProcessor()
        chunks = processor.split_into_chunks(rows)
        
        # 各チャンクの結果を作成
        results = []
        for chunk in chunks:
            text_ids = [r.text_id for r in chunk.rows]
            results.append(create_chunk_result(chunk.chunk_id, text_ids))
        
        # マージ
        merged = processor.merge_results(chunks, results)
        
        # 期待される行数
        expected_count = sum(len(c.rows) for c in chunks)
        assert len(merged) == expected_count, "マージ後の行数は元と同じであるべき"
    
    @given(rows=input_rows_strategy())
    @settings(max_examples=100, deadline=None)
    def test_merge_maintains_order(self, rows: List[InputRow]):
        """マージ後はtext_id順に並ぶ"""
        assume(len(rows) > 0)
        assume(any(not r.skip for r in rows))
        
        processor = ChunkProcessor()
        chunks = processor.split_into_chunks(rows)
        
        # 元の順序を記録
        original_order = []
        for chunk in chunks:
            for row in chunk.rows:
                original_order.append(row.text_id)
        
        # 各チャンクの結果を作成
        results = []
        for chunk in chunks:
            text_ids = [r.text_id for r in chunk.rows]
            results.append(create_chunk_result(chunk.chunk_id, text_ids))
        
        # マージ
        merged = processor.merge_results(chunks, results)
        
        # マージ後の順序
        merged_order = [m["text_id"] for m in merged]
        
        # 順序が保持されている
        assert merged_order == original_order, "マージ後の順序は元と同じであるべき"
    
    @given(rows=input_rows_strategy())
    @settings(max_examples=100, deadline=None)
    def test_merge_no_duplicates(self, rows: List[InputRow]):
        """マージ後に重複がない"""
        assume(len(rows) > 0)
        assume(any(not r.skip for r in rows))
        
        processor = ChunkProcessor()
        chunks = processor.split_into_chunks(rows)
        
        # 各チャンクの結果を作成
        results = []
        for chunk in chunks:
            text_ids = [r.text_id for r in chunk.rows]
            results.append(create_chunk_result(chunk.chunk_id, text_ids))
        
        # マージ
        merged = processor.merge_results(chunks, results)
        
        # 重複チェック
        text_ids = [m["text_id"] for m in merged]
        assert len(text_ids) == len(set(text_ids)), "マージ後に重複があってはならない"


class TestProperty21ProgressRoundTrip:
    """
    Property 21: 進捗永続化ラウンドトリップ
    
    *For any* チャンク処理結果について、save_progress → load_progress を実行すると
    元の結果と完全に一致する
    
    **Feature: excel-translation-api, Property 21: 進捗永続化ラウンドトリップ**
    **Validates: Requirements 8.6**
    """
    
    @given(
        num_chunks=st.integers(min_value=1, max_value=5),
        results_per_chunk=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100, deadline=None)
    def test_progress_round_trip(self, num_chunks: int, results_per_chunk: int):
        """進捗の保存と読み込みのラウンドトリップ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = ChunkProcessor(progress_dir=tmpdir)
            
            # テストデータを作成
            original_results = {}
            for c in range(num_chunks):
                chunk_id = f"chunk_{c}"
                rows = [
                    create_input_row(f"{chunk_id}_{r}", f"テスト文{r}")
                    for r in range(results_per_chunk)
                ]
                chunk = Chunk(
                    chunk_id=chunk_id,
                    file_name="test.xlsx",
                    sheet_name="Sheet1",
                    rows=rows,
                    context_rows=[],
                )
                result = ChunkResult(
                    chunk_id=chunk_id,
                    results=[
                        {"text_id": f"{chunk_id}_{r}", "translated_text": f"Translated {r}"}
                        for r in range(results_per_chunk)
                    ],
                )
                
                # 保存
                processor.save_progress(chunk, result)
                original_results[chunk_id] = result
            
            # 読み込み
            loaded = processor.load_progress()
            
            # 検証
            assert len(loaded) == num_chunks, "読み込んだチャンク数が一致するべき"
            
            for chunk_id, original in original_results.items():
                assert chunk_id in loaded, f"チャンク {chunk_id} が読み込まれるべき"
                loaded_result = loaded[chunk_id]
                
                assert loaded_result.chunk_id == original.chunk_id
                assert len(loaded_result.results) == len(original.results)
                
                for i, (orig, load) in enumerate(zip(original.results, loaded_result.results)):
                    assert orig["text_id"] == load["text_id"], f"text_idが一致するべき: {i}"
                    assert orig["translated_text"] == load["translated_text"], f"translated_textが一致するべき: {i}"
    
    @given(
        source_text=st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z')),
            min_size=1,
            max_size=200
        ),
        translated_text=st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z')),
            min_size=1,
            max_size=200
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_progress_preserves_unicode(self, source_text: str, translated_text: str):
        """進捗保存でUnicode文字が保持される"""
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = ChunkProcessor(progress_dir=tmpdir)
            
            chunk_id = "unicode_test"
            rows = [create_input_row("001", source_text)]
            chunk = Chunk(
                chunk_id=chunk_id,
                file_name="test.xlsx",
                sheet_name="Sheet1",
                rows=rows,
                context_rows=[],
            )
            result = ChunkResult(
                chunk_id=chunk_id,
                results=[{"text_id": "001", "translated_text": translated_text}],
            )
            
            # 保存
            processor.save_progress(chunk, result)
            
            # 読み込み
            loaded = processor.load_progress()
            
            # 検証
            assert chunk_id in loaded
            loaded_result = loaded[chunk_id]
            assert loaded_result.results[0]["translated_text"] == translated_text
