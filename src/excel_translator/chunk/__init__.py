"""チャンク分割・マージモジュール"""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

from excel_translator.parser import InputRow

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """チャンクデータ"""
    chunk_id: str
    file_name: str
    sheet_name: str
    rows: List[InputRow]
    context_rows: List[InputRow] = field(default_factory=list)  # 文脈参照用（翻訳対象外）
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "chunk_id": self.chunk_id,
            "file_name": self.file_name,
            "sheet_name": self.sheet_name,
            "rows": [
                {
                    "text_id": r.text_id,
                    "character": r.character,
                    "source_text": r.source_text,
                    "sheet_name": r.sheet_name,
                    "file_name": r.file_name,
                    "row_number": r.row_number,
                    "skip": r.skip,
                    "skip_reason": r.skip_reason,
                }
                for r in self.rows
            ],
            "context_rows": [
                {
                    "text_id": r.text_id,
                    "character": r.character,
                    "source_text": r.source_text,
                    "sheet_name": r.sheet_name,
                    "file_name": r.file_name,
                    "row_number": r.row_number,
                    "skip": r.skip,
                    "skip_reason": r.skip_reason,
                }
                for r in self.context_rows
            ],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """辞書形式から復元"""
        rows = [
            InputRow(
                text_id=r["text_id"],
                character=r.get("character"),
                source_text=r["source_text"],
                sheet_name=r["sheet_name"],
                file_name=r["file_name"],
                row_number=r["row_number"],
                skip=r.get("skip", False),
                skip_reason=r.get("skip_reason"),
            )
            for r in data.get("rows", [])
        ]
        context_rows = [
            InputRow(
                text_id=r["text_id"],
                character=r.get("character"),
                source_text=r["source_text"],
                sheet_name=r["sheet_name"],
                file_name=r["file_name"],
                row_number=r["row_number"],
                skip=r.get("skip", False),
                skip_reason=r.get("skip_reason"),
            )
            for r in data.get("context_rows", [])
        ]
        return cls(
            chunk_id=data["chunk_id"],
            file_name=data["file_name"],
            sheet_name=data["sheet_name"],
            rows=rows,
            context_rows=context_rows,
        )


@dataclass
class ChunkResult:
    """チャンク処理結果"""
    chunk_id: str
    results: List[Dict[str, Any]]  # [{text_id, translated_text, ...}, ...]
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "chunk_id": self.chunk_id,
            "results": self.results,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkResult":
        """辞書形式から復元"""
        return cls(
            chunk_id=data["chunk_id"],
            results=data.get("results", []),
        )


class ChunkProcessorError(Exception):
    """チャンク処理エラー"""
    pass


def estimate_tokens(text: str) -> int:
    """
    テキストのトークン数を推定（簡易版）
    
    日本語: 約1.5文字/トークン
    英語: 約4文字/トークン
    混合テキストの場合は保守的に見積もる
    """
    if not text:
        return 0
    
    # 日本語文字数をカウント
    jp_chars = sum(1 for c in text if '\u3000' <= c <= '\u9fff' or '\u30a0' <= c <= '\u30ff')
    other_chars = len(text) - jp_chars
    
    # 日本語は1.5文字/トークン、その他は4文字/トークン
    jp_tokens = jp_chars / 1.5
    other_tokens = other_chars / 4
    
    return int(jp_tokens + other_tokens) + 1  # 切り上げ


class ChunkProcessor:
    """チャンク分割・マージクラス"""
    
    DEFAULT_MAX_TOKENS = 2000
    DEFAULT_CONTEXT_LINES = 3
    
    def __init__(
        self,
        max_tokens_per_chunk: int = DEFAULT_MAX_TOKENS,
        context_lines: int = DEFAULT_CONTEXT_LINES,
        progress_dir: str = ".translation_progress",
    ):
        """
        Args:
            max_tokens_per_chunk: チャンクあたりの最大トークン数
            context_lines: 文脈参照として含める前チャンク末尾の行数
            progress_dir: 進捗保存ディレクトリ
        """
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.context_lines = context_lines
        self.progress_dir = Path(progress_dir)
    
    def _generate_chunk_id(self, file_name: str, sheet_name: str, chunk_index: int) -> str:
        """チャンクIDを生成"""
        base = f"{file_name}_{sheet_name}_{chunk_index}"
        hash_suffix = hashlib.md5(base.encode()).hexdigest()[:8]
        return f"{Path(file_name).stem}_{sheet_name}_{chunk_index}_{hash_suffix}"
    
    def _estimate_row_tokens(self, row: InputRow) -> int:
        """行のトークン数を推定"""
        tokens = estimate_tokens(row.source_text)
        if row.character:
            tokens += estimate_tokens(row.character)
        tokens += estimate_tokens(row.text_id)
        return tokens
    
    def _group_by_file_and_sheet(
        self,
        rows: List[InputRow]
    ) -> Dict[str, Dict[str, List[InputRow]]]:
        """
        行をファイル・シート単位でグループ化
        
        Returns:
            {file_name: {sheet_name: [rows]}}
        """
        grouped: Dict[str, Dict[str, List[InputRow]]] = {}
        
        for row in rows:
            if row.file_name not in grouped:
                grouped[row.file_name] = {}
            if row.sheet_name not in grouped[row.file_name]:
                grouped[row.file_name][row.sheet_name] = []
            grouped[row.file_name][row.sheet_name].append(row)
        
        return grouped
    
    def split_into_chunks(self, rows: List[InputRow]) -> List[Chunk]:
        """
        行データをチャンクに分割
        
        - ファイル単位でグループ化
        - トークン上限を超える場合は分割
        - 分割時は前チャンク末尾を次チャンクの文脈参照として含める
        
        Args:
            rows: 入力行リスト
            
        Returns:
            チャンクリスト
        """
        if not rows:
            return []
        
        # ファイル・シート単位でグループ化
        grouped = self._group_by_file_and_sheet(rows)
        
        chunks: List[Chunk] = []
        
        for file_name in sorted(grouped.keys()):
            for sheet_name in sorted(grouped[file_name].keys()):
                sheet_rows = grouped[file_name][sheet_name]
                
                # 翻訳対象行のみを抽出（skipフラグがFalseの行）
                translatable_rows = [r for r in sheet_rows if not r.skip]
                
                if not translatable_rows:
                    logger.info(f"シート '{sheet_name}' に翻訳対象行がありません")
                    continue
                
                # トークン数に基づいてチャンクに分割
                current_chunk_rows: List[InputRow] = []
                current_tokens = 0
                chunk_index = 0
                previous_chunk_tail: List[InputRow] = []
                
                for row in translatable_rows:
                    row_tokens = self._estimate_row_tokens(row)
                    
                    # 現在のチャンクに追加するとトークン上限を超える場合
                    if current_chunk_rows and current_tokens + row_tokens > self.max_tokens_per_chunk:
                        # 現在のチャンクを確定
                        chunk_id = self._generate_chunk_id(file_name, sheet_name, chunk_index)
                        chunk = Chunk(
                            chunk_id=chunk_id,
                            file_name=file_name,
                            sheet_name=sheet_name,
                            rows=current_chunk_rows.copy(),
                            context_rows=previous_chunk_tail.copy(),
                        )
                        chunks.append(chunk)
                        
                        # 次のチャンクの文脈参照用に末尾を保存
                        previous_chunk_tail = current_chunk_rows[-self.context_lines:]
                        
                        # 新しいチャンクを開始
                        current_chunk_rows = [row]
                        current_tokens = row_tokens
                        chunk_index += 1
                    else:
                        current_chunk_rows.append(row)
                        current_tokens += row_tokens
                
                # 最後のチャンクを追加
                if current_chunk_rows:
                    chunk_id = self._generate_chunk_id(file_name, sheet_name, chunk_index)
                    chunk = Chunk(
                        chunk_id=chunk_id,
                        file_name=file_name,
                        sheet_name=sheet_name,
                        rows=current_chunk_rows,
                        context_rows=previous_chunk_tail,
                    )
                    chunks.append(chunk)
        
        logger.info(f"チャンク分割完了: {len(chunks)} チャンク")
        return chunks
    
    def merge_results(
        self,
        chunks: List[Chunk],
        results: List[ChunkResult]
    ) -> List[Dict[str, Any]]:
        """
        チャンク処理結果をtext_id順にマージ
        
        Args:
            chunks: 元のチャンクリスト
            results: チャンク処理結果リスト
            
        Returns:
            マージされた結果リスト（text_id順）
            
        Raises:
            ChunkProcessorError: 重複・欠落がある場合
        """
        # 期待されるtext_idを収集
        expected_text_ids: Set[str] = set()
        text_id_order: Dict[str, int] = {}  # text_id -> 元の順序
        order_counter = 0
        
        for chunk in chunks:
            for row in chunk.rows:
                if row.text_id not in expected_text_ids:
                    expected_text_ids.add(row.text_id)
                    text_id_order[row.text_id] = order_counter
                    order_counter += 1
        
        # 結果をtext_idでインデックス化
        result_map: Dict[str, Dict[str, Any]] = {}
        
        for chunk_result in results:
            for item in chunk_result.results:
                text_id = item.get("text_id")
                if not text_id:
                    logger.warning(f"text_idが空の結果をスキップ: {item}")
                    continue
                
                if text_id in result_map:
                    logger.warning(f"重複するtext_idを検出: {text_id}")
                
                result_map[text_id] = item
        
        # 重複チェック
        found_text_ids = set(result_map.keys())
        duplicates = len(results) - len(found_text_ids) if results else 0
        if duplicates > 0:
            logger.warning(f"重複した結果が {duplicates} 件あります")
        
        # 欠落チェック
        missing = expected_text_ids - found_text_ids
        if missing:
            logger.warning(f"欠落している結果が {len(missing)} 件あります: {list(missing)[:5]}...")
            raise ChunkProcessorError(
                f"結果に欠落があります: {len(missing)} 件のtext_idが見つかりません"
            )
        
        # 余分な結果チェック
        extra = found_text_ids - expected_text_ids
        if extra:
            logger.warning(f"余分な結果が {len(extra)} 件あります: {list(extra)[:5]}...")
        
        # text_id順にソートしてマージ
        merged = []
        for text_id in sorted(expected_text_ids, key=lambda x: text_id_order.get(x, 0)):
            if text_id in result_map:
                merged.append(result_map[text_id])
        
        logger.info(f"マージ完了: {len(merged)} 件")
        return merged
    
    def _get_progress_file_path(self, chunk_id: str) -> Path:
        """進捗ファイルのパスを取得"""
        return self.progress_dir / f"{chunk_id}.json"
    
    def save_progress(self, chunk: Chunk, result: ChunkResult) -> None:
        """
        チャンク処理結果を保存
        
        Args:
            chunk: チャンク
            result: 処理結果
        """
        # ディレクトリを作成
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存データを構築
        data = {
            "chunk": chunk.to_dict(),
            "result": result.to_dict(),
        }
        
        # JSONファイルに保存
        file_path = self._get_progress_file_path(chunk.chunk_id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"進捗を保存: {file_path}")
    
    def load_progress(self) -> Dict[str, ChunkResult]:
        """
        保存済みの進捗を読み込む
        
        Returns:
            {chunk_id: ChunkResult} の辞書
        """
        if not self.progress_dir.exists():
            return {}
        
        progress: Dict[str, ChunkResult] = {}
        
        for file_path in self.progress_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                result = ChunkResult.from_dict(data.get("result", {}))
                progress[result.chunk_id] = result
                
            except Exception as e:
                logger.warning(f"進捗ファイルの読み込みに失敗: {file_path}, エラー: {e}")
        
        logger.info(f"進捗を読み込み: {len(progress)} チャンク")
        return progress
    
    def clear_progress(self) -> None:
        """保存済みの進捗をクリア"""
        if not self.progress_dir.exists():
            return
        
        for file_path in self.progress_dir.glob("*.json"):
            try:
                file_path.unlink()
            except Exception as e:
                logger.warning(f"進捗ファイルの削除に失敗: {file_path}, エラー: {e}")
        
        logger.info("進捗をクリアしました")
    
    def get_pending_chunks(
        self,
        chunks: List[Chunk],
        progress: Dict[str, ChunkResult]
    ) -> List[Chunk]:
        """
        未処理のチャンクを取得
        
        Args:
            chunks: 全チャンクリスト
            progress: 保存済み進捗
            
        Returns:
            未処理チャンクリスト
        """
        pending = [c for c in chunks if c.chunk_id not in progress]
        logger.info(f"未処理チャンク: {len(pending)} / {len(chunks)}")
        return pending
