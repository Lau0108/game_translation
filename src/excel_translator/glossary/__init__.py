"""用語集管理モジュール"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GlossaryEntry:
    """用語集エントリ"""
    term_source: str  # 原文用語
    term_target: str  # 訳語
    category: str = ""  # カテゴリ（キャラ名、アイテム名、地名等）
    context_note: Optional[str] = None  # 文脈メモ
    do_not_translate: bool = False  # 翻訳禁止フラグ


@dataclass
class GlossaryVerificationResult:
    """用語一貫性検証結果"""
    term_source: str
    term_target: str
    found_in_source: bool
    found_in_target: bool
    is_consistent: bool
    message: str


class GlossaryManagerError(Exception):
    """用語集管理エラー"""
    pass


# カラム名エイリアス辞書
GLOSSARY_COLUMN_ALIASES = {
    "term_source": ["term_source", "原文", "source", "原語", "日本語", "jp", "ja", "japanese"],
    "term_target": ["term_target", "訳語", "target", "翻訳", "英語", "en", "english", "translation"],
    "category": ["category", "カテゴリ", "分類", "type", "種別"],
    "context_note": ["context_note", "備考", "note", "メモ", "context", "説明"],
    "do_not_translate": ["do_not_translate", "翻訳禁止", "dnt", "no_translate", "禁止"],
}


class GlossaryManager:
    """用語集管理クラス"""
    
    def __init__(self):
        self._entries: List[GlossaryEntry] = []
        self._column_aliases = GLOSSARY_COLUMN_ALIASES.copy()
    
    def load(self, file_path: str, sheet_name: Optional[str] = None) -> List[GlossaryEntry]:
        """
        用語集を読み込む
        
        Args:
            file_path: 用語集ファイルパス（Excel）
            sheet_name: シート名（省略時は最初のシート）
            
        Returns:
            GlossaryEntryのリスト
        """
        path = Path(file_path)
        if not path.exists():
            raise GlossaryManagerError(f"用語集ファイルが見つかりません: {file_path}")
        
        if not path.suffix.lower() in [".xlsx", ".xls", ".xlsm"]:
            raise GlossaryManagerError(f"サポートされていないファイル形式: {path.suffix}")
        
        try:
            # Excelファイルを読み込み
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name, dtype=str)
            else:
                df = pd.read_excel(file_path, dtype=str)
            
            # カラムマッピング
            df = self._map_columns(df)
            
            # エントリに変換
            entries = self._parse_entries(df)
            
            self._entries = entries
            logger.info(f"用語集を読み込みました: {len(entries)} 件")
            
            return entries
            
        except Exception as e:
            raise GlossaryManagerError(f"用語集の読み込みに失敗しました: {e}")
    
    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        カラムを正規名にマッピング
        
        Args:
            df: DataFrame
            
        Returns:
            マッピング済みDataFrame
        """
        new_columns = {}
        
        for col in df.columns:
            col_str = str(col).strip().lower() if pd.notna(col) else ""
            mapped = False
            
            for canonical, aliases in self._column_aliases.items():
                aliases_lower = [a.lower() for a in aliases]
                if col_str in aliases_lower:
                    new_columns[col] = canonical
                    mapped = True
                    break
            
            if not mapped:
                new_columns[col] = str(col)
        
        df = df.rename(columns=new_columns)
        
        # 必須カラムの存在確認
        required = ["term_source", "term_target"]
        for col in required:
            if col not in df.columns:
                raise GlossaryManagerError(
                    f"必須カラム '{col}' が見つかりません。"
                    f"利用可能なカラム: {list(df.columns)}"
                )
        
        return df
    
    def _parse_entries(self, df: pd.DataFrame) -> List[GlossaryEntry]:
        """
        DataFrameからGlossaryEntryのリストを生成
        
        Args:
            df: DataFrame
            
        Returns:
            GlossaryEntryのリスト
        """
        entries = []
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            term_source = str(row.get("term_source", "")).strip()
            term_target = str(row.get("term_target", "")).strip()
            
            # 空の行はスキップ
            if not term_source or not term_target:
                continue
            
            # NaN値の処理
            if term_source.lower() == "nan" or term_target.lower() == "nan":
                continue
            
            category = str(row.get("category", "")).strip() if pd.notna(row.get("category")) else ""
            if category.lower() == "nan":
                category = ""
            
            context_note = str(row.get("context_note", "")).strip() if pd.notna(row.get("context_note")) else None
            if context_note and context_note.lower() == "nan":
                context_note = None
            
            # do_not_translateの解析
            dnt_value = row.get("do_not_translate", "")
            do_not_translate = self._parse_bool(dnt_value)
            
            entry = GlossaryEntry(
                term_source=term_source,
                term_target=term_target,
                category=category,
                context_note=context_note,
                do_not_translate=do_not_translate,
            )
            entries.append(entry)
        
        return entries
    
    def _parse_bool(self, value) -> bool:
        """
        様々な形式のブール値を解析
        
        Args:
            value: 解析する値
            
        Returns:
            ブール値
        """
        if pd.isna(value):
            return False
        
        str_value = str(value).strip().lower()
        
        if str_value in ["true", "yes", "1", "○", "◯", "はい", "y"]:
            return True
        
        return False
    
    def filter_by_text(
        self,
        text: str,
        entries: Optional[List[GlossaryEntry]] = None
    ) -> List[GlossaryEntry]:
        """
        テキスト内にマッチする用語のみを抽出
        
        Args:
            text: 検索対象テキスト
            entries: 用語集エントリ（省略時は読み込み済みエントリを使用）
            
        Returns:
            マッチした用語のリスト
        """
        if entries is None:
            entries = self._entries
        
        if not text or not entries:
            return []
        
        matched = []
        
        for entry in entries:
            # 原文用語がテキスト内に含まれているかチェック
            if entry.term_source in text:
                matched.append(entry)
        
        logger.debug(f"用語フィルタリング: {len(matched)}/{len(entries)} 件マッチ")
        
        return matched
    
    def verify_usage(
        self,
        source: str,
        translated: str,
        entries: Optional[List[GlossaryEntry]] = None
    ) -> List[GlossaryVerificationResult]:
        """
        翻訳結果で用語が正しく使用されているか検証
        
        Args:
            source: 原文
            translated: 翻訳結果
            entries: 用語集エントリ（省略時は読み込み済みエントリを使用）
            
        Returns:
            検証結果のリスト（不一致のみ）
        """
        if entries is None:
            entries = self._entries
        
        if not source or not translated or not entries:
            return []
        
        results = []
        
        for entry in entries:
            # 原文に用語が含まれているかチェック
            found_in_source = entry.term_source in source
            
            if not found_in_source:
                # 原文に用語が含まれていない場合はスキップ
                continue
            
            # 翻訳結果に訳語が含まれているかチェック
            found_in_target = entry.term_target in translated
            
            # do_not_translateの場合は原文がそのまま残っているかチェック
            if entry.do_not_translate:
                found_in_target = entry.term_source in translated
            
            is_consistent = found_in_target
            
            if not is_consistent:
                if entry.do_not_translate:
                    message = f"翻訳禁止用語 '{entry.term_source}' が翻訳結果に含まれていません"
                else:
                    message = f"用語 '{entry.term_source}' の訳語 '{entry.term_target}' が翻訳結果に含まれていません"
                
                result = GlossaryVerificationResult(
                    term_source=entry.term_source,
                    term_target=entry.term_target,
                    found_in_source=found_in_source,
                    found_in_target=found_in_target,
                    is_consistent=is_consistent,
                    message=message,
                )
                results.append(result)
        
        if results:
            logger.warning(f"用語一貫性検証: {len(results)} 件の不一致を検出")
        
        return results
    
    def get_entries(self) -> List[GlossaryEntry]:
        """
        読み込み済みの用語集エントリを取得
        
        Returns:
            GlossaryEntryのリスト
        """
        return self._entries.copy()
    
    def add_entry(self, entry: GlossaryEntry) -> None:
        """
        用語集エントリを追加
        
        Args:
            entry: 追加するエントリ
        """
        self._entries.append(entry)
    
    def clear(self) -> None:
        """用語集をクリア"""
        self._entries = []
    
    def format_for_prompt(
        self,
        entries: List[GlossaryEntry],
        include_context: bool = True
    ) -> str:
        """
        プロンプト用に用語集をフォーマット
        
        Args:
            entries: フォーマットする用語集エントリ
            include_context: 文脈メモを含めるか
            
        Returns:
            フォーマットされた文字列
        """
        if not entries:
            return ""
        
        lines = ["## 用語集（必ず以下の訳語を使用してください）", ""]
        
        for entry in entries:
            if entry.do_not_translate:
                line = f"- {entry.term_source} → {entry.term_source}（翻訳禁止）"
            else:
                line = f"- {entry.term_source} → {entry.term_target}"
            
            if include_context and entry.context_note:
                line += f"（{entry.context_note}）"
            
            lines.append(line)
        
        return "\n".join(lines)
