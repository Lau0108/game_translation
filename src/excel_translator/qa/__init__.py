"""QAチェックモジュール

翻訳結果の品質チェック（用語一貫性、文字数、プレースホルダー検証）を行う。
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from excel_translator.glossary import GlossaryEntry, GlossaryManager, GlossaryVerificationResult

logger = logging.getLogger(__name__)


@dataclass
class GlossaryCheckResult:
    """用語一貫性チェック結果"""
    text_id: str
    source_text: str
    translated_text: str
    issues: List[GlossaryVerificationResult]
    is_consistent: bool
    status: str  # "✅" or "⚠️"
    detail: str  # 詳細メッセージ


@dataclass
class CharCountResult:
    """文字数チェック結果"""
    text_id: str
    source_text: str
    translated_text: str
    source_count: int
    translated_count: int
    limit: Optional[int]
    is_within_limit: bool
    status: str  # "✅" or "⚠️"
    detail: str


@dataclass
class PlaceholderValidationResult:
    """プレースホルダー検証結果"""
    text_id: str
    source_text: str
    translated_text: str
    source_placeholders: List[str]
    translated_placeholders: List[str]
    missing_placeholders: List[str]
    extra_placeholders: List[str]
    is_valid: bool
    status: str  # "✅" or "⚠️"
    detail: str


@dataclass
class QACheckSummary:
    """QAチェックサマリ"""
    total_rows: int
    glossary_issues: int
    char_count_issues: int
    placeholder_issues: int
    overall_pass_rate: float


class QAChecker:
    """QAチェッククラス
    
    翻訳結果の品質チェックを行う:
    - 用語一貫性チェック
    - 文字数チェック
    - プレースホルダー検証
    """
    
    # プレースホルダーパターン
    PLACEHOLDER_PATTERNS = [
        re.compile(r'<<VAR_\d+>>'),      # 変数プレースホルダー
        re.compile(r'<<TAG_\d+>>'),      # タグプレースホルダー
        re.compile(r'<<MARKER_\d+>>'),   # マーカープレースホルダー
        re.compile(r'\{[^}]+\}'),        # 元の変数形式 {variable}
    ]
    
    def __init__(self, glossary_manager: Optional[GlossaryManager] = None):
        """
        Args:
            glossary_manager: 用語集マネージャー（省略時は新規作成）
        """
        self._glossary_manager = glossary_manager or GlossaryManager()
    
    def check_glossary_consistency(
        self,
        text_id: str,
        source_text: str,
        translated_text: str,
        glossary_entries: Optional[List[GlossaryEntry]] = None
    ) -> GlossaryCheckResult:
        """
        用語一貫性チェック
        
        Args:
            text_id: テキストID
            source_text: 原文
            translated_text: 翻訳結果
            glossary_entries: 用語集エントリ（省略時は読み込み済みエントリを使用）
            
        Returns:
            GlossaryCheckResult
        """
        if not source_text or not translated_text:
            return GlossaryCheckResult(
                text_id=text_id,
                source_text=source_text or "",
                translated_text=translated_text or "",
                issues=[],
                is_consistent=True,
                status="✅",
                detail=""
            )
        
        # 用語一貫性検証
        issues = self._glossary_manager.verify_usage(
            source_text,
            translated_text,
            glossary_entries
        )
        
        is_consistent = len(issues) == 0
        
        if is_consistent:
            status = "✅"
            detail = ""
        else:
            status = "⚠️"
            # 不一致の詳細を生成
            details = []
            for issue in issues:
                details.append(f"'{issue.term_source}' → '{issue.term_target}' が未使用")
            detail = "; ".join(details)
        
        return GlossaryCheckResult(
            text_id=text_id,
            source_text=source_text,
            translated_text=translated_text,
            issues=issues,
            is_consistent=is_consistent,
            status=status,
            detail=detail
        )
    
    def check_glossary_consistency_batch(
        self,
        rows: List[Dict],
        glossary_entries: Optional[List[GlossaryEntry]] = None
    ) -> List[GlossaryCheckResult]:
        """
        複数行の用語一貫性チェック
        
        Args:
            rows: 行データのリスト（text_id, source_text, translated_text を含む）
            glossary_entries: 用語集エントリ
            
        Returns:
            GlossaryCheckResultのリスト
        """
        results = []
        for row in rows:
            result = self.check_glossary_consistency(
                text_id=row.get("text_id", ""),
                source_text=row.get("source_text", ""),
                translated_text=row.get("translated_text", ""),
                glossary_entries=glossary_entries
            )
            results.append(result)
        return results
    
    def check_char_count(
        self,
        text_id: str,
        source_text: str,
        translated_text: str,
        char_limit: Optional[int] = None
    ) -> CharCountResult:
        """
        文字数チェック
        
        Args:
            text_id: テキストID
            source_text: 原文
            translated_text: 翻訳結果
            char_limit: 文字数上限（省略時は上限なし）
            
        Returns:
            CharCountResult
        """
        source_count = len(source_text) if source_text else 0
        translated_count = len(translated_text) if translated_text else 0
        
        if char_limit is None:
            # 上限なしの場合は常にOK
            is_within_limit = True
            status = "✅"
            detail = f"文字数: {translated_count}"
        else:
            is_within_limit = translated_count <= char_limit
            if is_within_limit:
                status = "✅"
                detail = f"文字数: {translated_count}/{char_limit}"
            else:
                status = "⚠️"
                over_count = translated_count - char_limit
                detail = f"文字数超過: {translated_count}/{char_limit} (+{over_count})"
        
        return CharCountResult(
            text_id=text_id,
            source_text=source_text or "",
            translated_text=translated_text or "",
            source_count=source_count,
            translated_count=translated_count,
            limit=char_limit,
            is_within_limit=is_within_limit,
            status=status,
            detail=detail
        )
    
    def check_char_count_batch(
        self,
        rows: List[Dict],
        char_limit: Optional[int] = None
    ) -> List[CharCountResult]:
        """
        複数行の文字数チェック
        
        Args:
            rows: 行データのリスト
            char_limit: 文字数上限
            
        Returns:
            CharCountResultのリスト
        """
        results = []
        for row in rows:
            result = self.check_char_count(
                text_id=row.get("text_id", ""),
                source_text=row.get("source_text", ""),
                translated_text=row.get("translated_text", ""),
                char_limit=char_limit
            )
            results.append(result)
        return results
    
    def _extract_placeholders(self, text: str) -> List[str]:
        """
        テキストからプレースホルダーを抽出
        
        Args:
            text: テキスト
            
        Returns:
            プレースホルダーのリスト
        """
        if not text:
            return []
        
        placeholders = []
        for pattern in self.PLACEHOLDER_PATTERNS:
            matches = pattern.findall(text)
            placeholders.extend(matches)
        
        return placeholders
    
    def validate_placeholders(
        self,
        text_id: str,
        source_text: str,
        translated_text: str
    ) -> PlaceholderValidationResult:
        """
        プレースホルダー検証
        
        原文と翻訳結果のプレースホルダーが一致するか検証する。
        
        Args:
            text_id: テキストID
            source_text: 原文
            translated_text: 翻訳結果
            
        Returns:
            PlaceholderValidationResult
        """
        source_placeholders = self._extract_placeholders(source_text)
        translated_placeholders = self._extract_placeholders(translated_text)
        
        # 欠落しているプレースホルダー（原文にあるが翻訳にない）
        source_set = set(source_placeholders)
        translated_set = set(translated_placeholders)
        
        missing = list(source_set - translated_set)
        extra = list(translated_set - source_set)
        
        is_valid = len(missing) == 0 and len(extra) == 0
        
        if is_valid:
            status = "✅"
            detail = ""
        else:
            status = "⚠️"
            details = []
            if missing:
                details.append(f"欠落: {', '.join(missing)}")
            if extra:
                details.append(f"余分: {', '.join(extra)}")
            detail = "; ".join(details)
        
        return PlaceholderValidationResult(
            text_id=text_id,
            source_text=source_text or "",
            translated_text=translated_text or "",
            source_placeholders=source_placeholders,
            translated_placeholders=translated_placeholders,
            missing_placeholders=missing,
            extra_placeholders=extra,
            is_valid=is_valid,
            status=status,
            detail=detail
        )
    
    def validate_placeholders_batch(
        self,
        rows: List[Dict]
    ) -> List[PlaceholderValidationResult]:
        """
        複数行のプレースホルダー検証
        
        Args:
            rows: 行データのリスト
            
        Returns:
            PlaceholderValidationResultのリスト
        """
        results = []
        for row in rows:
            result = self.validate_placeholders(
                text_id=row.get("text_id", ""),
                source_text=row.get("source_text", ""),
                translated_text=row.get("translated_text", "")
            )
            results.append(result)
        return results
    
    def run_all_checks(
        self,
        rows: List[Dict],
        glossary_entries: Optional[List[GlossaryEntry]] = None,
        char_limit: Optional[int] = None
    ) -> Tuple[List[GlossaryCheckResult], List[CharCountResult], List[PlaceholderValidationResult], QACheckSummary]:
        """
        全てのQAチェックを実行
        
        Args:
            rows: 行データのリスト
            glossary_entries: 用語集エントリ
            char_limit: 文字数上限
            
        Returns:
            (用語チェック結果, 文字数チェック結果, プレースホルダー検証結果, サマリ)
        """
        glossary_results = self.check_glossary_consistency_batch(rows, glossary_entries)
        char_count_results = self.check_char_count_batch(rows, char_limit)
        placeholder_results = self.validate_placeholders_batch(rows)
        
        # サマリを計算
        total_rows = len(rows)
        glossary_issues = sum(1 for r in glossary_results if not r.is_consistent)
        char_count_issues = sum(1 for r in char_count_results if not r.is_within_limit)
        placeholder_issues = sum(1 for r in placeholder_results if not r.is_valid)
        
        total_issues = glossary_issues + char_count_issues + placeholder_issues
        total_checks = total_rows * 3  # 3種類のチェック
        
        if total_checks > 0:
            overall_pass_rate = (total_checks - total_issues) / total_checks
        else:
            overall_pass_rate = 1.0
        
        summary = QACheckSummary(
            total_rows=total_rows,
            glossary_issues=glossary_issues,
            char_count_issues=char_count_issues,
            placeholder_issues=placeholder_issues,
            overall_pass_rate=overall_pass_rate
        )
        
        logger.info(
            f"QAチェック完了: {total_rows}行, "
            f"用語不一致: {glossary_issues}, "
            f"文字数超過: {char_count_issues}, "
            f"プレースホルダー問題: {placeholder_issues}"
        )
        
        return glossary_results, char_count_results, placeholder_results, summary
    
    def format_results_for_excel(
        self,
        glossary_results: List[GlossaryCheckResult],
        char_count_results: List[CharCountResult],
        placeholder_results: List[PlaceholderValidationResult]
    ) -> List[Dict]:
        """
        Excel出力用に結果をフォーマット
        
        Args:
            glossary_results: 用語チェック結果
            char_count_results: 文字数チェック結果
            placeholder_results: プレースホルダー検証結果
            
        Returns:
            Excel出力用の辞書リスト
        """
        results = []
        
        for i in range(len(glossary_results)):
            glossary = glossary_results[i]
            char_count = char_count_results[i] if i < len(char_count_results) else None
            placeholder = placeholder_results[i] if i < len(placeholder_results) else None
            
            row = {
                "text_id": glossary.text_id,
                "glossary_check": glossary.status,
                "glossary_detail": glossary.detail,
                "char_count": char_count.translated_count if char_count else 0,
                "length_ok": char_count.status if char_count else "✅",
                "length_detail": char_count.detail if char_count else "",
                "placeholder_check": placeholder.status if placeholder else "✅",
                "placeholder_detail": placeholder.detail if placeholder else "",
            }
            results.append(row)
        
        return results
