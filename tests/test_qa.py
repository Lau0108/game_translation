"""QAチェックモジュールのテスト"""

import pytest
from typing import Dict, List, Optional

from excel_translator.qa import (
    QAChecker,
    GlossaryCheckResult,
    CharCountResult,
    PlaceholderValidationResult,
    QACheckSummary,
)
from excel_translator.glossary import GlossaryEntry, GlossaryManager


# ============================================================================
# ユニットテスト
# ============================================================================

class TestQACheckerGlossaryConsistency:
    """用語一貫性チェックのテスト"""
    
    def test_consistent_translation(self):
        """一貫した翻訳では問題なし"""
        checker = QAChecker()
        entries = [
            GlossaryEntry("勇者", "Hero", "キャラ名"),
        ]
        
        result = checker.check_glossary_consistency(
            text_id="test_001",
            source_text="勇者は旅に出た。",
            translated_text="The Hero set out on a journey.",
            glossary_entries=entries
        )
        
        assert result.is_consistent == True
        assert result.status == "✅"
        assert result.detail == ""
    
    def test_inconsistent_translation(self):
        """不一致な翻訳を検出"""
        checker = QAChecker()
        entries = [
            GlossaryEntry("勇者", "Hero", "キャラ名"),
        ]
        
        result = checker.check_glossary_consistency(
            text_id="test_001",
            source_text="勇者は旅に出た。",
            translated_text="The Warrior set out on a journey.",  # Heroではない
            glossary_entries=entries
        )
        
        assert result.is_consistent == False
        assert result.status == "⚠️"
        assert "勇者" in result.detail
        assert "Hero" in result.detail
    
    def test_empty_text(self):
        """空テキストの場合"""
        checker = QAChecker()
        
        result = checker.check_glossary_consistency(
            text_id="test_001",
            source_text="",
            translated_text="",
            glossary_entries=[]
        )
        
        assert result.is_consistent == True
        assert result.status == "✅"
    
    def test_no_glossary_entries(self):
        """用語集エントリがない場合"""
        checker = QAChecker()
        
        result = checker.check_glossary_consistency(
            text_id="test_001",
            source_text="勇者は旅に出た。",
            translated_text="The Hero set out on a journey.",
            glossary_entries=[]
        )
        
        assert result.is_consistent == True
        assert result.status == "✅"
    
    def test_batch_check(self):
        """バッチチェック"""
        checker = QAChecker()
        entries = [
            GlossaryEntry("勇者", "Hero", "キャラ名"),
        ]
        
        rows = [
            {"text_id": "001", "source_text": "勇者は旅に出た。", "translated_text": "The Hero set out."},
            {"text_id": "002", "source_text": "勇者は戦った。", "translated_text": "The Warrior fought."},
        ]
        
        results = checker.check_glossary_consistency_batch(rows, entries)
        
        assert len(results) == 2
        assert results[0].is_consistent == True
        assert results[1].is_consistent == False


class TestQACheckerCharCount:
    """文字数チェックのテスト"""
    
    def test_within_limit(self):
        """上限内の場合"""
        checker = QAChecker()
        
        result = checker.check_char_count(
            text_id="test_001",
            source_text="こんにちは",
            translated_text="Hello",
            char_limit=10
        )
        
        assert result.is_within_limit == True
        assert result.status == "✅"
        assert result.translated_count == 5
    
    def test_exceeds_limit(self):
        """上限超過の場合"""
        checker = QAChecker()
        
        result = checker.check_char_count(
            text_id="test_001",
            source_text="こんにちは",
            translated_text="Hello, how are you today?",
            char_limit=10
        )
        
        assert result.is_within_limit == False
        assert result.status == "⚠️"
        assert "超過" in result.detail
    
    def test_no_limit(self):
        """上限なしの場合"""
        checker = QAChecker()
        
        result = checker.check_char_count(
            text_id="test_001",
            source_text="こんにちは",
            translated_text="Hello, how are you today? This is a very long sentence.",
            char_limit=None
        )
        
        assert result.is_within_limit == True
        assert result.status == "✅"
    
    def test_exact_limit(self):
        """ちょうど上限の場合"""
        checker = QAChecker()
        
        result = checker.check_char_count(
            text_id="test_001",
            source_text="こんにちは",
            translated_text="Hello",
            char_limit=5
        )
        
        assert result.is_within_limit == True
        assert result.status == "✅"
    
    def test_batch_check(self):
        """バッチチェック"""
        checker = QAChecker()
        
        rows = [
            {"text_id": "001", "source_text": "短い", "translated_text": "Short"},
            {"text_id": "002", "source_text": "長い", "translated_text": "This is a very long text"},
        ]
        
        results = checker.check_char_count_batch(rows, char_limit=10)
        
        assert len(results) == 2
        assert results[0].is_within_limit == True
        assert results[1].is_within_limit == False


class TestQACheckerPlaceholderValidation:
    """プレースホルダー検証のテスト"""
    
    def test_valid_placeholders(self):
        """プレースホルダーが一致する場合"""
        checker = QAChecker()
        
        result = checker.validate_placeholders(
            text_id="test_001",
            source_text="こんにちは、{player_name}さん！",
            translated_text="Hello, {player_name}!"
        )
        
        assert result.is_valid == True
        assert result.status == "✅"
        assert len(result.missing_placeholders) == 0
        assert len(result.extra_placeholders) == 0
    
    def test_missing_placeholder(self):
        """プレースホルダーが欠落している場合"""
        checker = QAChecker()
        
        result = checker.validate_placeholders(
            text_id="test_001",
            source_text="こんにちは、{player_name}さん！",
            translated_text="Hello, friend!"
        )
        
        assert result.is_valid == False
        assert result.status == "⚠️"
        assert "{player_name}" in result.missing_placeholders
        assert "欠落" in result.detail
    
    def test_extra_placeholder(self):
        """余分なプレースホルダーがある場合"""
        checker = QAChecker()
        
        result = checker.validate_placeholders(
            text_id="test_001",
            source_text="こんにちは！",
            translated_text="Hello, {player_name}!"
        )
        
        assert result.is_valid == False
        assert result.status == "⚠️"
        assert "{player_name}" in result.extra_placeholders
        assert "余分" in result.detail
    
    def test_multiple_placeholders(self):
        """複数のプレースホルダー"""
        checker = QAChecker()
        
        result = checker.validate_placeholders(
            text_id="test_001",
            source_text="{player_name}は{item_name}を手に入れた！",
            translated_text="{player_name} obtained {item_name}!"
        )
        
        assert result.is_valid == True
        assert result.status == "✅"
        assert len(result.source_placeholders) == 2
        assert len(result.translated_placeholders) == 2
    
    def test_protected_placeholders(self):
        """保護されたプレースホルダー（<<VAR_N>>形式）"""
        checker = QAChecker()
        
        result = checker.validate_placeholders(
            text_id="test_001",
            source_text="こんにちは、<<VAR_0>>さん！<<TAG_0>>",
            translated_text="Hello, <<VAR_0>>!<<TAG_0>>"
        )
        
        assert result.is_valid == True
        assert result.status == "✅"
    
    def test_no_placeholders(self):
        """プレースホルダーがない場合"""
        checker = QAChecker()
        
        result = checker.validate_placeholders(
            text_id="test_001",
            source_text="こんにちは！",
            translated_text="Hello!"
        )
        
        assert result.is_valid == True
        assert result.status == "✅"
    
    def test_batch_validation(self):
        """バッチ検証"""
        checker = QAChecker()
        
        rows = [
            {"text_id": "001", "source_text": "{name}さん", "translated_text": "{name}"},
            {"text_id": "002", "source_text": "{name}さん", "translated_text": "friend"},
        ]
        
        results = checker.validate_placeholders_batch(rows)
        
        assert len(results) == 2
        assert results[0].is_valid == True
        assert results[1].is_valid == False


class TestQACheckerRunAllChecks:
    """全チェック実行のテスト"""
    
    def test_run_all_checks(self):
        """全チェックの実行"""
        checker = QAChecker()
        entries = [
            GlossaryEntry("勇者", "Hero", "キャラ名"),
        ]
        
        rows = [
            {
                "text_id": "001",
                "source_text": "勇者は{item}を手に入れた。",
                "translated_text": "The Hero obtained {item}."
            },
            {
                "text_id": "002",
                "source_text": "勇者は戦った。",
                "translated_text": "The Warrior fought."  # 用語不一致
            },
        ]
        
        glossary_results, char_results, placeholder_results, summary = checker.run_all_checks(
            rows, entries, char_limit=50
        )
        
        assert len(glossary_results) == 2
        assert len(char_results) == 2
        assert len(placeholder_results) == 2
        assert summary.total_rows == 2
        assert summary.glossary_issues == 1
    
    def test_summary_calculation(self):
        """サマリ計算"""
        checker = QAChecker()
        
        rows = [
            {"text_id": "001", "source_text": "テスト", "translated_text": "Test"},
            {"text_id": "002", "source_text": "テスト", "translated_text": "Test"},
        ]
        
        _, _, _, summary = checker.run_all_checks(rows, [], char_limit=100)
        
        assert summary.total_rows == 2
        assert summary.glossary_issues == 0
        assert summary.char_count_issues == 0
        assert summary.placeholder_issues == 0
        assert summary.overall_pass_rate == 1.0


class TestQACheckerFormatResults:
    """結果フォーマットのテスト"""
    
    def test_format_results_for_excel(self):
        """Excel出力用フォーマット"""
        checker = QAChecker()
        
        rows = [
            {"text_id": "001", "source_text": "テスト", "translated_text": "Test"},
        ]
        
        glossary_results, char_results, placeholder_results, _ = checker.run_all_checks(
            rows, [], char_limit=100
        )
        
        formatted = checker.format_results_for_excel(
            glossary_results, char_results, placeholder_results
        )
        
        assert len(formatted) == 1
        assert formatted[0]["text_id"] == "001"
        assert "glossary_check" in formatted[0]
        assert "char_count" in formatted[0]
        assert "length_ok" in formatted[0]
        assert "placeholder_check" in formatted[0]
