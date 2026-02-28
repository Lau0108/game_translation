"""レビューパイプラインモジュールのテスト

ReviewPipelineのユニットテストとプロパティテスト
"""

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings, strategies as st, assume

from excel_translator.review import (
    ReviewPipeline,
    PassResult,
    RuleCheckResult,
    BacktranslationResult,
    ReviewPipelineError,
)


# ============================================================================
# テストユーティリティ
# ============================================================================

class MockTranslationResponse:
    """モック翻訳レスポンス"""
    def __init__(
        self,
        translations: List[Dict[str, Any]],
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        response_time_ms: int = 0,
    ):
        self.translations = translations
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_tokens = cache_read_tokens
        self.cache_write_tokens = cache_write_tokens
        self.response_time_ms = response_time_ms


class MockTranslationProvider:
    """テスト用モックプロバイダー"""
    
    def __init__(self, responses: Optional[List[MockTranslationResponse]] = None):
        self._responses = responses or []
        self._call_count = 0
        self._calls: List[Dict[str, str]] = []
    
    async def translate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> MockTranslationResponse:
        self._calls.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        })
        
        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
            self._call_count += 1
            return response
        
        self._call_count += 1
        return MockTranslationResponse(translations=[])
    
    @property
    def call_count(self) -> int:
        return self._call_count
    
    @property
    def calls(self) -> List[Dict[str, str]]:
        return self._calls


class MockPromptBuilder:
    """テスト用モックPromptBuilder"""

    def __init__(self):
        self.build_system_prompt_calls = 0
        self.build_review_prompt_calls = 0
        self.build_consistency_prompt_calls = 0
        self.build_backtrans_prompt_calls = 0
        self.build_backtrans_step1_prompt_calls = 0
        self.build_backtrans_step2_prompt_calls = 0

    def build_system_prompt(self, target_lang: Optional[str] = None, pass_type: Optional[str] = None) -> str:
        self.build_system_prompt_calls += 1
        return "System prompt"

    def build_review_prompt(
        self,
        source_texts: List[Dict],
        translations: List[Dict],
        character_profiles: Optional[List] = None,
    ) -> str:
        self.build_review_prompt_calls += 1
        return "Review prompt"

    def build_consistency_prompt(
        self,
        all_translations: List[Dict],
    ) -> str:
        self.build_consistency_prompt_calls += 1
        return "Consistency prompt"

    def build_backtrans_prompt(
        self,
        translations: List[Dict],
        source_lang: Optional[str] = None,
    ) -> str:
        self.build_backtrans_prompt_calls += 1
        return "Backtrans prompt"

    def build_backtrans_step1_prompt(
        self,
        translations: List[Dict],
        source_lang: Optional[str] = None,
    ) -> str:
        self.build_backtrans_step1_prompt_calls += 1
        return "Backtrans step1 prompt"

    def build_backtrans_step2_prompt(
        self,
        comparisons: List[Dict],
        source_lang: Optional[str] = None,
    ) -> str:
        self.build_backtrans_step2_prompt_calls += 1
        return "Backtrans step2 prompt"


def create_source_text(
    text_id: str,
    source_text: str,
    character: Optional[str] = None,
) -> Dict[str, Any]:
    """テスト用原文テキストを作成"""
    return {
        "text_id": text_id,
        "character": character,
        "source_text": source_text,
    }


def create_translation(
    text_id: str,
    source_text: str,
    translated_text: str,
    character: Optional[str] = None,
) -> Dict[str, Any]:
    """テスト用翻訳結果を作成"""
    return {
        "text_id": text_id,
        "source_text": source_text,
        "translated_text": translated_text,
        "character": character,
    }


# ============================================================================
# ユニットテスト
# ============================================================================

class TestReviewPipelineBasic:
    """ReviewPipeline基本機能のテスト"""
    
    def test_init_without_provider(self):
        """プロバイダーなしで初期化"""
        pipeline = ReviewPipeline()
        assert pipeline.translation_provider is None
        assert pipeline.prompt_builder is None
    
    def test_init_with_provider(self):
        """プロバイダーありで初期化"""
        provider = MockTranslationProvider()
        builder = MockPromptBuilder()
        
        pipeline = ReviewPipeline(
            translation_provider=provider,
            prompt_builder=builder,
        )
        
        assert pipeline.translation_provider == provider
        assert pipeline.prompt_builder == builder
    
    def test_get_statistics(self):
        """統計情報の取得"""
        pipeline = ReviewPipeline()
        stats = pipeline.get_statistics()
        
        assert "2nd_pass_reviewed" in stats
        assert "2nd_pass_modified" in stats
        assert "3rd_pass_rules_checked" in stats
        assert "4th_pass_checked" in stats
    
    def test_reset_statistics(self):
        """統計情報のリセット"""
        pipeline = ReviewPipeline()
        pipeline._stats["2nd_pass_reviewed"] = 10
        
        pipeline.reset_statistics()
        
        assert pipeline._stats["2nd_pass_reviewed"] == 0


class TestRun2ndPass:
    """run_2nd_passのテスト"""
    
    @pytest.mark.asyncio
    async def test_2nd_pass_without_provider_raises_error(self):
        """プロバイダーなしで2nd passはエラー"""
        pipeline = ReviewPipeline()
        
        with pytest.raises(ReviewPipelineError):
            await pipeline.run_2nd_pass([], [])
    
    @pytest.mark.asyncio
    async def test_2nd_pass_returns_modified_only(self):
        """2nd passは修正が必要な行のみ返す"""
        # 修正が必要な行のみを返すレスポンス
        response = MockTranslationResponse(translations=[
            {
                "text_id": "001",
                "revised_translation": "Revised Hello",
                "reason": "より自然な表現に修正",
            }
        ])
        
        provider = MockTranslationProvider(responses=[response])
        builder = MockPromptBuilder()
        
        pipeline = ReviewPipeline(
            translation_provider=provider,
            prompt_builder=builder,
        )
        
        source_texts = [
            create_source_text("001", "こんにちは"),
            create_source_text("002", "さようなら"),
        ]
        
        pass1_results = [
            {"text_id": "001", "translated_text": "Hello"},
            {"text_id": "002", "translated_text": "Goodbye"},
        ]
        
        results, stats = await pipeline.run_2nd_pass(source_texts, pass1_results)

        # 修正が必要な行のみ返される
        assert len(results) == 1
        assert results[0].text_id == "001"
        assert results[0].result == "Revised Hello"
        assert results[0].reason == "より自然な表現に修正"
        assert results[0].changed is True
    
    @pytest.mark.asyncio
    async def test_2nd_pass_empty_when_no_modifications(self):
        """修正不要の場合は空リスト"""
        response = MockTranslationResponse(translations=[])
        
        provider = MockTranslationProvider(responses=[response])
        builder = MockPromptBuilder()
        
        pipeline = ReviewPipeline(
            translation_provider=provider,
            prompt_builder=builder,
        )
        
        results, stats = await pipeline.run_2nd_pass(
            [create_source_text("001", "テスト")],
            [{"text_id": "001", "translated_text": "Test"}],
        )

        assert len(results) == 0


class TestRun3rdPassRules:
    """run_3rd_pass_rulesのテスト"""
    
    def test_rules_check_detects_bracket_mixing(self):
        """括弧の混在を検出"""
        pipeline = ReviewPipeline(target_lang="ja")
        
        translations = [
            create_translation("001", "テスト", "「Hello」(world)"),
        ]
        
        issues = pipeline.run_3rd_pass_rules(translations)
        
        bracket_issues = [i for i in issues if i.issue_type == "bracket_mixed"]
        assert len(bracket_issues) > 0
    
    def test_rules_check_detects_number_mixing(self):
        """数字表記の混在を検出"""
        pipeline = ReviewPipeline()
        
        translations = [
            create_translation("001", "テスト", "123と１２３"),
        ]
        
        issues = pipeline.run_3rd_pass_rules(translations)
        
        number_issues = [i for i in issues if i.issue_type == "number_mixed"]
        assert len(number_issues) > 0
    
    def test_rules_check_no_issues_for_consistent_text(self):
        """一貫したテキストでは問題なし"""
        pipeline = ReviewPipeline(target_lang="en")
        
        translations = [
            create_translation("001", "テスト", "Hello world"),
            create_translation("002", "テスト2", "Goodbye world"),
        ]
        
        issues = pipeline.run_3rd_pass_rules(translations)
        
        # 英語翻訳で一貫している場合は問題なし
        assert len(issues) == 0


class TestRun3rdPassAI:
    """run_3rd_pass_aiのテスト"""
    
    @pytest.mark.asyncio
    async def test_3rd_pass_ai_without_provider_raises_error(self):
        """プロバイダーなしで3rd pass AIはエラー"""
        pipeline = ReviewPipeline()
        
        with pytest.raises(ReviewPipelineError):
            await pipeline.run_3rd_pass_ai([])
    
    @pytest.mark.asyncio
    async def test_3rd_pass_ai_returns_fixes(self):
        """3rd pass AIは修正を返す"""
        response = MockTranslationResponse(translations=[
            {
                "text_id": "001",
                "revised_translation": "Consistent Hello",
                "reason": "文体統一のため修正",
            }
        ])
        
        provider = MockTranslationProvider(responses=[response])
        builder = MockPromptBuilder()
        
        pipeline = ReviewPipeline(
            translation_provider=provider,
            prompt_builder=builder,
        )
        
        translations = [
            create_translation("001", "こんにちは", "Hello"),
        ]
        
        results, stats = await pipeline.run_3rd_pass_ai(translations)

        assert len(results) == 1
        assert results[0].text_id == "001"
        assert results[0].changed is True


class TestRun4thPass:
    """run_4th_passのテスト"""
    
    @pytest.mark.asyncio
    async def test_4th_pass_without_provider_raises_error(self):
        """プロバイダーなしで4th passはエラー"""
        pipeline = ReviewPipeline()
        
        with pytest.raises(ReviewPipelineError):
            await pipeline.run_4th_pass([])
    
    @pytest.mark.asyncio
    async def test_4th_pass_filters_by_target_ids(self):
        """4th passは対象IDでフィルタリング"""
        # Step 1: バックトランスレーション結果
        step1_response = MockTranslationResponse(translations=[
            {
                "text_id": "001",
                "backtranslation": "こんにちは",
            }
        ])
        # Step 2: 比較結果
        step2_response = MockTranslationResponse(translations=[
            {
                "text_id": "001",
                "similarity_score": 9,
                "has_discrepancy": False,
            }
        ])

        provider = MockTranslationProvider(responses=[step1_response, step2_response])
        builder = MockPromptBuilder()

        pipeline = ReviewPipeline(
            translation_provider=provider,
            prompt_builder=builder,
        )

        translations = [
            create_translation("001", "こんにちは", "Hello"),
            create_translation("002", "さようなら", "Goodbye"),
        ]

        # 001のみを対象
        results, stats = await pipeline.run_4th_pass(
            translations,
            target_text_ids=["001"],
        )

        assert len(results) == 1
        assert results[0].text_id == "001"
    
    @pytest.mark.asyncio
    async def test_4th_pass_returns_empty_for_no_targets(self):
        """対象なしの場合は空リスト"""
        provider = MockTranslationProvider()
        builder = MockPromptBuilder()

        pipeline = ReviewPipeline(
            translation_provider=provider,
            prompt_builder=builder,
        )

        results, stats = await pipeline.run_4th_pass(
            [],
            target_text_ids=[],
        )

        assert len(results) == 0


class TestGetModifiedTextIds:
    """get_modified_text_idsのテスト"""
    
    def test_returns_modified_ids_from_pass2(self):
        """2nd passの修正IDを返す"""
        pipeline = ReviewPipeline()
        
        pass2_results = [
            PassResult(text_id="001", result="Modified", changed=True),
            PassResult(text_id="002", result="Same", changed=False),
        ]
        
        modified = pipeline.get_modified_text_ids(pass2_results=pass2_results)
        
        assert "001" in modified
        assert "002" not in modified
    
    def test_returns_modified_ids_from_both_passes(self):
        """2nd/3rd pass両方の修正IDを返す"""
        pipeline = ReviewPipeline()
        
        pass2_results = [
            PassResult(text_id="001", result="Modified", changed=True),
        ]
        pass3_results = [
            PassResult(text_id="002", result="Modified", changed=True),
        ]
        
        modified = pipeline.get_modified_text_ids(
            pass2_results=pass2_results,
            pass3_results=pass3_results,
        )
        
        assert "001" in modified
        assert "002" in modified


class TestApplyPassResults:
    """apply_pass_resultsのテスト"""
    
    def test_applies_results_to_translations(self):
        """パス結果を翻訳に適用"""
        pipeline = ReviewPipeline()
        
        translations = [
            create_translation("001", "こんにちは", "Hello"),
            create_translation("002", "さようなら", "Goodbye"),
        ]
        
        pass_results = [
            PassResult(text_id="001", result="Hi", reason="カジュアルに", changed=True),
        ]
        
        updated = pipeline.apply_pass_results(
            translations,
            pass_results,
            "pass_2",
        )
        
        assert updated[0]["pass_2"] == "Hi"
        assert updated[0]["pass_2_reason"] == "カジュアルに"
        assert updated[0]["translated_text"] == "Hi"
        
        # 002は変更なし
        assert "pass_2" not in updated[1]


# ============================================================================
# Property-Based Tests
# ============================================================================

@st.composite
def source_texts_strategy(draw):
    """原文テキストリストを生成するストラテジー"""
    num_texts = draw(st.integers(min_value=1, max_value=10))
    texts = []
    
    for i in range(num_texts):
        text_id = f"TEXT_{i+1:03d}"
        source_text = draw(st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N', 'P')),
            min_size=1,
            max_size=100
        ))
        character = draw(st.one_of(
            st.none(),
            st.sampled_from(["キャラA", "キャラB", "キャラC"])
        ))
        
        texts.append({
            "text_id": text_id,
            "character": character,
            "source_text": source_text,
        })
    
    return texts


@st.composite
def translations_strategy(draw):
    """翻訳結果リストを生成するストラテジー"""
    num_texts = draw(st.integers(min_value=1, max_value=10))
    translations = []
    
    for i in range(num_texts):
        text_id = f"TEXT_{i+1:03d}"
        source_text = draw(st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N', 'P')),
            min_size=1,
            max_size=100
        ))
        translated_text = draw(st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N', 'P')),
            min_size=1,
            max_size=100
        ))
        character = draw(st.one_of(
            st.none(),
            st.sampled_from(["キャラA", "キャラB", "キャラC"])
        ))
        
        translations.append({
            "text_id": text_id,
            "source_text": source_text,
            "translated_text": translated_text,
            "character": character,
        })
    
    return translations


@st.composite
def review_response_strategy(draw, text_ids: List[str]):
    """レビューレスポンスを生成するストラテジー"""
    # 修正する行数（0〜全件）
    num_modifications = draw(st.integers(min_value=0, max_value=len(text_ids)))
    
    # 修正対象をランダムに選択
    modified_ids = draw(st.lists(
        st.sampled_from(text_ids) if text_ids else st.just(""),
        min_size=num_modifications,
        max_size=num_modifications,
        unique=True,
    ))
    
    translations = []
    for text_id in modified_ids:
        if text_id:
            translations.append({
                "text_id": text_id,
                "revised_translation": f"Revised {text_id}",
                "reason": "修正理由",
            })
    
    return MockTranslationResponse(translations=translations)


class TestProperty14SecondPassSelfReview:
    """
    Property 14: 2nd passセルフレビュー
    
    *For any* 1st pass翻訳結果について、2nd pass実行後は修正が必要な行のみが返され、
    修正理由が記録される
    
    **Feature: excel-translation-api, Property 14: 2nd passセルフレビュー**
    **Validates: Requirements 6.2, 6.3**
    """
    
    @pytest.mark.asyncio
    @given(source_texts=source_texts_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_2nd_pass_returns_only_modified_rows(self, source_texts: List[Dict]):
        """2nd passは修正が必要な行のみを返す"""
        assume(len(source_texts) > 0)
        
        text_ids = [t["text_id"] for t in source_texts]
        
        # 一部の行のみ修正するレスポンスを生成
        num_modifications = len(text_ids) // 2  # 半分を修正
        modified_ids = text_ids[:num_modifications]
        
        response_translations = [
            {
                "text_id": tid,
                "revised_translation": f"Revised {tid}",
                "reason": "修正理由",
            }
            for tid in modified_ids
        ]
        
        response = MockTranslationResponse(translations=response_translations)
        provider = MockTranslationProvider(responses=[response])
        builder = MockPromptBuilder()
        
        pipeline = ReviewPipeline(
            translation_provider=provider,
            prompt_builder=builder,
        )
        
        pass1_results = [
            {"text_id": t["text_id"], "translated_text": f"Original {t['text_id']}"}
            for t in source_texts
        ]
        
        results, stats = await pipeline.run_2nd_pass(source_texts, pass1_results)

        # 返された結果は修正された行のみ
        assert len(results) == num_modifications

        # 全ての結果にchanged=Trueが設定されている
        for result in results:
            assert result.changed is True
            assert result.text_id in modified_ids
    
    @pytest.mark.asyncio
    @given(source_texts=source_texts_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_2nd_pass_records_reasons(self, source_texts: List[Dict]):
        """2nd passは修正理由を記録する"""
        assume(len(source_texts) > 0)
        
        text_ids = [t["text_id"] for t in source_texts]
        
        # 全行を修正するレスポンス
        response_translations = [
            {
                "text_id": tid,
                "revised_translation": f"Revised {tid}",
                "reason": f"理由 for {tid}",
            }
            for tid in text_ids
        ]
        
        response = MockTranslationResponse(translations=response_translations)
        provider = MockTranslationProvider(responses=[response])
        builder = MockPromptBuilder()
        
        pipeline = ReviewPipeline(
            translation_provider=provider,
            prompt_builder=builder,
        )
        
        pass1_results = [
            {"text_id": t["text_id"], "translated_text": f"Original {t['text_id']}"}
            for t in source_texts
        ]
        
        results, stats = await pipeline.run_2nd_pass(source_texts, pass1_results)

        # 全ての結果に理由が記録されている
        for result in results:
            assert result.reason is not None
            assert len(result.reason) > 0
    
    @pytest.mark.asyncio
    @given(source_texts=source_texts_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_2nd_pass_empty_when_no_changes_needed(self, source_texts: List[Dict]):
        """修正不要の場合は空リストを返す"""
        assume(len(source_texts) > 0)
        
        # 空のレスポンス（修正不要）
        response = MockTranslationResponse(translations=[])
        provider = MockTranslationProvider(responses=[response])
        builder = MockPromptBuilder()
        
        pipeline = ReviewPipeline(
            translation_provider=provider,
            prompt_builder=builder,
        )
        
        pass1_results = [
            {"text_id": t["text_id"], "translated_text": f"Original {t['text_id']}"}
            for t in source_texts
        ]
        
        results, stats = await pipeline.run_2nd_pass(source_texts, pass1_results)

        # 修正不要の場合は空リスト
        assert len(results) == 0



class TestProperty15ThirdPassConsistencyCheck:
    """
    Property 15: 3rd pass一貫性チェック
    
    *For any* 翻訳結果セットについて、3rd pass実行時はルールベースチェック（正規表現）が
    先に実行され、その後AIチェックが実行される
    
    **Feature: excel-translation-api, Property 15: 3rd pass一貫性チェック**
    **Validates: Requirements 6.4, 6.5**
    """
    
    @given(translations=translations_strategy())
    @settings(max_examples=100, deadline=None)
    def test_rules_check_executes_first(self, translations: List[Dict]):
        """ルールベースチェックが先に実行される"""
        assume(len(translations) > 0)
        
        pipeline = ReviewPipeline()
        
        # ルールベースチェックを実行
        rule_issues = pipeline.run_3rd_pass_rules(translations)
        
        # ルールベースチェックは常に実行可能（プロバイダー不要）
        assert isinstance(rule_issues, list)
        
        # 統計が更新されている
        stats = pipeline.get_statistics()
        assert stats["3rd_pass_rules_checked"] == len(translations)
    
    @pytest.mark.asyncio
    @given(translations=translations_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_ai_check_executes_after_rules(self, translations: List[Dict]):
        """AIチェックはルールベースチェック後に実行される"""
        assume(len(translations) > 0)
        
        response = MockTranslationResponse(translations=[])
        provider = MockTranslationProvider(responses=[response])
        builder = MockPromptBuilder()
        
        pipeline = ReviewPipeline(
            translation_provider=provider,
            prompt_builder=builder,
        )
        
        # 1. ルールベースチェックを先に実行
        rule_issues = pipeline.run_3rd_pass_rules(translations)
        
        # 2. AIチェックを実行
        ai_results, ai_stats = await pipeline.run_3rd_pass_ai(translations, rule_issues)

        # 両方が実行された
        stats = pipeline.get_statistics()
        assert stats["3rd_pass_rules_checked"] == len(translations)
        assert stats["3rd_pass_ai_checked"] == len(translations)
    
    @given(translations=translations_strategy())
    @settings(max_examples=100, deadline=None)
    def test_rules_check_detects_inconsistencies(self, translations: List[Dict]):
        """ルールベースチェックは不整合を検出する"""
        assume(len(translations) > 0)
        
        pipeline = ReviewPipeline()
        
        # 不整合を含む翻訳を追加
        inconsistent_translations = translations.copy()
        inconsistent_translations.append({
            "text_id": "INCONSISTENT_001",
            "source_text": "テスト",
            "translated_text": "「Hello」(world) 123と１２３",  # 括弧混在 + 数字混在
            "character": None,
        })
        
        rule_issues = pipeline.run_3rd_pass_rules(inconsistent_translations)
        
        # 不整合が検出される
        inconsistent_issues = [
            i for i in rule_issues 
            if i.text_id == "INCONSISTENT_001"
        ]
        assert len(inconsistent_issues) > 0
    
    @given(translations=translations_strategy())
    @settings(max_examples=100, deadline=None)
    def test_rules_check_returns_rule_check_results(self, translations: List[Dict]):
        """ルールベースチェックはRuleCheckResultを返す"""
        assume(len(translations) > 0)
        
        pipeline = ReviewPipeline()
        
        rule_issues = pipeline.run_3rd_pass_rules(translations)
        
        # 全ての結果がRuleCheckResult型
        for issue in rule_issues:
            assert isinstance(issue, RuleCheckResult)
            assert hasattr(issue, "text_id")
            assert hasattr(issue, "issue_type")
            assert hasattr(issue, "description")
    
    @pytest.mark.asyncio
    @given(translations=translations_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_ai_check_returns_pass_results(self, translations: List[Dict]):
        """AIチェックはPassResultを返す"""
        assume(len(translations) > 0)
        
        # 修正を返すレスポンス
        response = MockTranslationResponse(translations=[
            {
                "text_id": translations[0]["text_id"],
                "revised_translation": "Consistent text",
                "reason": "文体統一",
            }
        ])
        
        provider = MockTranslationProvider(responses=[response])
        builder = MockPromptBuilder()
        
        pipeline = ReviewPipeline(
            translation_provider=provider,
            prompt_builder=builder,
        )
        
        ai_results, ai_stats = await pipeline.run_3rd_pass_ai(translations)

        # 全ての結果がPassResult型
        for result in ai_results:
            assert isinstance(result, PassResult)
            assert hasattr(result, "text_id")
            assert hasattr(result, "result")
            assert hasattr(result, "reason")
            assert hasattr(result, "changed")



class TestProperty16FourthPassBacktranslation:
    """
    Property 16: 4th passバックトランスレーション
    
    *For any* 翻訳結果について、4th pass実行時は修正された行・曖昧な行のみが対象となり、
    バックトランスレーション結果が記録される
    
    **Feature: excel-translation-api, Property 16: 4th passバックトランスレーション**
    **Validates: Requirements 6.6, 6.7**
    """
    
    @pytest.mark.asyncio
    @given(translations=translations_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_4th_pass_targets_only_specified_ids(self, translations: List[Dict]):
        """4th passは指定されたIDのみを対象とする"""
        assume(len(translations) >= 2)
        
        text_ids = [t["text_id"] for t in translations]
        target_ids = text_ids[:len(text_ids) // 2]  # 半分のみを対象

        # Step 1: 対象行のバックトランスレーション結果
        step1_translations = [
            {
                "text_id": tid,
                "backtranslation": f"Back {tid}",
            }
            for tid in target_ids
        ]
        # Step 2: 比較結果
        step2_translations = [
            {
                "text_id": tid,
                "similarity_score": 9,
                "has_discrepancy": False,
            }
            for tid in target_ids
        ]

        step1_response = MockTranslationResponse(translations=step1_translations)
        step2_response = MockTranslationResponse(translations=step2_translations)
        provider = MockTranslationProvider(responses=[step1_response, step2_response])
        builder = MockPromptBuilder()

        pipeline = ReviewPipeline(
            translation_provider=provider,
            prompt_builder=builder,
        )

        results, stats_dict = await pipeline.run_4th_pass(
            translations,
            target_text_ids=target_ids,
        )

        # 対象行のみが処理される
        result_ids = {r.text_id for r in results}
        for tid in target_ids:
            if tid in result_ids:
                assert tid in target_ids

        # 統計が正しく更新される
        stats = pipeline.get_statistics()
        assert stats["4th_pass_checked"] == len(target_ids)
    
    @pytest.mark.asyncio
    @given(translations=translations_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_4th_pass_records_backtranslation(self, translations: List[Dict]):
        """4th passはバックトランスレーション結果を記録する"""
        assume(len(translations) > 0)

        text_ids = [t["text_id"] for t in translations]

        # Step 1: バックトランスレーション結果
        step1_translations = [
            {
                "text_id": tid,
                "backtranslation": f"バックトランスレーション {tid}",
            }
            for tid in text_ids
        ]
        # Step 2: 比較結果
        step2_translations = [
            {
                "text_id": tid,
                "similarity_score": 9,
                "has_discrepancy": False,
            }
            for tid in text_ids
        ]

        step1_response = MockTranslationResponse(translations=step1_translations)
        step2_response = MockTranslationResponse(translations=step2_translations)
        provider = MockTranslationProvider(responses=[step1_response, step2_response])
        builder = MockPromptBuilder()

        pipeline = ReviewPipeline(
            translation_provider=provider,
            prompt_builder=builder,
        )

        results, stats_dict = await pipeline.run_4th_pass(translations)

        # 全ての結果にバックトランスレーションが記録されている
        for result in results:
            assert isinstance(result, BacktranslationResult)
            assert result.backtranslation is not None
    
    @pytest.mark.asyncio
    @given(translations=translations_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_4th_pass_detects_discrepancies(self, translations: List[Dict]):
        """4th passは乖離を検出する"""
        assume(len(translations) > 0)

        text_ids = [t["text_id"] for t in translations]

        # Step 1: バックトランスレーション結果
        step1_translations = [
            {
                "text_id": tid,
                "backtranslation": f"Back {tid}",
            }
            for tid in text_ids
        ]

        # Step 2: 一部に乖離がある比較結果
        step2_translations = []
        for i, tid in enumerate(text_ids):
            has_discrepancy = i % 2 == 0  # 偶数番目に乖離あり
            step2_translations.append({
                "text_id": tid,
                "similarity_score": 5 if has_discrepancy else 9,
                "has_discrepancy": has_discrepancy,
                "discrepancy_description": "意味の乖離あり" if has_discrepancy else None,
                "suggested_revision": f"Revised {tid}" if has_discrepancy else None,
            })

        step1_response = MockTranslationResponse(translations=step1_translations)
        step2_response = MockTranslationResponse(translations=step2_translations)
        provider = MockTranslationProvider(responses=[step1_response, step2_response])
        builder = MockPromptBuilder()

        pipeline = ReviewPipeline(
            translation_provider=provider,
            prompt_builder=builder,
        )

        results, stats_dict = await pipeline.run_4th_pass(translations)

        # 乖離が正しく記録されている
        discrepancy_results = [r for r in results if r.has_discrepancy]
        non_discrepancy_results = [r for r in results if not r.has_discrepancy]

        for result in discrepancy_results:
            assert result.discrepancy_description is not None

        # 統計が正しく更新される
        stats = pipeline.get_statistics()
        assert stats["4th_pass_discrepancies"] == len(discrepancy_results)
    
    @pytest.mark.asyncio
    async def test_4th_pass_empty_for_empty_translations(self):
        """翻訳リストが空の場合は空リストを返す"""
        provider = MockTranslationProvider()
        builder = MockPromptBuilder()

        pipeline = ReviewPipeline(
            translation_provider=provider,
            prompt_builder=builder,
        )

        # 空の翻訳リストを指定
        results, stats_dict = await pipeline.run_4th_pass(
            translations=[],
            target_text_ids=None,
        )

        # 空リストが返される
        assert len(results) == 0

        # プロバイダーは呼ばれない
        assert provider.call_count == 0
    
    @pytest.mark.asyncio
    @given(translations=translations_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_4th_pass_processes_all_when_no_target_ids(self, translations: List[Dict]):
        """target_text_idsがNoneの場合は全件処理"""
        assume(len(translations) > 0)

        text_ids = [t["text_id"] for t in translations]

        # Step 1: バックトランスレーション結果
        step1_translations = [
            {
                "text_id": tid,
                "backtranslation": f"Back {tid}",
            }
            for tid in text_ids
        ]
        # Step 2: 比較結果
        step2_translations = [
            {
                "text_id": tid,
                "similarity_score": 9,
                "has_discrepancy": False,
            }
            for tid in text_ids
        ]

        step1_response = MockTranslationResponse(translations=step1_translations)
        step2_response = MockTranslationResponse(translations=step2_translations)
        provider = MockTranslationProvider(responses=[step1_response, step2_response])
        builder = MockPromptBuilder()

        pipeline = ReviewPipeline(
            translation_provider=provider,
            prompt_builder=builder,
        )

        # target_text_idsを指定しない（全件処理）
        results, stats_dict = await pipeline.run_4th_pass(
            translations,
            target_text_ids=None,
        )

        # 全件が処理される
        stats = pipeline.get_statistics()
        assert stats["4th_pass_checked"] == len(translations)
    
    def test_get_modified_text_ids_for_4th_pass(self):
        """4th pass対象決定用のget_modified_text_ids"""
        pipeline = ReviewPipeline()
        
        # 2nd passで修正された行
        pass2_results = [
            PassResult(text_id="001", result="Modified 001", changed=True),
            PassResult(text_id="002", result="Same 002", changed=False),
        ]
        
        # 3rd passで修正された行
        pass3_results = [
            PassResult(text_id="003", result="Modified 003", changed=True),
        ]
        
        modified_ids = pipeline.get_modified_text_ids(
            pass2_results=pass2_results,
            pass3_results=pass3_results,
        )
        
        # 修正された行のみが含まれる
        assert "001" in modified_ids
        assert "002" not in modified_ids
        assert "003" in modified_ids



class TestProperty17PassModificationReasonRecording:
    """
    Property 17: パス修正理由記録
    
    *For any* パスで修正が行われた場合、修正理由が対応するreasonフィールドに記録される
    
    **Feature: excel-translation-api, Property 17: パス修正理由記録**
    **Validates: Requirements 6.8**
    """
    
    @pytest.mark.asyncio
    @given(source_texts=source_texts_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_2nd_pass_records_reason_for_modifications(self, source_texts: List[Dict]):
        """2nd passは修正理由を記録する"""
        assume(len(source_texts) > 0)
        
        text_ids = [t["text_id"] for t in source_texts]
        
        # 修正理由を含むレスポンス
        response_translations = [
            {
                "text_id": tid,
                "revised_translation": f"Revised {tid}",
                "reason": f"修正理由: {tid}の表現を改善",
            }
            for tid in text_ids
        ]
        
        response = MockTranslationResponse(translations=response_translations)
        provider = MockTranslationProvider(responses=[response])
        builder = MockPromptBuilder()
        
        pipeline = ReviewPipeline(
            translation_provider=provider,
            prompt_builder=builder,
        )
        
        pass1_results = [
            {"text_id": t["text_id"], "translated_text": f"Original {t['text_id']}"}
            for t in source_texts
        ]
        
        results, stats = await pipeline.run_2nd_pass(source_texts, pass1_results)

        # 全ての修正に理由が記録されている
        for result in results:
            assert result.reason is not None
            assert len(result.reason) > 0
            assert result.text_id in result.reason or "修正理由" in result.reason
    
    @pytest.mark.asyncio
    @given(translations=translations_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_3rd_pass_ai_records_reason_for_modifications(self, translations: List[Dict]):
        """3rd pass AIは修正理由を記録する"""
        assume(len(translations) > 0)
        
        text_ids = [t["text_id"] for t in translations]
        
        # 修正理由を含むレスポンス
        response_translations = [
            {
                "text_id": tid,
                "revised_translation": f"Consistent {tid}",
                "reason": f"一貫性修正: {tid}の文体を統一",
            }
            for tid in text_ids
        ]
        
        response = MockTranslationResponse(translations=response_translations)
        provider = MockTranslationProvider(responses=[response])
        builder = MockPromptBuilder()
        
        pipeline = ReviewPipeline(
            translation_provider=provider,
            prompt_builder=builder,
        )
        
        results, stats = await pipeline.run_3rd_pass_ai(translations)

        # 全ての修正に理由が記録されている
        for result in results:
            assert result.reason is not None
            assert len(result.reason) > 0
    
    @given(translations=translations_strategy())
    @settings(max_examples=100, deadline=None)
    def test_3rd_pass_rules_records_description_for_issues(self, translations: List[Dict]):
        """3rd passルールベースは問題の説明を記録する"""
        assume(len(translations) > 0)
        
        pipeline = ReviewPipeline()
        
        # 問題を含む翻訳を追加
        problematic_translations = translations.copy()
        problematic_translations.append({
            "text_id": "PROBLEM_001",
            "source_text": "テスト",
            "translated_text": "「Test」(example) 123と１２３",  # 複数の問題
            "character": None,
        })
        
        issues = pipeline.run_3rd_pass_rules(problematic_translations)
        
        # 全ての問題に説明が記録されている
        for issue in issues:
            assert issue.description is not None
            assert len(issue.description) > 0
            assert issue.issue_type is not None
    
    def test_apply_pass_results_preserves_reasons(self):
        """apply_pass_resultsは理由を保持する"""
        pipeline = ReviewPipeline()
        
        translations = [
            create_translation("001", "こんにちは", "Hello"),
            create_translation("002", "さようなら", "Goodbye"),
        ]
        
        pass_results = [
            PassResult(
                text_id="001",
                result="Hi there",
                reason="よりカジュアルな表現に変更",
                changed=True,
            ),
        ]
        
        updated = pipeline.apply_pass_results(
            translations,
            pass_results,
            "pass_2",
        )
        
        # 理由が保持されている
        assert updated[0]["pass_2_reason"] == "よりカジュアルな表現に変更"
    
    @pytest.mark.asyncio
    @given(translations=translations_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_4th_pass_records_discrepancy_description(self, translations: List[Dict]):
        """4th passは乖離の説明を記録する"""
        assume(len(translations) > 0)

        text_ids = [t["text_id"] for t in translations]

        # Step 1: バックトランスレーション結果
        step1_translations = [
            {
                "text_id": tid,
                "backtranslation": f"バック {tid}",
            }
            for tid in text_ids
        ]
        # Step 2: 乖離の説明を含む比較結果
        step2_translations = [
            {
                "text_id": tid,
                "similarity_score": 5,
                "has_discrepancy": True,
                "discrepancy_description": f"意味の乖離: {tid}のニュアンスが異なる",
                "suggested_revision": f"修正案 {tid}",
            }
            for tid in text_ids
        ]

        step1_response = MockTranslationResponse(translations=step1_translations)
        step2_response = MockTranslationResponse(translations=step2_translations)
        provider = MockTranslationProvider(responses=[step1_response, step2_response])
        builder = MockPromptBuilder()

        pipeline = ReviewPipeline(
            translation_provider=provider,
            prompt_builder=builder,
        )

        results, stats_dict = await pipeline.run_4th_pass(translations)

        # 乖離がある結果には説明が記録されている
        for result in results:
            if result.has_discrepancy:
                assert result.discrepancy_description is not None
                assert len(result.discrepancy_description) > 0
    
    @pytest.mark.asyncio
    @given(source_texts=source_texts_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_empty_reason_when_no_modification(self, source_texts: List[Dict]):
        """修正なしの場合は理由も空"""
        assume(len(source_texts) > 0)
        
        # 空のレスポンス（修正なし）
        response = MockTranslationResponse(translations=[])
        provider = MockTranslationProvider(responses=[response])
        builder = MockPromptBuilder()
        
        pipeline = ReviewPipeline(
            translation_provider=provider,
            prompt_builder=builder,
        )
        
        pass1_results = [
            {"text_id": t["text_id"], "translated_text": f"Original {t['text_id']}"}
            for t in source_texts
        ]
        
        results, stats = await pipeline.run_2nd_pass(source_texts, pass1_results)

        # 修正なしの場合は結果が空
        assert len(results) == 0
