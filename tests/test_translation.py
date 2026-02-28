"""翻訳サービスモジュールのテスト

TranslationProvider, TranslationServiceのユニットテストとプロパティテスト
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import given, settings, strategies as st, assume

from excel_translator.translation import (
    TranslationProvider,
    TranslationResponse,
    TranslationService,
    TranslationError,
    APIError,
    RateLimitError,
    AuthenticationError,
    ResponseParseError,
    ClaudeProvider,
    GeminiProvider,
    GPT4Provider,
    PassResult,
    PipelineResult,
    create_provider,
)
from excel_translator.chunk import Chunk
from excel_translator.parser import InputRow
from excel_translator.config import QualityMode


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
        skip_reason=None,
    )


def create_test_chunk(
    chunk_id: str = "test_chunk",
    rows: Optional[List[InputRow]] = None,
    context_rows: Optional[List[InputRow]] = None,
) -> Chunk:
    """テスト用Chunkを作成"""
    if rows is None:
        rows = [
            create_input_row("001", "こんにちは", "キャラA"),
            create_input_row("002", "さようなら", "キャラB"),
        ]
    return Chunk(
        chunk_id=chunk_id,
        file_name="test.xlsx",
        sheet_name="Sheet1",
        rows=rows,
        context_rows=context_rows or [],
    )


class MockTranslationProvider(TranslationProvider):
    """テスト用モックプロバイダー"""
    
    def __init__(
        self,
        api_key: str = "test_key",
        model: Optional[str] = None,
        responses: Optional[List[TranslationResponse]] = None,
        errors: Optional[List[Exception]] = None,
    ):
        super().__init__(api_key, model)
        self._responses = responses or []
        self._errors = errors or []
        self._call_count = 0
        self._calls: List[Dict[str, str]] = []
    
    @property
    def provider_name(self) -> str:
        return "mock"
    
    @property
    def default_model(self) -> str:
        return "mock-model"
    
    async def translate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> TranslationResponse:
        self._calls.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        })
        
        # エラーを返す場合
        if self._call_count < len(self._errors) and self._errors[self._call_count]:
            error = self._errors[self._call_count]
            self._call_count += 1
            raise error
        
        # レスポンスを返す場合
        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
            self._call_count += 1
            return response
        
        # デフォルトレスポンス
        self._call_count += 1
        return TranslationResponse(
            translations=[],
            input_tokens=100,
            output_tokens=50,
            response_time_ms=500,
            provider="mock",
            model="mock-model",
        )
    
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
        self.build_translation_prompt_calls = 0
        self.build_review_prompt_calls = 0
        self.build_consistency_prompt_calls = 0
        self.build_backtrans_prompt_calls = 0
    
    def build_system_prompt(self, target_lang: Optional[str] = None) -> str:
        self.build_system_prompt_calls += 1
        return "System prompt"
    
    def build_translation_prompt(
        self,
        texts: List[Dict],
        glossary_entries: Optional[List] = None,
        character_profiles: Optional[List] = None,
        context_lines: Optional[List] = None,
    ) -> str:
        self.build_translation_prompt_calls += 1
        return "Translation prompt"
    
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


# ============================================================================
# ユニットテスト
# ============================================================================

class TestTranslationResponse:
    """TranslationResponseのテスト"""
    
    def test_create_response(self):
        """レスポンスの作成"""
        response = TranslationResponse(
            translations=[{"text_id": "001", "translated_text": "Hello"}],
            input_tokens=100,
            output_tokens=50,
            response_time_ms=500,
            provider="claude",
            model="claude-3-5-sonnet",
        )
        
        assert len(response.translations) == 1
        assert response.input_tokens == 100
        assert response.output_tokens == 50
        assert response.provider == "claude"


class TestTranslationProviderParseJson:
    """TranslationProvider._parse_json_responseのテスト"""
    
    def test_parse_translations_key(self):
        """translationsキーを持つJSONの解析"""
        provider = MockTranslationProvider()
        content = '{"translations": [{"text_id": "001", "translated_text": "Hello"}]}'
        
        result = provider._parse_json_response(content)
        
        assert len(result) == 1
        assert result[0]["text_id"] == "001"
    
    def test_parse_reviews_key(self):
        """reviewsキーを持つJSONの解析（2nd pass）"""
        provider = MockTranslationProvider()
        content = '{"reviews": [{"text_id": "001", "revised_translation": "Hello"}]}'
        
        result = provider._parse_json_response(content)
        
        assert len(result) == 1
        assert result[0]["text_id"] == "001"
    
    def test_parse_consistency_fixes_key(self):
        """consistency_fixesキーを持つJSONの解析（3rd pass）"""
        provider = MockTranslationProvider()
        content = '{"consistency_fixes": [{"text_id": "001", "revised_translation": "Hello"}]}'
        
        result = provider._parse_json_response(content)
        
        assert len(result) == 1
    
    def test_parse_backtranslations_key(self):
        """backtranslationsキーを持つJSONの解析（4th pass）"""
        provider = MockTranslationProvider()
        content = '{"backtranslations": [{"text_id": "001", "backtranslation": "こんにちは"}]}'
        
        result = provider._parse_json_response(content)
        
        assert len(result) == 1
    
    def test_parse_with_code_block(self):
        """コードブロック付きJSONの解析"""
        provider = MockTranslationProvider()
        content = '```json\n{"translations": [{"text_id": "001"}]}\n```'
        
        result = provider._parse_json_response(content)
        
        assert len(result) == 1
    
    def test_parse_list_directly(self):
        """リスト形式のJSONの解析"""
        provider = MockTranslationProvider()
        content = '[{"text_id": "001", "translated_text": "Hello"}]'
        
        result = provider._parse_json_response(content)
        
        assert len(result) == 1
    
    def test_parse_invalid_json_raises_error(self):
        """無効なJSONはエラー"""
        provider = MockTranslationProvider()
        content = 'not valid json'
        
        with pytest.raises(ResponseParseError):
            provider._parse_json_response(content)


class TestTranslationServiceBasic:
    """TranslationService基本機能のテスト"""
    
    def test_get_provider(self):
        """プロバイダーの取得"""
        mock_provider = MockTranslationProvider()
        service = TranslationService(
            providers={"mock": mock_provider},
            default_provider="mock",
        )
        
        provider = service.get_provider()
        assert provider == mock_provider
    
    def test_get_provider_by_name(self):
        """名前指定でプロバイダーを取得"""
        mock1 = MockTranslationProvider()
        mock2 = MockTranslationProvider()
        service = TranslationService(
            providers={"mock1": mock1, "mock2": mock2},
            default_provider="mock1",
        )
        
        provider = service.get_provider("mock2")
        assert provider == mock2
    
    def test_get_provider_not_found(self):
        """存在しないプロバイダーはエラー"""
        service = TranslationService(
            providers={"mock": MockTranslationProvider()},
            default_provider="mock",
        )
        
        with pytest.raises(ValueError) as exc_info:
            service.get_provider("nonexistent")
        assert "見つかりません" in str(exc_info.value)
    
    def test_get_passes_for_draft_mode(self):
        """Draftモードは1st passのみ"""
        service = TranslationService(
            providers={"mock": MockTranslationProvider()},
            default_provider="mock",
        )
        
        passes = service._get_passes_for_mode("draft")
        assert passes == [1]
    
    def test_get_passes_for_standard_mode(self):
        """Standardモードは1st+2nd+3rd pass"""
        service = TranslationService(
            providers={"mock": MockTranslationProvider()},
            default_provider="mock",
        )
        
        passes = service._get_passes_for_mode("standard")
        assert passes == [1, 2, 3]
    
    def test_get_passes_for_thorough_mode(self):
        """Thoroughモードは全パス"""
        service = TranslationService(
            providers={"mock": MockTranslationProvider()},
            default_provider="mock",
        )
        
        passes = service._get_passes_for_mode("thorough")
        assert passes == [1, 2, 3, 4]


class TestTranslationServiceRetry:
    """TranslationService.translate_with_retryのテスト"""
    
    @pytest.mark.asyncio
    async def test_successful_translation(self):
        """成功する翻訳"""
        response = TranslationResponse(
            translations=[{"text_id": "001", "translated_text": "Hello"}],
            input_tokens=100,
            output_tokens=50,
            response_time_ms=500,
            provider="mock",
            model="mock-model",
        )
        provider = MockTranslationProvider(responses=[response])
        service = TranslationService(
            providers={"mock": provider},
            default_provider="mock",
        )
        
        result = await service.translate_with_retry(
            provider, "system", "user"
        )
        
        assert result == response
        assert provider.call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self):
        """レート制限エラーでリトライ"""
        response = TranslationResponse(
            translations=[],
            input_tokens=100,
            output_tokens=50,
            response_time_ms=500,
            provider="mock",
            model="mock-model",
        )
        provider = MockTranslationProvider(
            responses=[response],
            errors=[RateLimitError("Rate limit", retry_after=0.01), None],
        )
        service = TranslationService(
            providers={"mock": provider},
            default_provider="mock",
            max_retries=3,
        )
        
        result = await service.translate_with_retry(
            provider, "system", "user"
        )
        
        assert provider.call_count == 2  # 1回失敗 + 1回成功
    
    @pytest.mark.asyncio
    async def test_retry_on_api_error(self):
        """APIエラーでリトライ"""
        response = TranslationResponse(
            translations=[],
            input_tokens=100,
            output_tokens=50,
            response_time_ms=500,
            provider="mock",
            model="mock-model",
        )
        provider = MockTranslationProvider(
            responses=[response],
            errors=[APIError("API error"), None],
        )
        service = TranslationService(
            providers={"mock": provider},
            default_provider="mock",
            max_retries=3,
        )
        
        result = await service.translate_with_retry(
            provider, "system", "user"
        )
        
        assert provider.call_count == 2
    
    @pytest.mark.asyncio
    async def test_no_retry_on_auth_error(self):
        """認証エラーはリトライしない"""
        provider = MockTranslationProvider(
            errors=[AuthenticationError("Auth error")],
        )
        service = TranslationService(
            providers={"mock": provider},
            default_provider="mock",
            max_retries=3,
        )
        
        with pytest.raises(AuthenticationError):
            await service.translate_with_retry(
                provider, "system", "user"
            )
        
        assert provider.call_count == 1  # リトライなし
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """リトライ上限超過"""
        provider = MockTranslationProvider(
            errors=[
                RateLimitError("Rate limit", retry_after=0.01),
                RateLimitError("Rate limit", retry_after=0.01),
                RateLimitError("Rate limit", retry_after=0.01),
                RateLimitError("Rate limit", retry_after=0.01),
            ],
        )
        service = TranslationService(
            providers={"mock": provider},
            default_provider="mock",
            max_retries=3,
        )
        
        with pytest.raises(TranslationError) as exc_info:
            await service.translate_with_retry(
                provider, "system", "user"
            )
        
        assert "リトライ上限" in str(exc_info.value)
        assert provider.call_count == 4  # 初回 + 3回リトライ


class TestCreateProvider:
    """create_provider関数のテスト"""
    
    def test_create_claude_provider(self):
        """Claudeプロバイダーの作成"""
        provider = create_provider("claude", "test_key")
        assert isinstance(provider, ClaudeProvider)
        assert provider.provider_name == "claude"
    
    def test_create_gemini_provider(self):
        """Geminiプロバイダーの作成"""
        provider = create_provider("gemini", "test_key")
        assert isinstance(provider, GeminiProvider)
        assert provider.provider_name == "gemini"
    
    def test_create_gpt4_provider(self):
        """GPT-4プロバイダーの作成"""
        provider = create_provider("gpt4", "test_key")
        assert isinstance(provider, GPT4Provider)
        assert provider.provider_name == "gpt4"
    
    def test_create_unknown_provider_raises_error(self):
        """不明なプロバイダーはエラー"""
        with pytest.raises(ValueError) as exc_info:
            create_provider("unknown", "test_key")
        assert "不明なプロバイダー" in str(exc_info.value)



# ============================================================================
# Property-Based Tests
# ============================================================================

@st.composite
def quality_mode_strategy(draw):
    """品質モードを生成するストラテジー"""
    return draw(st.sampled_from(["draft", "standard", "thorough"]))


@st.composite
def input_rows_for_translation_strategy(draw):
    """翻訳用InputRowリストを生成するストラテジー"""
    num_rows = draw(st.integers(min_value=1, max_value=5))
    rows = []
    
    for r in range(num_rows):
        text_id = f"ROW_{r+1:03d}"
        source_text = draw(st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N')),
            min_size=1,
            max_size=50
        ))
        character = draw(st.one_of(
            st.none(),
            st.sampled_from(["キャラA", "キャラB", "キャラC"])
        ))
        
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


class TestProperty18QualityModeExecution:
    """
    Property 18: 品質モード実行
    
    *For any* 品質モード（Draft/Standard/Thorough）について、
    Draftは1st passのみ、Standardは1st+2nd+3rd pass、Thoroughは全パスが実行される
    
    **Feature: excel-translation-api, Property 18: 品質モード実行**
    **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
    """
    
    @given(mode=quality_mode_strategy())
    @settings(max_examples=100, deadline=None)
    def test_mode_determines_passes(self, mode: str):
        """品質モードに応じたパスが決定される"""
        service = TranslationService(
            providers={"mock": MockTranslationProvider()},
            default_provider="mock",
        )
        
        passes = service._get_passes_for_mode(mode)
        
        if mode == "draft":
            assert passes == [1], "Draftモードは1st passのみ"
        elif mode == "standard":
            assert passes == [1, 2, 3], "Standardモードは1st+2nd+3rd pass"
        elif mode == "thorough":
            assert passes == [1, 2, 3, 4], "Thoroughモードは全パス"
    
    @pytest.mark.asyncio
    @given(mode=quality_mode_strategy(), rows=input_rows_for_translation_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_pipeline_executes_correct_passes(self, mode: str, rows: List[InputRow]):
        """パイプラインが正しいパスを実行する"""
        assume(len(rows) > 0)
        
        # 各パスのレスポンスを準備
        def create_pass_response(pass_num: int, text_ids: List[str]) -> TranslationResponse:
            if pass_num == 1:
                translations = [
                    {"text_id": tid, "translated_text": f"Translated {tid}"}
                    for tid in text_ids
                ]
            elif pass_num == 2:
                # 2nd passは修正が必要な行のみ返す（空でもOK）
                translations = []
            elif pass_num == 3:
                # 3rd passも修正が必要な行のみ
                translations = []
            else:
                # 4th passはバックトランスレーション
                translations = []
            
            return TranslationResponse(
                translations=translations,
                input_tokens=100,
                output_tokens=50,
                response_time_ms=500,
                provider="mock",
                model="mock-model",
            )
        
        text_ids = [r.text_id for r in rows]
        
        # 十分な数のレスポンスを準備
        responses = [
            create_pass_response(1, text_ids),
            create_pass_response(2, text_ids),
            create_pass_response(3, text_ids),
            create_pass_response(4, text_ids),
        ]
        
        provider = MockTranslationProvider(responses=responses)
        service = TranslationService(
            providers={"mock": provider},
            default_provider="mock",
        )
        
        chunk = Chunk(
            chunk_id="test_chunk",
            file_name="test.xlsx",
            sheet_name="Sheet1",
            rows=rows,
            context_rows=[],
        )
        
        prompt_builder = MockPromptBuilder()
        
        results, stats = await service.run_pipeline(
            chunk=chunk,
            prompt_builder=prompt_builder,
            mode=mode,
        )
        
        # 実行されたパスを検証
        expected_passes = service._get_passes_for_mode(mode)
        
        # 1st passは必ず実行される
        assert 1 in stats["passes_executed"], "1st passは必ず実行されるべき"
        
        # モードに応じたパスが実行される
        for pass_num in expected_passes:
            if pass_num == 1:
                assert prompt_builder.build_translation_prompt_calls >= 1
            elif pass_num == 2:
                assert prompt_builder.build_review_prompt_calls >= 1
            elif pass_num == 3:
                assert prompt_builder.build_consistency_prompt_calls >= 1
            # 4th passは修正があった行のみなので、呼ばれない場合もある
    
    @given(mode=quality_mode_strategy())
    @settings(max_examples=100, deadline=None)
    def test_draft_mode_is_fastest(self, mode: str):
        """Draftモードは最も少ないパス数"""
        service = TranslationService(
            providers={"mock": MockTranslationProvider()},
            default_provider="mock",
        )
        
        draft_passes = service._get_passes_for_mode("draft")
        current_passes = service._get_passes_for_mode(mode)
        
        assert len(draft_passes) <= len(current_passes), "Draftモードは最も少ないパス数"
    
    @given(mode=quality_mode_strategy())
    @settings(max_examples=100, deadline=None)
    def test_thorough_mode_is_most_complete(self, mode: str):
        """Thoroughモードは最も多いパス数"""
        service = TranslationService(
            providers={"mock": MockTranslationProvider()},
            default_provider="mock",
        )
        
        thorough_passes = service._get_passes_for_mode("thorough")
        current_passes = service._get_passes_for_mode(mode)
        
        assert len(thorough_passes) >= len(current_passes), "Thoroughモードは最も多いパス数"
    
    @given(mode=quality_mode_strategy())
    @settings(max_examples=100, deadline=None)
    def test_passes_are_sequential(self, mode: str):
        """パスは連続した番号"""
        service = TranslationService(
            providers={"mock": MockTranslationProvider()},
            default_provider="mock",
        )
        
        passes = service._get_passes_for_mode(mode)
        
        # パスは1から始まる連続した番号
        assert passes[0] == 1, "パスは1から始まる"
        for i in range(1, len(passes)):
            assert passes[i] == passes[i-1] + 1, "パスは連続した番号"


class TestProperty27Retry:
    """
    Property 27: リトライ
    
    *For any* APIエラーが発生した場合、指数バックオフでリトライが実行され、
    リトライ上限に達した場合はエラーが記録されて次の行に進む
    
    **Feature: excel-translation-api, Property 27: リトライ**
    **Validates: Requirements 13.1, 13.2**
    """
    
    @pytest.mark.asyncio
    @given(
        num_failures=st.integers(min_value=0, max_value=5),
        max_retries=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100, deadline=None)
    async def test_retry_count_matches_failures(self, num_failures: int, max_retries: int):
        """リトライ回数は失敗回数に応じる"""
        # 成功するレスポンス
        success_response = TranslationResponse(
            translations=[],
            input_tokens=100,
            output_tokens=50,
            response_time_ms=500,
            provider="mock",
            model="mock-model",
        )
        
        # エラーリストを作成（num_failures回失敗後に成功）
        errors = [RateLimitError("Rate limit", retry_after=0.001) for _ in range(num_failures)]
        errors.append(None)  # 最後は成功
        
        responses = [success_response]
        
        provider = MockTranslationProvider(responses=responses, errors=errors)
        service = TranslationService(
            providers={"mock": provider},
            default_provider="mock",
            max_retries=max_retries,
        )
        
        if num_failures <= max_retries:
            # リトライ内で成功
            result = await service.translate_with_retry(
                provider, "system", "user"
            )
            assert provider.call_count == num_failures + 1
        else:
            # リトライ上限超過
            with pytest.raises(TranslationError):
                await service.translate_with_retry(
                    provider, "system", "user"
                )
            assert provider.call_count == max_retries + 1
    
    @pytest.mark.asyncio
    @given(max_retries=st.integers(min_value=1, max_value=5))
    @settings(max_examples=100, deadline=None)
    async def test_retry_limit_is_respected(self, max_retries: int):
        """リトライ上限が尊重される"""
        # 常に失敗するプロバイダー
        errors = [RateLimitError("Rate limit", retry_after=0.001) for _ in range(max_retries + 2)]
        
        provider = MockTranslationProvider(errors=errors)
        service = TranslationService(
            providers={"mock": provider},
            default_provider="mock",
            max_retries=max_retries,
        )
        
        with pytest.raises(TranslationError):
            await service.translate_with_retry(
                provider, "system", "user"
            )
        
        # 初回 + max_retries回のリトライ
        assert provider.call_count == max_retries + 1
    
    @pytest.mark.asyncio
    @given(max_retries=st.integers(min_value=1, max_value=3))
    @settings(max_examples=100, deadline=None)
    async def test_errors_are_recorded(self, max_retries: int):
        """エラーが記録される"""
        errors = [RateLimitError("Rate limit", retry_after=0.001) for _ in range(max_retries + 2)]
        
        provider = MockTranslationProvider(errors=errors)
        service = TranslationService(
            providers={"mock": provider},
            default_provider="mock",
            max_retries=max_retries,
        )
        
        try:
            await service.translate_with_retry(
                provider, "system", "user"
            )
        except TranslationError:
            pass
        
        # エラーが記録されている
        stats = service.get_statistics()
        assert len(stats["errors"]) > 0, "エラーが記録されるべき"
    
    @pytest.mark.asyncio
    @given(max_retries=st.integers(min_value=1, max_value=3))
    @settings(max_examples=100, deadline=None)
    async def test_retry_count_is_tracked(self, max_retries: int):
        """リトライ回数が追跡される"""
        # 1回失敗して成功
        success_response = TranslationResponse(
            translations=[],
            input_tokens=100,
            output_tokens=50,
            response_time_ms=500,
            provider="mock",
            model="mock-model",
        )
        
        errors = [RateLimitError("Rate limit", retry_after=0.001), None]
        
        provider = MockTranslationProvider(responses=[success_response], errors=errors)
        service = TranslationService(
            providers={"mock": provider},
            default_provider="mock",
            max_retries=max_retries,
        )
        
        await service.translate_with_retry(
            provider, "system", "user"
        )
        
        stats = service.get_statistics()
        assert stats["total_retries"] >= 1, "リトライ回数が追跡されるべき"


class TestProperty29LLMComparison:
    """
    Property 29: LLM比較
    
    *For any* 比較モードでの翻訳について、選択された全てのLLMプロバイダーから結果が取得され、
    出力には各LLMの結果が別列として含まれる
    
    **Feature: excel-translation-api, Property 29: LLM比較**
    **Validates: Requirements 15.3, 15.4, 15.5**
    """
    
    @pytest.mark.asyncio
    @given(
        num_providers=st.integers(min_value=1, max_value=3),
        rows=input_rows_for_translation_strategy(),
    )
    @settings(max_examples=100, deadline=None)
    async def test_comparison_returns_all_providers(self, num_providers: int, rows: List[InputRow]):
        """比較モードは全プロバイダーの結果を返す"""
        assume(len(rows) > 0)
        
        # プロバイダーを作成
        providers = {}
        provider_names = []
        
        for i in range(num_providers):
            name = f"provider_{i}"
            provider_names.append(name)
            
            text_ids = [r.text_id for r in rows]
            response = TranslationResponse(
                translations=[
                    {"text_id": tid, "translated_text": f"Translated by {name}: {tid}"}
                    for tid in text_ids
                ],
                input_tokens=100,
                output_tokens=50,
                response_time_ms=500,
                provider=name,
                model=f"{name}-model",
            )
            
            providers[name] = MockTranslationProvider(responses=[response])
        
        service = TranslationService(
            providers=providers,
            default_provider=provider_names[0],
        )
        
        chunk = Chunk(
            chunk_id="test_chunk",
            file_name="test.xlsx",
            sheet_name="Sheet1",
            rows=rows,
            context_rows=[],
        )
        
        prompt_builder = MockPromptBuilder()
        
        comparison_results = await service.run_comparison(
            chunk=chunk,
            prompt_builder=prompt_builder,
            provider_names=provider_names,
            mode="draft",
        )
        
        # 全プロバイダーの結果が含まれる
        assert len(comparison_results) == num_providers
        for name in provider_names:
            assert name in comparison_results, f"プロバイダー {name} の結果が含まれるべき"
    
    @pytest.mark.asyncio
    @given(rows=input_rows_for_translation_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_comparison_results_are_separate(self, rows: List[InputRow]):
        """比較結果は各プロバイダーで分離されている"""
        assume(len(rows) > 0)
        
        text_ids = [r.text_id for r in rows]
        
        # 2つのプロバイダーを作成（異なる翻訳結果）
        response1 = TranslationResponse(
            translations=[
                {"text_id": tid, "translated_text": f"Provider1: {tid}"}
                for tid in text_ids
            ],
            input_tokens=100,
            output_tokens=50,
            response_time_ms=500,
            provider="provider1",
            model="model1",
        )
        
        response2 = TranslationResponse(
            translations=[
                {"text_id": tid, "translated_text": f"Provider2: {tid}"}
                for tid in text_ids
            ],
            input_tokens=100,
            output_tokens=50,
            response_time_ms=600,
            provider="provider2",
            model="model2",
        )
        
        providers = {
            "provider1": MockTranslationProvider(responses=[response1]),
            "provider2": MockTranslationProvider(responses=[response2]),
        }
        
        service = TranslationService(
            providers=providers,
            default_provider="provider1",
        )
        
        chunk = Chunk(
            chunk_id="test_chunk",
            file_name="test.xlsx",
            sheet_name="Sheet1",
            rows=rows,
            context_rows=[],
        )
        
        prompt_builder = MockPromptBuilder()
        
        comparison_results = await service.run_comparison(
            chunk=chunk,
            prompt_builder=prompt_builder,
            provider_names=["provider1", "provider2"],
            mode="draft",
        )
        
        # 結果が分離されている
        results1, stats1 = comparison_results["provider1"]
        results2, stats2 = comparison_results["provider2"]
        
        # 各プロバイダーの結果は異なる
        if len(results1) > 0 and len(results2) > 0:
            assert results1[0].pass_1 != results2[0].pass_1, "各プロバイダーの結果は異なるべき"
    
    @pytest.mark.asyncio
    @given(rows=input_rows_for_translation_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_comparison_includes_response_time(self, rows: List[InputRow]):
        """比較結果にはレスポンス時間が含まれる"""
        assume(len(rows) > 0)
        
        text_ids = [r.text_id for r in rows]
        
        response = TranslationResponse(
            translations=[
                {"text_id": tid, "translated_text": f"Translated: {tid}"}
                for tid in text_ids
            ],
            input_tokens=100,
            output_tokens=50,
            response_time_ms=500,
            provider="mock",
            model="mock-model",
        )
        
        provider = MockTranslationProvider(responses=[response])
        service = TranslationService(
            providers={"mock": provider},
            default_provider="mock",
        )
        
        chunk = Chunk(
            chunk_id="test_chunk",
            file_name="test.xlsx",
            sheet_name="Sheet1",
            rows=rows,
            context_rows=[],
        )
        
        prompt_builder = MockPromptBuilder()
        
        comparison_results = await service.run_comparison(
            chunk=chunk,
            prompt_builder=prompt_builder,
            provider_names=["mock"],
            mode="draft",
        )
        
        results, stats = comparison_results["mock"]
        
        # レスポンス時間が記録されている
        assert stats["total_response_time_ms"] > 0, "レスポンス時間が記録されるべき"
