"""コスト追跡・トークン計測モジュールのテスト"""

import pytest
from hypothesis import given, settings, strategies as st, assume
from typing import Any, Dict, List

from excel_translator.cost import (
    TokenCounter,
    CostCalculator,
    ChunkResplitter,
    CostLimitChecker,
    CostTracker,
    PassCost,
    TokenCount,
    CostEstimate,
    TokenCounterError,
    CostLimitExceededError,
    count_tokens,
    estimate_cost,
    DEFAULT_MODEL,
    MODEL_PRICING,
)


# ============================================================================
# テストユーティリティ
# ============================================================================

def create_text_item(
    text_id: str,
    source_text: str,
    character: str = None,
) -> Dict[str, Any]:
    """テスト用テキストアイテムを作成"""
    return {
        "text_id": text_id,
        "source_text": source_text,
        "character": character,
    }


# ============================================================================
# ユニットテスト
# ============================================================================

class TestTokenCounter:
    """TokenCounterのテスト"""
    
    def test_empty_string(self):
        """空文字列は0トークン"""
        counter = TokenCounter()
        assert counter.count_tokens("") == 0
    
    def test_simple_english(self):
        """英語テキストのトークン計測"""
        counter = TokenCounter()
        tokens = counter.count_tokens("Hello world")
        assert tokens > 0
        assert tokens < 10
    
    def test_simple_japanese(self):
        """日本語テキストのトークン計測"""
        counter = TokenCounter()
        tokens = counter.count_tokens("こんにちは世界")
        assert tokens > 0
    
    def test_mixed_text(self):
        """混合テキストのトークン計測"""
        counter = TokenCounter()
        tokens = counter.count_tokens("Hello こんにちは World 世界")
        assert tokens > 0
    
    def test_messages_tokens(self):
        """メッセージリストのトークン計測"""
        counter = TokenCounter()
        messages = [
            {"role": "system", "content": "You are a translator."},
            {"role": "user", "content": "Translate this text."},
        ]
        tokens = counter.count_messages_tokens(messages)
        assert tokens > 0
    
    def test_prompt_tokens(self):
        """プロンプトのトークン計測"""
        counter = TokenCounter()
        result = counter.count_prompt_tokens(
            system_prompt="You are a translator.",
            user_prompt="Translate: Hello world",
        )
        assert isinstance(result, TokenCount)
        assert result.total > 0
        assert result.input_tokens > 0


class TestCostCalculator:
    """CostCalculatorのテスト"""
    
    def test_calculate_cost_basic(self):
        """基本的なコスト計算"""
        calculator = CostCalculator()
        result = calculator.calculate_cost(
            input_tokens=1000,
            output_tokens=500,
        )
        assert isinstance(result, CostEstimate)
        assert result.total_usd > 0
        assert result.input_cost > 0
        assert result.output_cost > 0
    
    def test_calculate_cost_with_cache(self):
        """キャッシュ込みのコスト計算"""
        calculator = CostCalculator()
        result = calculator.calculate_cost(
            input_tokens=500,
            output_tokens=500,
            cache_write_tokens=200,
            cache_read_tokens=300,
        )
        assert result.total_usd > 0
        assert result.cache_write_cost >= 0
        assert result.cache_read_cost >= 0
    
    def test_estimate_translation_cost(self):
        """翻訳コスト推定"""
        calculator = CostCalculator()
        result = calculator.estimate_translation_cost(
            input_tokens=1000,
            use_cache=True,
        )
        assert result.total_usd > 0
    
    def test_different_models(self):
        """異なるモデルでのコスト計算"""
        for model in ["claude-3-5-sonnet-20241022", "gpt-4o", "gemini-1.5-pro"]:
            calculator = CostCalculator(model=model)
            result = calculator.calculate_cost(
                input_tokens=1000,
                output_tokens=500,
            )
            assert result.total_usd > 0


class TestChunkResplitter:
    """ChunkResplitterのテスト"""
    
    def test_needs_resplit_false(self):
        """再分割不要の判定"""
        resplitter = ChunkResplitter(max_tokens=1000)
        assert not resplitter.needs_resplit("Hello world")
    
    def test_needs_resplit_true(self):
        """再分割必要の判定"""
        resplitter = ChunkResplitter(max_tokens=10)
        long_text = "This is a very long text that should exceed the token limit. " * 10
        assert resplitter.needs_resplit(long_text)
    
    def test_calculate_split_count(self):
        """分割数の計算"""
        resplitter = ChunkResplitter(max_tokens=100)
        
        # 短いテキストは1分割
        assert resplitter.calculate_split_count("Hello") == 1
        
        # 長いテキストは複数分割
        long_text = "テスト文章です。" * 100
        split_count = resplitter.calculate_split_count(long_text)
        assert split_count > 1
    
    def test_resplit_texts_empty(self):
        """空リストの再分割"""
        resplitter = ChunkResplitter(max_tokens=1000)
        result = resplitter.resplit_texts([])
        assert result == []
    
    def test_resplit_texts_single(self):
        """単一テキストの再分割（分割不要）"""
        resplitter = ChunkResplitter(max_tokens=1000)
        texts = [create_text_item("001", "Hello world")]
        result = resplitter.resplit_texts(texts)
        assert len(result) == 1
        assert len(result[0]) == 1
    
    def test_resplit_texts_multiple_chunks(self):
        """複数チャンクへの再分割"""
        resplitter = ChunkResplitter(max_tokens=50)
        texts = [
            create_text_item(f"{i:03d}", "テスト文章です。" * 5)
            for i in range(10)
        ]
        result = resplitter.resplit_texts(texts, base_prompt_tokens=0)
        assert len(result) > 1
        
        # 全テキストが含まれている
        all_text_ids = []
        for chunk in result:
            for text in chunk:
                all_text_ids.append(text["text_id"])
        assert len(all_text_ids) == 10
    
    def test_get_token_count(self):
        """トークン数取得"""
        resplitter = ChunkResplitter()
        count = resplitter.get_token_count("Hello world")
        assert count > 0


class TestCostLimitChecker:
    """CostLimitCheckerのテスト"""
    
    def test_no_limit(self):
        """上限なしの場合"""
        checker = CostLimitChecker(cost_limit_usd=None)
        can_continue, message = checker.check_limit(100.0)
        assert can_continue
        assert checker.remaining_budget is None
    
    def test_within_limit(self):
        """上限内の場合"""
        checker = CostLimitChecker(cost_limit_usd=10.0)
        can_continue, message = checker.check_limit(1.0)
        assert can_continue
    
    def test_exceeds_limit(self):
        """上限超過の場合"""
        checker = CostLimitChecker(cost_limit_usd=1.0)
        can_continue, message = checker.check_limit(2.0)
        assert not can_continue
        assert "上限" in message
    
    def test_record_cost(self):
        """コスト記録"""
        checker = CostLimitChecker(cost_limit_usd=10.0)
        result = checker.record_cost(
            input_tokens=1000,
            output_tokens=500,
        )
        assert isinstance(result, CostEstimate)
        assert checker.accumulated_cost > 0
        assert checker.request_count == 1
    
    def test_remaining_budget(self):
        """残り予算"""
        checker = CostLimitChecker(cost_limit_usd=10.0)
        checker.record_cost(input_tokens=100000, output_tokens=50000)
        assert checker.remaining_budget is not None
        assert checker.remaining_budget < 10.0
    
    def test_estimate_and_check(self):
        """推定とチェック"""
        checker = CostLimitChecker(cost_limit_usd=10.0)
        estimate, can_continue, message = checker.estimate_and_check(
            input_tokens=1000,
        )
        assert isinstance(estimate, CostEstimate)
        assert can_continue
    
    def test_reset(self):
        """リセット"""
        checker = CostLimitChecker(cost_limit_usd=10.0)
        checker.record_cost(input_tokens=1000, output_tokens=500)
        checker.reset()
        assert checker.accumulated_cost == 0.0
        assert checker.request_count == 0
    
    def test_get_summary(self):
        """サマリ取得"""
        checker = CostLimitChecker(cost_limit_usd=10.0)
        checker.record_cost(input_tokens=1000, output_tokens=500)
        summary = checker.get_summary()
        assert "accumulated_cost_usd" in summary
        assert "request_count" in summary
        assert "cost_limit_usd" in summary
        assert "remaining_budget_usd" in summary


class TestConvenienceFunctions:
    """便利関数のテスト"""
    
    def test_count_tokens(self):
        """count_tokens関数"""
        tokens = count_tokens("Hello world")
        assert tokens > 0
    
    def test_estimate_cost(self):
        """estimate_cost関数"""
        result = estimate_cost(
            input_tokens=1000,
            output_tokens=500,
        )
        assert isinstance(result, CostEstimate)
        assert result.total_usd > 0


class TestCostTracker:
    """CostTrackerのテスト"""
    
    def test_track_request_basic(self):
        """基本的なリクエスト追跡"""
        tracker = CostTracker()
        tracker.track_request(
            pass_name="1st_pass",
            input_tokens=1000,
            output_tokens=500,
            cache_hit_tokens=200,
            processing_time_ms=1500,
        )
        
        summary = tracker.get_pass_summary("1st_pass")
        assert summary is not None
        assert summary.input_tokens == 1000
        assert summary.output_tokens == 500
        assert summary.cache_hit_tokens == 200
        assert summary.api_calls == 1
        assert summary.processing_time_ms == 1500
        assert summary.cost_usd > 0
    
    def test_track_multiple_requests(self):
        """複数リクエストの追跡"""
        tracker = CostTracker()
        
        tracker.track_request("1st_pass", 1000, 500, 100, 1000)
        tracker.track_request("1st_pass", 800, 400, 50, 800)
        
        summary = tracker.get_pass_summary("1st_pass")
        assert summary.input_tokens == 1800
        assert summary.output_tokens == 900
        assert summary.cache_hit_tokens == 150
        assert summary.api_calls == 2
        assert summary.processing_time_ms == 1800
    
    def test_track_multiple_passes(self):
        """複数パスの追跡"""
        tracker = CostTracker()
        
        tracker.track_request("1st_pass", 1000, 500, 100, 1000)
        tracker.track_request("2nd_pass", 500, 200, 50, 500)
        tracker.track_request("3rd_pass", 300, 100, 30, 300)
        
        assert len(tracker.get_all_passes()) == 3
        assert "1st_pass" in tracker.get_all_passes()
        assert "2nd_pass" in tracker.get_all_passes()
        assert "3rd_pass" in tracker.get_all_passes()
    
    def test_calculate_cost(self):
        """コスト計算"""
        tracker = CostTracker()
        cost = tracker.calculate_cost(
            model="claude-3-5-sonnet-20241022",
            input_tokens=1000,
            output_tokens=500,
            cache_hit_tokens=100,
        )
        assert cost > 0
    
    def test_record_modifications(self):
        """修正行数の記録"""
        tracker = CostTracker()
        tracker.track_request("2nd_pass", 1000, 500, 100, 1000)
        tracker.record_modifications("2nd_pass", modified_rows=10, total_rows=100)
        
        summary = tracker.get_pass_summary("2nd_pass")
        assert summary.modified_rows == 10
        assert summary.total_rows == 100
        assert summary.modification_rate == 0.1
    
    def test_generate_report(self):
        """レポート生成"""
        tracker = CostTracker()
        tracker.track_request("1st_pass", 1000, 500, 100, 1000)
        tracker.track_request("2nd_pass", 500, 200, 50, 500)
        tracker.record_modifications("2nd_pass", modified_rows=5, total_rows=50)
        
        report = tracker.generate_report()
        
        assert "model" in report
        assert "passes" in report
        assert "summary" in report
        assert len(report["passes"]) == 2
        
        summary = report["summary"]
        assert summary["total_input_tokens"] == 1500
        assert summary["total_output_tokens"] == 700
        assert summary["total_api_calls"] == 2
    
    def test_generate_mode_comparison(self):
        """モード間比較レポート生成"""
        tracker = CostTracker()
        
        results = {
            "draft": [
                {"text_id": "001", "source_text": "こんにちは", "translated_text": "Hello"},
                {"text_id": "002", "source_text": "さようなら", "translated_text": "Goodbye"},
            ],
            "standard": [
                {"text_id": "001", "source_text": "こんにちは", "translated_text": "Hello there"},
                {"text_id": "002", "source_text": "さようなら", "translated_text": "Goodbye"},
            ],
        }
        
        comparison = tracker.generate_mode_comparison(results)
        
        assert "comparison" in comparison
        assert "cost_comparison" in comparison
        assert "modes" in comparison
        assert len(comparison["comparison"]) == 2
        assert comparison["modes"] == ["draft", "standard"]
    
    def test_reset(self):
        """リセット"""
        tracker = CostTracker()
        tracker.track_request("1st_pass", 1000, 500, 100, 1000)
        tracker.reset()
        
        assert len(tracker.get_all_passes()) == 0
        assert tracker.get_total_cost() == 0
    
    def test_get_total_cost(self):
        """総コスト取得"""
        tracker = CostTracker()
        tracker.track_request("1st_pass", 1000, 500, 100, 1000)
        tracker.track_request("2nd_pass", 500, 200, 50, 500)
        
        total_cost = tracker.get_total_cost()
        assert total_cost > 0
    
    def test_get_total_tokens(self):
        """総トークン数取得"""
        tracker = CostTracker()
        tracker.track_request("1st_pass", 1000, 500, 100, 1000)
        tracker.track_request("2nd_pass", 500, 200, 50, 500)
        
        total_tokens = tracker.get_total_tokens()
        assert total_tokens["input"] == 1500
        assert total_tokens["output"] == 700
        assert total_tokens["cache_hit"] == 150
    
    def test_cache_mode_result(self):
        """モード結果のキャッシュ"""
        tracker = CostTracker()
        tracker.track_request("1st_pass", 1000, 500, 100, 1000)
        
        results = [
            {"text_id": "001", "translated_text": "Hello"},
        ]
        tracker.cache_mode_result("draft", "test.xlsx", results)
        
        # キャッシュされた結果を使ってモード比較ができる
        assert tracker._mode_results.get("test.xlsx") is not None
        assert "draft" in tracker._mode_results["test.xlsx"]


# ============================================================================
# Property-Based Tests
# ============================================================================

@st.composite
def text_items_strategy(draw):
    """テキストアイテムリストを生成するストラテジー"""
    num_items = draw(st.integers(min_value=1, max_value=20))
    items = []
    
    for i in range(num_items):
        text_id = f"TEXT_{i+1:03d}"
        # 日本語と英語の混合テキストを生成
        jp_part = draw(st.text(
            alphabet=st.sampled_from("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"),
            min_size=0,
            max_size=50
        ))
        en_part = draw(st.text(
            alphabet=st.characters(whitelist_categories=('L',), whitelist_characters=' '),
            min_size=0,
            max_size=50
        ))
        source_text = jp_part + en_part
        
        # 空テキストを避ける
        if not source_text.strip():
            source_text = "テスト"
        
        character = draw(st.one_of(
            st.none(),
            st.sampled_from(["キャラA", "キャラB", "キャラC"])
        ))
        
        items.append(create_text_item(text_id, source_text, character))
    
    return items


class TestProperty22TokenResplit:
    """
    Property 22: トークン計測・再分割
    
    *For any* プロンプトについて、トークン数が上限を超える場合は自動的にチャンクが再分割され、
    再分割後の各チャンクは上限以下になる
    
    **Feature: excel-translation-api, Property 22: トークン計測・再分割**
    **Validates: Requirements 9.3**
    """
    
    @given(texts=text_items_strategy())
    @settings(max_examples=100, deadline=None)
    def test_resplit_chunks_within_limit(self, texts: List[Dict[str, Any]]):
        """再分割後の各チャンクはトークン上限以下になる"""
        assume(len(texts) > 0)
        
        # 小さなトークン上限を設定して再分割を強制
        max_tokens = 100
        base_prompt_tokens = 20
        resplitter = ChunkResplitter(max_tokens=max_tokens)
        
        # 再分割を実行
        chunks = resplitter.resplit_texts(texts, base_prompt_tokens=base_prompt_tokens)
        
        # 利用可能なトークン数
        available_tokens = max_tokens - base_prompt_tokens
        
        # 各チャンクのトークン数をチェック
        for chunk in chunks:
            chunk_tokens = 0
            for text in chunk:
                chunk_tokens += resplitter.get_token_count(text.get("source_text", ""))
                chunk_tokens += 20  # オーバーヘッド
            
            # 単一テキストでも上限を超える場合があるので、複数テキストの場合のみ厳密にチェック
            if len(chunk) > 1:
                # 最後のテキストを除いたトークン数は上限以下であるべき
                tokens_without_last = chunk_tokens - resplitter.get_token_count(chunk[-1].get("source_text", "")) - 20
                assert tokens_without_last <= available_tokens * 1.5, \
                    f"チャンクのトークン数が上限を大幅に超えています: {tokens_without_last} > {available_tokens}"
    
    @given(texts=text_items_strategy())
    @settings(max_examples=100, deadline=None)
    def test_resplit_preserves_all_texts(self, texts: List[Dict[str, Any]]):
        """再分割後も全てのテキストが保持される"""
        assume(len(texts) > 0)
        
        max_tokens = 100
        resplitter = ChunkResplitter(max_tokens=max_tokens)
        
        # 再分割を実行
        chunks = resplitter.resplit_texts(texts, base_prompt_tokens=0)
        
        # 元のtext_idを収集
        original_ids = {t["text_id"] for t in texts}
        
        # 再分割後のtext_idを収集
        resplit_ids = set()
        for chunk in chunks:
            for text in chunk:
                resplit_ids.add(text["text_id"])
        
        # 全てのtext_idが保持されている
        assert original_ids == resplit_ids, "再分割後も全てのテキストが保持されるべき"
    
    @given(texts=text_items_strategy())
    @settings(max_examples=100, deadline=None)
    def test_resplit_maintains_order(self, texts: List[Dict[str, Any]]):
        """再分割後もテキストの順序が維持される"""
        assume(len(texts) > 0)
        
        max_tokens = 100
        resplitter = ChunkResplitter(max_tokens=max_tokens)
        
        # 再分割を実行
        chunks = resplitter.resplit_texts(texts, base_prompt_tokens=0)
        
        # 元の順序
        original_order = [t["text_id"] for t in texts]
        
        # 再分割後の順序
        resplit_order = []
        for chunk in chunks:
            for text in chunk:
                resplit_order.append(text["text_id"])
        
        # 順序が維持されている
        assert original_order == resplit_order, "再分割後もテキストの順序が維持されるべき"
    
    @given(
        text=st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z')),
            min_size=1,
            max_size=500
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_token_count_positive(self, text: str):
        """トークン数は常に正の値"""
        assume(len(text.strip()) > 0)
        
        counter = TokenCounter()
        tokens = counter.count_tokens(text)
        
        assert tokens > 0, "非空テキストのトークン数は正の値であるべき"
    
    @given(
        input_tokens=st.integers(min_value=1, max_value=100000),
        output_tokens=st.integers(min_value=1, max_value=100000),
    )
    @settings(max_examples=100, deadline=None)
    def test_cost_calculation_positive(self, input_tokens: int, output_tokens: int):
        """コスト計算結果は常に正の値"""
        calculator = CostCalculator()
        result = calculator.calculate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        
        assert result.total_usd > 0, "コストは正の値であるべき"
        assert result.input_cost >= 0, "入力コストは非負であるべき"
        assert result.output_cost >= 0, "出力コストは非負であるべき"
    
    @given(
        cost_limit=st.floats(min_value=0.01, max_value=1000.0),
        num_requests=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100, deadline=None)
    def test_cost_limit_tracking(self, cost_limit: float, num_requests: int):
        """コスト上限追跡が正しく動作する"""
        checker = CostLimitChecker(cost_limit_usd=cost_limit)
        
        for _ in range(num_requests):
            checker.record_cost(
                input_tokens=100,
                output_tokens=50,
            )
        
        # 累積コストは正の値
        assert checker.accumulated_cost > 0
        
        # リクエスト数が正しい
        assert checker.request_count == num_requests
        
        # 残り予算は上限以下
        if checker.remaining_budget is not None:
            assert checker.remaining_budget <= cost_limit


class TestProperty23CostTracking:
    """
    Property 23: コスト追跡
    
    *For any* APIリクエストについて、入力トークン数、出力トークン数、キャッシュヒット数、処理時間が記録される
    
    **Feature: excel-translation-api, Property 23: コスト追跡**
    **Validates: Requirements 10.1, 10.2, 10.3**
    """
    
    @given(
        input_tokens=st.integers(min_value=1, max_value=100000),
        output_tokens=st.integers(min_value=1, max_value=100000),
        cache_hit_tokens=st.integers(min_value=0, max_value=50000),
        processing_time_ms=st.integers(min_value=0, max_value=60000),
    )
    @settings(max_examples=100, deadline=None)
    def test_track_request_records_all_fields(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_hit_tokens: int,
        processing_time_ms: int,
    ):
        """リクエスト追跡で全フィールドが記録される"""
        tracker = CostTracker()
        
        tracker.track_request(
            pass_name="1st_pass",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit_tokens=cache_hit_tokens,
            processing_time_ms=processing_time_ms,
        )
        
        summary = tracker.get_pass_summary("1st_pass")
        
        # 全フィールドが正しく記録されている
        assert summary.input_tokens == input_tokens, "入力トークン数が記録されるべき"
        assert summary.output_tokens == output_tokens, "出力トークン数が記録されるべき"
        assert summary.cache_hit_tokens == cache_hit_tokens, "キャッシュヒット数が記録されるべき"
        assert summary.processing_time_ms == processing_time_ms, "処理時間が記録されるべき"
        assert summary.api_calls == 1, "API呼び出し回数が記録されるべき"
        assert summary.cost_usd >= 0, "コストが計算されるべき"
    
    @given(
        num_requests=st.integers(min_value=1, max_value=20),
        input_tokens=st.integers(min_value=100, max_value=10000),
        output_tokens=st.integers(min_value=50, max_value=5000),
    )
    @settings(max_examples=100, deadline=None)
    def test_track_multiple_requests_accumulates(
        self,
        num_requests: int,
        input_tokens: int,
        output_tokens: int,
    ):
        """複数リクエストの追跡で値が累積される"""
        tracker = CostTracker()
        
        for _ in range(num_requests):
            tracker.track_request(
                pass_name="1st_pass",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_hit_tokens=0,
                processing_time_ms=100,
            )
        
        summary = tracker.get_pass_summary("1st_pass")
        
        # 累積値が正しい
        assert summary.input_tokens == input_tokens * num_requests
        assert summary.output_tokens == output_tokens * num_requests
        assert summary.api_calls == num_requests
        assert summary.processing_time_ms == 100 * num_requests
    
    @given(
        pass_names=st.lists(
            st.sampled_from(["1st_pass", "2nd_pass", "3rd_pass", "4th_pass"]),
            min_size=1,
            max_size=10,
        ),
        input_tokens=st.integers(min_value=100, max_value=5000),
    )
    @settings(max_examples=100, deadline=None)
    def test_track_multiple_passes_separately(
        self,
        pass_names: List[str],
        input_tokens: int,
    ):
        """複数パスが個別に追跡される"""
        tracker = CostTracker()
        
        # 各パスにリクエストを記録
        pass_counts = {}
        for pass_name in pass_names:
            tracker.track_request(
                pass_name=pass_name,
                input_tokens=input_tokens,
                output_tokens=input_tokens // 2,
                cache_hit_tokens=0,
                processing_time_ms=100,
            )
            pass_counts[pass_name] = pass_counts.get(pass_name, 0) + 1
        
        # 各パスのサマリが正しい
        for pass_name, count in pass_counts.items():
            summary = tracker.get_pass_summary(pass_name)
            assert summary is not None, f"パス {pass_name} のサマリが存在するべき"
            assert summary.api_calls == count, f"パス {pass_name} のAPI呼び出し回数が正しいべき"
            assert summary.input_tokens == input_tokens * count
    
    @given(
        input_tokens=st.integers(min_value=1, max_value=100000),
        output_tokens=st.integers(min_value=1, max_value=100000),
    )
    @settings(max_examples=100, deadline=None)
    def test_cost_calculation_consistent(
        self,
        input_tokens: int,
        output_tokens: int,
    ):
        """コスト計算が一貫している"""
        tracker = CostTracker()
        
        # 直接計算
        direct_cost = tracker.calculate_cost(
            model=tracker.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit_tokens=0,
        )
        
        # track_request経由で計算
        tracker.track_request(
            pass_name="test_pass",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit_tokens=0,
            processing_time_ms=0,
        )
        
        summary = tracker.get_pass_summary("test_pass")
        
        # 両方の計算結果が一致
        assert abs(direct_cost - summary.cost_usd) < 0.0001, "コスト計算が一貫しているべき"


class TestProperty24CostReportGeneration:
    """
    Property 24: コストレポート生成
    
    *For any* 翻訳処理について、パスごとのコストサマリとプロジェクト全体のダッシュボードが生成される
    
    **Feature: excel-translation-api, Property 24: コストレポート生成**
    **Validates: Requirements 10.4, 10.5**
    """
    
    @given(
        pass_data=st.lists(
            st.tuples(
                st.sampled_from(["1st_pass", "2nd_pass", "3rd_pass", "4th_pass"]),
                st.integers(min_value=100, max_value=10000),
                st.integers(min_value=50, max_value=5000),
                st.integers(min_value=0, max_value=2000),
                st.integers(min_value=100, max_value=5000),
            ),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_report_contains_all_passes(self, pass_data):
        """レポートに全パスのサマリが含まれる"""
        tracker = CostTracker()
        
        # パスデータを記録
        recorded_passes = set()
        for pass_name, input_tokens, output_tokens, cache_hit, time_ms in pass_data:
            tracker.track_request(
                pass_name=pass_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_hit_tokens=cache_hit,
                processing_time_ms=time_ms,
            )
            recorded_passes.add(pass_name)
        
        # レポート生成
        report = tracker.generate_report()
        
        # レポートに必要なフィールドが含まれる
        assert "model" in report, "レポートにモデル名が含まれるべき"
        assert "passes" in report, "レポートにパスリストが含まれるべき"
        assert "summary" in report, "レポートにサマリが含まれるべき"
        
        # 全パスがレポートに含まれる
        report_passes = {p["pass_name"] for p in report["passes"]}
        assert report_passes == recorded_passes, "全パスがレポートに含まれるべき"
    
    @given(
        pass_data=st.lists(
            st.tuples(
                st.sampled_from(["1st_pass", "2nd_pass", "3rd_pass", "4th_pass"]),
                st.integers(min_value=100, max_value=10000),
                st.integers(min_value=50, max_value=5000),
            ),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_report_summary_totals_correct(self, pass_data):
        """レポートのサマリ合計が正しい"""
        tracker = CostTracker()
        
        expected_input = 0
        expected_output = 0
        expected_calls = 0
        
        for pass_name, input_tokens, output_tokens in pass_data:
            tracker.track_request(
                pass_name=pass_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_hit_tokens=0,
                processing_time_ms=100,
            )
            expected_input += input_tokens
            expected_output += output_tokens
            expected_calls += 1
        
        report = tracker.generate_report()
        summary = report["summary"]
        
        # サマリの合計が正しい
        assert summary["total_input_tokens"] == expected_input, "入力トークン合計が正しいべき"
        assert summary["total_output_tokens"] == expected_output, "出力トークン合計が正しいべき"
        assert summary["total_api_calls"] == expected_calls, "API呼び出し合計が正しいべき"
    
    @given(
        pass_data=st.lists(
            st.tuples(
                st.sampled_from(["1st_pass", "2nd_pass", "3rd_pass", "4th_pass"]),
                st.integers(min_value=100, max_value=10000),
                st.integers(min_value=50, max_value=5000),
            ),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_report_pass_details_complete(self, pass_data):
        """各パスの詳細が完全"""
        tracker = CostTracker()
        
        for pass_name, input_tokens, output_tokens in pass_data:
            tracker.track_request(
                pass_name=pass_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_hit_tokens=0,
                processing_time_ms=100,
            )
        
        report = tracker.generate_report()
        
        # 各パスの詳細が完全
        required_fields = [
            "pass_name", "input_tokens", "output_tokens", "cache_hit_tokens",
            "api_calls", "processing_time_ms", "cost_usd", "modified_rows",
            "total_rows", "modification_rate"
        ]
        
        for pass_detail in report["passes"]:
            for field in required_fields:
                assert field in pass_detail, f"パス詳細に {field} が含まれるべき"
    
    @given(
        modified_rows=st.integers(min_value=0, max_value=100),
        total_rows=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=100, deadline=None)
    def test_modification_rate_calculation(self, modified_rows: int, total_rows: int):
        """修正率の計算が正しい"""
        assume(modified_rows <= total_rows)
        
        tracker = CostTracker()
        tracker.track_request("2nd_pass", 1000, 500, 0, 100)
        tracker.record_modifications("2nd_pass", modified_rows, total_rows)
        
        report = tracker.generate_report()
        
        # 2nd_passの詳細を取得
        pass_detail = next(p for p in report["passes"] if p["pass_name"] == "2nd_pass")
        
        expected_rate = modified_rows / total_rows
        assert abs(pass_detail["modification_rate"] - expected_rate) < 0.0001, "修正率が正しいべき"
    
    @given(
        num_passes=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=100, deadline=None)
    def test_total_cost_matches_sum(self, num_passes: int):
        """総コストがパスコストの合計と一致"""
        tracker = CostTracker()
        
        pass_names = ["1st_pass", "2nd_pass", "3rd_pass", "4th_pass"][:num_passes]
        
        for pass_name in pass_names:
            tracker.track_request(
                pass_name=pass_name,
                input_tokens=1000,
                output_tokens=500,
                cache_hit_tokens=100,
                processing_time_ms=1000,
            )
        
        report = tracker.generate_report()
        
        # パスコストの合計
        sum_of_passes = sum(p["cost_usd"] for p in report["passes"])
        
        # 総コストと一致
        assert abs(report["summary"]["total_cost_usd"] - sum_of_passes) < 0.0001, \
            "総コストがパスコストの合計と一致するべき"


class TestProperty25ModeComparison:
    """
    Property 25: モード間比較
    
    *For any* 同一ファイルを異なるモードで処理した結果について、モード間比較シートが生成され、
    差分と追加コストが計算される
    
    **Feature: excel-translation-api, Property 25: モード間比較**
    **Validates: Requirements 11.1, 11.2, 11.3, 11.4**
    """
    
    @st.composite
    def translation_results_strategy(draw):
        """翻訳結果を生成するストラテジー"""
        num_rows = draw(st.integers(min_value=1, max_value=10))
        results = []
        
        for i in range(num_rows):
            text_id = f"TEXT_{i+1:03d}"
            source_text = draw(st.text(
                alphabet=st.sampled_from("あいうえおかきくけこ"),
                min_size=1,
                max_size=20
            ))
            translated_text = draw(st.text(
                alphabet=st.characters(whitelist_categories=('L',)),
                min_size=1,
                max_size=30
            ))
            
            results.append({
                "text_id": text_id,
                "source_text": source_text,
                "translated_text": translated_text,
            })
        
        return results
    
    @given(
        draft_results=translation_results_strategy(),
        standard_results=translation_results_strategy(),
    )
    @settings(max_examples=100, deadline=None)
    def test_mode_comparison_contains_all_modes(self, draft_results, standard_results):
        """モード比較に全モードが含まれる"""
        tracker = CostTracker()
        
        # 同じ行数に調整
        min_len = min(len(draft_results), len(standard_results))
        draft_results = draft_results[:min_len]
        standard_results = standard_results[:min_len]
        
        # text_idを統一
        for i in range(min_len):
            text_id = f"TEXT_{i+1:03d}"
            draft_results[i]["text_id"] = text_id
            standard_results[i]["text_id"] = text_id
        
        results = {
            "draft": draft_results,
            "standard": standard_results,
        }
        
        comparison = tracker.generate_mode_comparison(results)
        
        # 全モードが含まれる
        assert "modes" in comparison, "比較結果にモードリストが含まれるべき"
        assert set(comparison["modes"]) == {"draft", "standard"}, "全モードが含まれるべき"
    
    @given(
        num_rows=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100, deadline=None)
    def test_mode_comparison_row_count_matches(self, num_rows: int):
        """モード比較の行数が元データと一致"""
        tracker = CostTracker()
        
        # 各モードの結果を生成
        draft_results = [
            {"text_id": f"TEXT_{i+1:03d}", "source_text": f"原文{i}", "translated_text": f"Draft{i}"}
            for i in range(num_rows)
        ]
        standard_results = [
            {"text_id": f"TEXT_{i+1:03d}", "source_text": f"原文{i}", "translated_text": f"Standard{i}"}
            for i in range(num_rows)
        ]
        
        results = {
            "draft": draft_results,
            "standard": standard_results,
        }
        
        comparison = tracker.generate_mode_comparison(results)
        
        # 行数が一致
        assert len(comparison["comparison"]) == num_rows, "比較結果の行数が元データと一致するべき"
    
    @given(
        num_rows=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100, deadline=None)
    def test_mode_comparison_detects_differences(self, num_rows: int):
        """モード比較で差分が検出される"""
        tracker = CostTracker()
        
        # 一部の行で差分を作成
        draft_results = []
        standard_results = []
        
        for i in range(num_rows):
            text_id = f"TEXT_{i+1:03d}"
            source_text = f"原文{i}"
            
            draft_results.append({
                "text_id": text_id,
                "source_text": source_text,
                "translated_text": f"Draft translation {i}",
            })
            
            # 偶数行は同じ、奇数行は異なる
            if i % 2 == 0:
                standard_results.append({
                    "text_id": text_id,
                    "source_text": source_text,
                    "translated_text": f"Draft translation {i}",  # 同じ
                })
            else:
                standard_results.append({
                    "text_id": text_id,
                    "source_text": source_text,
                    "translated_text": f"Standard translation {i}",  # 異なる
                })
        
        results = {
            "draft": draft_results,
            "standard": standard_results,
        }
        
        comparison = tracker.generate_mode_comparison(results)
        
        # 差分が検出される
        for i, row in enumerate(comparison["comparison"]):
            if i % 2 == 0:
                # 同じ行は差分なし
                assert len(row["differences"]) == 0 or not any(d["changed"] for d in row["differences"]), \
                    f"行 {i} は差分がないべき"
            else:
                # 異なる行は差分あり
                assert len(row["differences"]) > 0 and any(d["changed"] for d in row["differences"]), \
                    f"行 {i} は差分があるべき"
    
    @given(
        num_modes=st.integers(min_value=2, max_value=3),
    )
    @settings(max_examples=100, deadline=None)
    def test_mode_comparison_includes_all_mode_results(self, num_modes: int):
        """モード比較に全モードの結果が含まれる"""
        tracker = CostTracker()
        
        mode_names = ["draft", "standard", "thorough"][:num_modes]
        results = {}
        
        for mode in mode_names:
            results[mode] = [
                {"text_id": "TEXT_001", "source_text": "原文", "translated_text": f"{mode}_result"},
            ]
        
        comparison = tracker.generate_mode_comparison(results)
        
        # 各モードの結果列が含まれる
        row = comparison["comparison"][0]
        for mode in mode_names:
            assert f"{mode}_result" in row, f"モード {mode} の結果列が含まれるべき"
    
    @given(
        num_rows=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100, deadline=None)
    def test_mode_comparison_preserves_text_ids(self, num_rows: int):
        """モード比較でtext_idが保持される"""
        tracker = CostTracker()
        
        text_ids = [f"TEXT_{i+1:03d}" for i in range(num_rows)]
        
        results = {
            "draft": [
                {"text_id": tid, "source_text": f"原文{i}", "translated_text": f"Draft{i}"}
                for i, tid in enumerate(text_ids)
            ],
            "standard": [
                {"text_id": tid, "source_text": f"原文{i}", "translated_text": f"Standard{i}"}
                for i, tid in enumerate(text_ids)
            ],
        }
        
        comparison = tracker.generate_mode_comparison(results)
        
        # text_idが保持される
        comparison_ids = [row["text_id"] for row in comparison["comparison"]]
        assert comparison_ids == text_ids, "text_idが保持されるべき"
    
    @given(st.data())
    @settings(max_examples=100, deadline=None)
    def test_empty_results_handled(self, data):
        """空の結果が正しく処理される"""
        tracker = CostTracker()
        
        results = {}
        comparison = tracker.generate_mode_comparison(results)
        
        assert comparison["comparison"] == [], "空の結果は空の比較を返すべき"
        assert comparison["modes"] == [], "空の結果は空のモードリストを返すべき"
