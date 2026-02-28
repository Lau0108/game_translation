"""コスト追跡・トークン計測モジュール

tiktokenを使用したトークン数計測、コスト推定、上限チェック、
Prompt Caching対応を提供する。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import tiktoken

logger = logging.getLogger(__name__)


# モデル別料金設定（USD per 1M tokens）
MODEL_PRICING = {
    "claude-3-5-sonnet-latest": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-3-5-haiku-20241022": {
        "input": 0.80,
        "output": 4.00,
        "cache_write": 1.00,
        "cache_read": 0.08,
    },
    "claude-3-7-sonnet-20250219": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-3-5-sonnet-20241022": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,  # 25% premium for cache write
        "cache_read": 0.30,   # 90% discount for cache read
    },
    "claude-3-opus-20240229": {
        "input": 15.00,
        "output": 75.00,
        "cache_write": 18.75,
        "cache_read": 1.50,
    },
    "claude-3-haiku-20240307": {
        "input": 0.25,
        "output": 1.25,
        "cache_write": 0.30,
        "cache_read": 0.03,
    },
    "deepseek-chat": {
        "input": 0.14,
        "output": 0.28,
        "cache_read": 0.014,
    },
    "gpt-4-turbo": {
        "input": 10.00,
        "output": 30.00,
    },
    "gpt-4o": {
        "input": 2.50,
        "output": 10.00,
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
    },
    "gemini-1.5-pro": {
        "input": 1.25,
        "output": 5.00,
    },
    "gemini-1.5-flash": {
        "input": 0.075,
        "output": 0.30,
    },
    # Gemini 2.5 models (2026 pricing)
    "gemini-2.5-pro": {
        "input": 1.25,      # ≤200K tokens
        "output": 10.00,    # includes thinking tokens
        "cache_read": 0.125,
    },
    "gemini-2.5-flash": {
        "input": 0.30,
        "output": 2.50,     # includes thinking tokens
        "cache_read": 0.03,
    },
    "gemini-2.5-flash-lite": {
        "input": 0.10,
        "output": 0.40,
        "cache_read": 0.01,
    },
    # Gemini 3 models (2026 pricing)
    "gemini-3-flash-preview": {
        "input": 0.50,
        "output": 3.00,     # includes thinking tokens
        "cache_read": 0.05,
    },
    "gemini-3-pro-preview": {
        "input": 2.00,      # ≤200K tokens
        "output": 12.00,    # includes thinking tokens
        "cache_read": 0.20,
    },
}

# デフォルトモデル
DEFAULT_MODEL = "claude-3-5-sonnet-20241022"


class TokenCounterError(Exception):
    """トークン計測エラー"""
    pass


class CostLimitExceededError(Exception):
    """コスト上限超過エラー"""
    pass


@dataclass
class TokenCount:
    """トークン数情報"""
    total: int
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass
class CostEstimate:
    """コスト推定情報"""
    total_usd: float
    input_cost: float = 0.0
    output_cost: float = 0.0
    cache_write_cost: float = 0.0
    cache_read_cost: float = 0.0
    model: str = DEFAULT_MODEL


class TokenCounter:
    """
    トークン計測クラス
    
    tiktokenを使用して正確なトークン数を計測する。
    Claude/GPT/Geminiの各モデルに対応。
    """
    
    # モデル別エンコーディングマッピング
    MODEL_ENCODING_MAP = {
        "claude": "cl100k_base",  # Claude uses similar tokenization to GPT-4
        "gpt-4": "cl100k_base",
        "gpt-4o": "o200k_base",
        "gpt-3.5": "cl100k_base",
        "gemini": "cl100k_base",  # Approximate with cl100k_base
    }
    
    def __init__(self, model: str = DEFAULT_MODEL):
        """
        TokenCounterを初期化
        
        Args:
            model: 使用するモデル名
        """
        self.model = model
        self._encoding = self._get_encoding(model)
    
    def _get_encoding(self, model: str) -> tiktoken.Encoding:
        """モデルに対応するエンコーディングを取得"""
        # モデル名からエンコーディングを決定
        model_lower = model.lower()
        
        if "gpt-4o" in model_lower:
            encoding_name = "o200k_base"
        elif "gpt-4" in model_lower or "gpt-3.5" in model_lower:
            encoding_name = "cl100k_base"
        elif "claude" in model_lower:
            encoding_name = "cl100k_base"
        elif "gemini" in model_lower:
            encoding_name = "cl100k_base"
        else:
            # デフォルトはcl100k_base
            encoding_name = "cl100k_base"
        
        try:
            return tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.warning(f"エンコーディング取得失敗: {e}, デフォルトを使用")
            return tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """
        テキストのトークン数を計測
        
        Args:
            text: 計測対象テキスト
            
        Returns:
            トークン数
        """
        if not text:
            return 0
        
        try:
            tokens = self._encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logger.warning(f"トークン計測エラー: {e}, 推定値を使用")
            return self._estimate_tokens(text)
    
    def _estimate_tokens(self, text: str) -> int:
        """
        トークン数を推定（フォールバック用）
        
        日本語: 約1.5文字/トークン
        英語: 約4文字/トークン
        """
        if not text:
            return 0
        
        jp_chars = sum(1 for c in text if '\u3000' <= c <= '\u9fff' or '\u30a0' <= c <= '\u30ff')
        other_chars = len(text) - jp_chars
        
        jp_tokens = jp_chars / 1.5
        other_tokens = other_chars / 4
        
        return int(jp_tokens + other_tokens) + 1
    
    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        メッセージリストのトークン数を計測
        
        Args:
            messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]
            
        Returns:
            合計トークン数
        """
        total = 0
        
        for message in messages:
            # メッセージごとのオーバーヘッド（約4トークン）
            total += 4
            
            role = message.get("role", "")
            content = message.get("content", "")
            
            total += self.count_tokens(role)
            total += self.count_tokens(content)
        
        # 全体のオーバーヘッド（約3トークン）
        total += 3
        
        return total
    
    def count_prompt_tokens(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> TokenCount:
        """
        プロンプトのトークン数を計測
        
        Args:
            system_prompt: システムプロンプト
            user_prompt: ユーザープロンプト
            
        Returns:
            TokenCount情報
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        total = self.count_messages_tokens(messages)
        
        return TokenCount(
            total=total,
            input_tokens=total,
        )


class CostCalculator:
    """
    コスト計算クラス
    
    トークン数からAPI利用料金を計算する。
    Prompt Caching対応。
    """
    
    def __init__(self, model: str = DEFAULT_MODEL):
        """
        CostCalculatorを初期化
        
        Args:
            model: 使用するモデル名
        """
        self.model = model
        self._pricing = self._get_pricing(model)
    
    def _get_pricing(self, model: str) -> Dict[str, float]:
        """モデルの料金設定を取得"""
        # 完全一致を試行
        if model in MODEL_PRICING:
            return MODEL_PRICING[model]
        
        # 部分一致を試行
        model_lower = model.lower()
        for key, pricing in MODEL_PRICING.items():
            if key.lower() in model_lower or model_lower in key.lower():
                return pricing
        
        # デフォルト料金
        logger.warning(f"モデル '{model}' の料金設定が見つかりません。デフォルトを使用")
        return MODEL_PRICING[DEFAULT_MODEL]
    
    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_write_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> CostEstimate:
        """
        コストを計算
        
        Args:
            input_tokens: 入力トークン数（キャッシュ対象外）
            output_tokens: 出力トークン数
            cache_write_tokens: キャッシュ書き込みトークン数
            cache_read_tokens: キャッシュ読み込みトークン数
            
        Returns:
            CostEstimate情報
        """
        pricing = self._pricing
        
        # 料金計算（per 1M tokens → per token）
        input_cost = (input_tokens / 1_000_000) * pricing.get("input", 0)
        output_cost = (output_tokens / 1_000_000) * pricing.get("output", 0)
        cache_write_cost = (cache_write_tokens / 1_000_000) * pricing.get("cache_write", pricing.get("input", 0))
        cache_read_cost = (cache_read_tokens / 1_000_000) * pricing.get("cache_read", pricing.get("input", 0) * 0.1)
        
        total = input_cost + output_cost + cache_write_cost + cache_read_cost
        
        return CostEstimate(
            total_usd=total,
            input_cost=input_cost,
            output_cost=output_cost,
            cache_write_cost=cache_write_cost,
            cache_read_cost=cache_read_cost,
            model=self.model,
        )
    
    def estimate_translation_cost(
        self,
        input_tokens: int,
        estimated_output_ratio: float = 1.2,
        use_cache: bool = True,
        cache_hit_ratio: float = 0.7,
    ) -> CostEstimate:
        """
        翻訳コストを推定
        
        Args:
            input_tokens: 入力トークン数
            estimated_output_ratio: 出力/入力比率（デフォルト1.2）
            use_cache: Prompt Cachingを使用するか
            cache_hit_ratio: キャッシュヒット率（デフォルト0.7）
            
        Returns:
            CostEstimate情報
        """
        estimated_output = int(input_tokens * estimated_output_ratio)
        
        if use_cache:
            cache_read = int(input_tokens * cache_hit_ratio)
            cache_write = int(input_tokens * (1 - cache_hit_ratio) * 0.5)  # 初回のみ書き込み
            regular_input = input_tokens - cache_read - cache_write
        else:
            cache_read = 0
            cache_write = 0
            regular_input = input_tokens
        
        return self.calculate_cost(
            input_tokens=regular_input,
            output_tokens=estimated_output,
            cache_write_tokens=cache_write,
            cache_read_tokens=cache_read,
        )




class ChunkResplitter:
    """
    チャンク再分割クラス
    
    トークン上限を超えるチャンクを自動的に再分割する。
    """
    
    def __init__(
        self,
        max_tokens: int = 2000,
        model: str = DEFAULT_MODEL,
    ):
        """
        ChunkResplitterを初期化
        
        Args:
            max_tokens: チャンクあたりの最大トークン数
            model: トークン計測に使用するモデル
        """
        self.max_tokens = max_tokens
        self.token_counter = TokenCounter(model)
    
    def needs_resplit(self, prompt: str) -> bool:
        """
        プロンプトが再分割を必要とするか判定
        
        Args:
            prompt: チェック対象のプロンプト
            
        Returns:
            再分割が必要な場合True
        """
        token_count = self.token_counter.count_tokens(prompt)
        return token_count > self.max_tokens
    
    def calculate_split_count(self, prompt: str) -> int:
        """
        必要な分割数を計算
        
        Args:
            prompt: 分割対象のプロンプト
            
        Returns:
            必要な分割数
        """
        token_count = self.token_counter.count_tokens(prompt)
        if token_count <= self.max_tokens:
            return 1
        
        # 余裕を持って分割（90%を目標）
        target_tokens = int(self.max_tokens * 0.9)
        split_count = (token_count + target_tokens - 1) // target_tokens
        
        return max(1, split_count)
    
    def resplit_texts(
        self,
        texts: List[Dict[str, Any]],
        base_prompt_tokens: int = 0,
    ) -> List[List[Dict[str, Any]]]:
        """
        テキストリストを再分割
        
        Args:
            texts: 分割対象のテキストリスト
                   [{"text_id": str, "source_text": str, ...}, ...]
            base_prompt_tokens: ベースプロンプト（システムプロンプト等）のトークン数
            
        Returns:
            分割されたテキストリストのリスト
        """
        if not texts:
            return []
        
        # 利用可能なトークン数
        available_tokens = self.max_tokens - base_prompt_tokens
        if available_tokens <= 0:
            raise TokenCounterError(
                f"ベースプロンプトがトークン上限を超えています: "
                f"{base_prompt_tokens} > {self.max_tokens}"
            )
        
        chunks: List[List[Dict[str, Any]]] = []
        current_chunk: List[Dict[str, Any]] = []
        current_tokens = 0
        
        for text in texts:
            # テキストのトークン数を計測
            text_content = text.get("source_text", "")
            text_tokens = self.token_counter.count_tokens(text_content)
            
            # text_id, character等のオーバーヘッドを追加
            overhead = 20  # JSON構造のオーバーヘッド
            text_tokens += overhead
            
            # 単一テキストが上限を超える場合は警告
            if text_tokens > available_tokens:
                logger.warning(
                    f"単一テキストがトークン上限を超えています: "
                    f"text_id={text.get('text_id')}, tokens={text_tokens}"
                )
                # それでも追加（後続処理で対応）
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_tokens = 0
                chunks.append([text])
                continue
            
            # 現在のチャンクに追加可能か判定
            if current_tokens + text_tokens > available_tokens:
                # 新しいチャンクを開始
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [text]
                current_tokens = text_tokens
            else:
                current_chunk.append(text)
                current_tokens += text_tokens
        
        # 最後のチャンクを追加
        if current_chunk:
            chunks.append(current_chunk)
        
        logger.info(f"テキストを {len(chunks)} チャンクに再分割しました")
        return chunks
    
    def get_token_count(self, text: str) -> int:
        """
        テキストのトークン数を取得
        
        Args:
            text: 計測対象テキスト
            
        Returns:
            トークン数
        """
        return self.token_counter.count_tokens(text)


class CostLimitChecker:
    """
    コスト上限チェッカー
    
    累積コストを追跡し、上限超過時に警告・停止する。
    """
    
    def __init__(
        self,
        cost_limit_usd: Optional[float] = None,
        model: str = DEFAULT_MODEL,
    ):
        """
        CostLimitCheckerを初期化
        
        Args:
            cost_limit_usd: コスト上限（USD）、Noneの場合は無制限
            model: コスト計算に使用するモデル
        """
        self.cost_limit_usd = cost_limit_usd
        self.calculator = CostCalculator(model)
        self._accumulated_cost = 0.0
        self._request_count = 0
    
    @property
    def accumulated_cost(self) -> float:
        """累積コストを取得"""
        return self._accumulated_cost
    
    @property
    def request_count(self) -> int:
        """リクエスト数を取得"""
        return self._request_count
    
    @property
    def remaining_budget(self) -> Optional[float]:
        """残り予算を取得"""
        if self.cost_limit_usd is None:
            return None
        return max(0, self.cost_limit_usd - self._accumulated_cost)
    
    def check_limit(self, estimated_cost: float) -> Tuple[bool, str]:
        """
        コスト上限をチェック
        
        Args:
            estimated_cost: 推定コスト（USD）
            
        Returns:
            (続行可能か, メッセージ)
        """
        if self.cost_limit_usd is None:
            return True, "コスト上限なし"
        
        projected_total = self._accumulated_cost + estimated_cost
        
        if projected_total > self.cost_limit_usd:
            return False, (
                f"コスト上限に達します: "
                f"累積=${self._accumulated_cost:.4f}, "
                f"推定追加=${estimated_cost:.4f}, "
                f"上限=${self.cost_limit_usd:.4f}"
            )
        
        # 80%警告
        if projected_total > self.cost_limit_usd * 0.8:
            return True, (
                f"コスト上限の80%を超えます: "
                f"予測合計=${projected_total:.4f} / "
                f"上限=${self.cost_limit_usd:.4f}"
            )
        
        return True, "OK"
    
    def record_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_write_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> CostEstimate:
        """
        コストを記録
        
        Args:
            input_tokens: 入力トークン数
            output_tokens: 出力トークン数
            cache_write_tokens: キャッシュ書き込みトークン数
            cache_read_tokens: キャッシュ読み込みトークン数
            
        Returns:
            CostEstimate情報
        """
        cost = self.calculator.calculate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_read_tokens=cache_read_tokens,
        )
        
        self._accumulated_cost += cost.total_usd
        self._request_count += 1
        
        return cost
    
    def estimate_and_check(
        self,
        input_tokens: int,
        estimated_output_ratio: float = 1.2,
        use_cache: bool = True,
    ) -> Tuple[CostEstimate, bool, str]:
        """
        コストを推定してチェック
        
        Args:
            input_tokens: 入力トークン数
            estimated_output_ratio: 出力/入力比率
            use_cache: Prompt Cachingを使用するか
            
        Returns:
            (CostEstimate, 続行可能か, メッセージ)
        """
        estimate = self.calculator.estimate_translation_cost(
            input_tokens=input_tokens,
            estimated_output_ratio=estimated_output_ratio,
            use_cache=use_cache,
        )
        
        can_continue, message = self.check_limit(estimate.total_usd)
        
        return estimate, can_continue, message
    
    def reset(self) -> None:
        """累積コストをリセット"""
        self._accumulated_cost = 0.0
        self._request_count = 0
    
    def get_summary(self) -> Dict[str, Any]:
        """サマリを取得"""
        return {
            "accumulated_cost_usd": self._accumulated_cost,
            "request_count": self._request_count,
            "cost_limit_usd": self.cost_limit_usd,
            "remaining_budget_usd": self.remaining_budget,
        }


@dataclass
class PassCost:
    """パスごとのコスト情報"""
    pass_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_tokens: int = 0
    api_calls: int = 0
    processing_time_ms: int = 0
    cost_usd: float = 0.0
    modified_rows: int = 0
    total_rows: int = 0
    
    @property
    def modification_rate(self) -> float:
        """修正率を計算"""
        if self.total_rows == 0:
            return 0.0
        return self.modified_rows / self.total_rows


class CostTracker:
    """
    コスト追跡クラス
    
    各パスのトークン消費量・料金を追跡し、コストレポートを生成する。
    Requirements 10.1-10.5, 11.1-11.4 を満たす。
    """
    
    def __init__(self, model: str = DEFAULT_MODEL):
        """
        CostTrackerを初期化
        
        Args:
            model: 使用するモデル名
        """
        self.model = model
        self.calculator = CostCalculator(model)
        self._pass_data: Dict[str, PassCost] = {}
        self._mode_results: Dict[str, Dict[str, Any]] = {}
    
    def track_request(
        self,
        pass_name: str,
        input_tokens: int,
        output_tokens: int,
        cache_hit_tokens: int = 0,
        processing_time_ms: int = 0,
    ) -> None:
        """
        リクエストのトークン消費を記録
        
        Args:
            pass_name: パス名（"1st_pass", "2nd_pass", "3rd_pass", "4th_pass"）
            input_tokens: 入力トークン数
            output_tokens: 出力トークン数
            cache_hit_tokens: キャッシュヒットトークン数
            processing_time_ms: 処理時間（ミリ秒）
        """
        if pass_name not in self._pass_data:
            self._pass_data[pass_name] = PassCost(pass_name=pass_name)
        
        pass_cost = self._pass_data[pass_name]
        pass_cost.input_tokens += input_tokens
        pass_cost.output_tokens += output_tokens
        pass_cost.cache_hit_tokens += cache_hit_tokens
        pass_cost.api_calls += 1
        pass_cost.processing_time_ms += processing_time_ms
        
        # コストを計算して加算
        cost = self.calculator.calculate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_hit_tokens,
        )
        pass_cost.cost_usd += cost.total_usd
    
    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_hit_tokens: int = 0,
    ) -> float:
        """
        料金を計算
        
        Args:
            model: モデル名
            input_tokens: 入力トークン数
            output_tokens: 出力トークン数
            cache_hit_tokens: キャッシュヒットトークン数
            
        Returns:
            料金（USD）
        """
        calculator = CostCalculator(model)
        cost = calculator.calculate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_hit_tokens,
        )
        return cost.total_usd
    
    def record_modifications(
        self,
        pass_name: str,
        modified_rows: int,
        total_rows: int,
    ) -> None:
        """
        修正行数を記録
        
        Args:
            pass_name: パス名
            modified_rows: 修正行数
            total_rows: 総行数
        """
        if pass_name not in self._pass_data:
            self._pass_data[pass_name] = PassCost(pass_name=pass_name)
        
        pass_cost = self._pass_data[pass_name]
        pass_cost.modified_rows += modified_rows
        pass_cost.total_rows = total_rows
    
    def get_pass_summary(self, pass_name: str) -> Optional[PassCost]:
        """
        パスごとのサマリを取得
        
        Args:
            pass_name: パス名
            
        Returns:
            PassCost情報、存在しない場合はNone
        """
        return self._pass_data.get(pass_name)
    
    def get_all_passes(self) -> List[str]:
        """
        記録されている全パス名を取得
        
        Returns:
            パス名のリスト
        """
        return list(self._pass_data.keys())
    
    def generate_report(self) -> Dict[str, Any]:
        """
        コストレポートを生成
        
        Returns:
            コストレポート辞書
        """
        passes = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_cache_hit_tokens = 0
        total_api_calls = 0
        total_processing_time_ms = 0
        total_cost_usd = 0.0
        total_modified_rows = 0
        total_rows = 0
        
        for pass_name, pass_cost in self._pass_data.items():
            passes.append({
                "pass_name": pass_cost.pass_name,
                "input_tokens": pass_cost.input_tokens,
                "output_tokens": pass_cost.output_tokens,
                "cache_hit_tokens": pass_cost.cache_hit_tokens,
                "api_calls": pass_cost.api_calls,
                "processing_time_ms": pass_cost.processing_time_ms,
                "cost_usd": pass_cost.cost_usd,
                "modified_rows": pass_cost.modified_rows,
                "total_rows": pass_cost.total_rows,
                "modification_rate": pass_cost.modification_rate,
            })
            
            total_input_tokens += pass_cost.input_tokens
            total_output_tokens += pass_cost.output_tokens
            total_cache_hit_tokens += pass_cost.cache_hit_tokens
            total_api_calls += pass_cost.api_calls
            total_processing_time_ms += pass_cost.processing_time_ms
            total_cost_usd += pass_cost.cost_usd
            total_modified_rows += pass_cost.modified_rows
            if pass_cost.total_rows > total_rows:
                total_rows = pass_cost.total_rows
        
        return {
            "model": self.model,
            "passes": passes,
            "summary": {
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_cache_hit_tokens": total_cache_hit_tokens,
                "total_api_calls": total_api_calls,
                "total_processing_time_ms": total_processing_time_ms,
                "total_cost_usd": total_cost_usd,
                "total_modified_rows": total_modified_rows,
                "total_rows": total_rows,
            },
        }
    
    def cache_mode_result(
        self,
        mode: str,
        file_name: str,
        results: List[Dict[str, Any]],
    ) -> None:
        """
        モード別の結果をキャッシュ
        
        Args:
            mode: 品質モード（"draft", "standard", "thorough"）
            file_name: ファイル名
            results: 翻訳結果リスト
        """
        if file_name not in self._mode_results:
            self._mode_results[file_name] = {}
        
        self._mode_results[file_name][mode] = {
            "results": results,
            "cost_report": self.generate_report(),
        }
    
    def generate_mode_comparison(
        self,
        results: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        モード間比較レポートを生成
        
        Args:
            results: モード別の結果 {"draft": [...], "standard": [...], "thorough": [...]}
            
        Returns:
            モード間比較レポート
        """
        modes = list(results.keys())
        if not modes:
            return {"comparison": [], "cost_comparison": {}, "modes": []}
        
        # 結果の比較
        comparison = []
        
        # 最初のモードの結果をベースにする
        base_mode = modes[0]
        base_results = results.get(base_mode, [])
        
        for i, base_item in enumerate(base_results):
            text_id = base_item.get("text_id", f"row_{i}")
            row_comparison = {
                "text_id": text_id,
                "source_text": base_item.get("source_text", ""),
            }
            
            for mode in modes:
                mode_results = results.get(mode, [])
                if i < len(mode_results):
                    row_comparison[f"{mode}_result"] = mode_results[i].get("translated_text", "")
                else:
                    row_comparison[f"{mode}_result"] = ""
            
            # 差分を計算
            differences = []
            for j, mode in enumerate(modes[1:], 1):
                prev_mode = modes[j - 1]
                prev_result = row_comparison.get(f"{prev_mode}_result", "")
                curr_result = row_comparison.get(f"{mode}_result", "")
                if prev_result != curr_result:
                    differences.append({
                        "from_mode": prev_mode,
                        "to_mode": mode,
                        "changed": True,
                    })
            
            row_comparison["differences"] = differences
            comparison.append(row_comparison)
        
        # コスト比較
        cost_comparison = {}
        for mode in modes:
            if mode in self._mode_results:
                for file_name, file_data in self._mode_results.items():
                    if mode in file_data:
                        cost_comparison[mode] = file_data[mode].get("cost_report", {}).get("summary", {})
                        break
        
        # 追加コストを計算
        if len(modes) >= 2:
            for i, mode in enumerate(modes[1:], 1):
                prev_mode = modes[i - 1]
                prev_cost = cost_comparison.get(prev_mode, {}).get("total_cost_usd", 0)
                curr_cost = cost_comparison.get(mode, {}).get("total_cost_usd", 0)
                if mode in cost_comparison:
                    cost_comparison[mode]["additional_cost_usd"] = curr_cost - prev_cost
        
        return {
            "comparison": comparison,
            "cost_comparison": cost_comparison,
            "modes": modes,
        }
    
    def reset(self) -> None:
        """追跡データをリセット"""
        self._pass_data.clear()
        self._mode_results.clear()
    
    def get_total_cost(self) -> float:
        """総コストを取得"""
        return sum(p.cost_usd for p in self._pass_data.values())
    
    def get_total_tokens(self) -> Dict[str, int]:
        """総トークン数を取得"""
        return {
            "input": sum(p.input_tokens for p in self._pass_data.values()),
            "output": sum(p.output_tokens for p in self._pass_data.values()),
            "cache_hit": sum(p.cache_hit_tokens for p in self._pass_data.values()),
        }


# 便利関数
def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    """
    テキストのトークン数を計測
    
    Args:
        text: 計測対象テキスト
        model: 使用するモデル
        
    Returns:
        トークン数
    """
    counter = TokenCounter(model)
    return counter.count_tokens(text)


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = DEFAULT_MODEL,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> CostEstimate:
    """
    コストを推定
    
    Args:
        input_tokens: 入力トークン数
        output_tokens: 出力トークン数
        model: 使用するモデル
        cache_write_tokens: キャッシュ書き込みトークン数
        cache_read_tokens: キャッシュ読み込みトークン数
        
    Returns:
        CostEstimate情報
    """
    calculator = CostCalculator(model)
    return calculator.calculate_cost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_write_tokens=cache_write_tokens,
        cache_read_tokens=cache_read_tokens,
    )
