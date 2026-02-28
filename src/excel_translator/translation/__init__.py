"""翻訳サービスモジュール

TranslationProvider抽象クラスと各LLMプロバイダー（Claude, Gemini, GPT-4）、
TranslationServiceクラスを提供する。
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..config import QualityMode, LLMProvider
from ..chunk import Chunk, ChunkResult
from ..cost import TokenCounter, CostCalculator, CostEstimate

logger = logging.getLogger(__name__)


class TranslationError(Exception):
    """翻訳エラー基底クラス"""
    pass


class APIError(TranslationError):
    """API呼び出しエラー"""
    pass


class RateLimitError(TranslationError):
    """レート制限エラー"""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class AuthenticationError(TranslationError):
    """認証エラー"""
    pass


class ResponseParseError(TranslationError):
    """レスポンス解析エラー"""
    pass


@dataclass
class TranslationResponse:
    """翻訳レスポンス"""
    translations: List[Dict[str, Any]]
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    response_time_ms: int = 0
    provider: str = ""
    model: str = ""
    raw_response: Optional[str] = None


class TranslationProvider(ABC):
    """
    翻訳プロバイダー抽象基底クラス
    
    各LLMプロバイダー（Claude, Gemini, GPT-4）の共通インターフェースを定義。
    """
    
    # 固定パラメータ
    TEMPERATURE = 0.0
    MAX_TOKENS = 16384
    
    def __init__(self, api_key: str, model: Optional[str] = None):
        """
        TranslationProviderを初期化
        
        Args:
            api_key: APIキー
            model: 使用するモデル名（省略時はデフォルト）
        """
        self.api_key = api_key
        self.model = model or self.default_model
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """プロバイダー名を返す"""
        pass
    
    @property
    @abstractmethod
    def default_model(self) -> str:
        """デフォルトモデル名を返す"""
        pass
    
    @abstractmethod
    async def translate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> TranslationResponse:
        """
        翻訳を実行
        
        Args:
            system_prompt: システムプロンプト
            user_prompt: ユーザープロンプト
            
        Returns:
            TranslationResponse
            
        Raises:
            APIError: API呼び出しエラー
            RateLimitError: レート制限エラー
            AuthenticationError: 認証エラー
            ResponseParseError: レスポンス解析エラー
        """
        pass
    
    def _parse_json_response(self, content: str) -> List[Dict[str, Any]]:
        """
        JSONレスポンスを解析
        
        Args:
            content: レスポンス文字列
            
        Returns:
            翻訳結果リスト
            
        Raises:
            ResponseParseError: 解析エラー
        """
        try:
            # コードブロックを除去
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            data = json.loads(content)
            
            # translations キーがある場合
            if isinstance(data, dict) and "translations" in data:
                return data["translations"]
            
            # reviews キーがある場合（2nd pass）
            if isinstance(data, dict) and "reviews" in data:
                return data["reviews"]
            
            # consistency_fixes キーがある場合（3rd pass）
            if isinstance(data, dict) and "consistency_fixes" in data:
                return data["consistency_fixes"]
            
            # backtranslations キーがある場合（4th pass）
            if isinstance(data, dict) and "backtranslations" in data:
                return data["backtranslations"]
            
            # results キーがある場合（4th pass Step 2等）
            if isinstance(data, dict) and "results" in data:
                return data["results"]
            
            # リストの場合はそのまま返す
            if isinstance(data, list):
                return data
            
            raise ResponseParseError(f"予期しないレスポンス形式: {type(data)}")
            
        except json.JSONDecodeError as e:
            raise ResponseParseError(f"JSON解析エラー: {e}")


class ClaudeProvider(TranslationProvider):
    """
    Anthropic Claude APIプロバイダー
    
    Claude 3.5 Sonnet/Opus/Haikuをサポート。
    Prompt Caching対応。
    """
    
    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
    
    def __init__(self, api_key: str, model: Optional[str] = None):
        super().__init__(api_key, model)
        self._client = None
    
    @property
    def provider_name(self) -> str:
        return "claude"
    
    @property
    def default_model(self) -> str:
        return self.DEFAULT_MODEL
    
    def _get_client(self):
        """Anthropicクライアントを取得（遅延初期化）"""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                raise TranslationError("anthropicパッケージがインストールされていません")
        return self._client
    
    async def translate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> TranslationResponse:
        """Claude APIで翻訳を実行"""
        client = self._get_client()
        
        start_time = time.time()
        
        try:
            # Prompt Caching用にシステムプロンプトをキャッシュ対象に
            response = await client.messages.create(
                model=self.model,
                max_tokens=self.MAX_TOKENS,
                temperature=self.TEMPERATURE,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
            )
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            # レスポンスからテキストを抽出
            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text
            
            # トークン使用量を取得
            usage = response.usage
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)
            cache_read = getattr(usage, "cache_read_input_tokens", 0)
            cache_write = getattr(usage, "cache_creation_input_tokens", 0)
            
            # JSONを解析
            translations = self._parse_json_response(content)
            
            return TranslationResponse(
                translations=translations,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                response_time_ms=elapsed_ms,
                provider=self.provider_name,
                model=self.model,
                raw_response=content,
            )
            
        except Exception as e:
            error_str = str(e).lower()
            
            if "rate_limit" in error_str or "429" in error_str:
                # Retry-Afterヘッダーを解析
                retry_after = None
                if hasattr(e, "response") and hasattr(e.response, "headers"):
                    retry_after = e.response.headers.get("retry-after")
                    if retry_after:
                        retry_after = float(retry_after)
                raise RateLimitError(str(e), retry_after)
            
            if "authentication" in error_str or "401" in error_str:
                raise AuthenticationError(f"Claude認証エラー: {e}")
            
            raise APIError(f"Claude APIエラー: {e}")


class GeminiProvider(TranslationProvider):
    """
    Google Gemini APIプロバイダー
    
    google-genai SDK を使用。Gemini 2.5/3系をサポート。
    """
    
    DEFAULT_MODEL = "gemini-2.5-pro"
    
    def __init__(self, api_key: str, model: Optional[str] = None):
        super().__init__(api_key, model)
        self._client = None
    
    @property
    def provider_name(self) -> str:
        return "gemini"
    
    @property
    def default_model(self) -> str:
        return self.DEFAULT_MODEL
    
    def _get_client(self):
        """Geminiクライアントを取得（遅延初期化）"""
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise TranslationError("google-genaiパッケージがインストールされていません")
        return self._client
    
    async def translate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> TranslationResponse:
        """Gemini APIで翻訳を実行"""
        client = self._get_client()
        
        start_time = time.time()
        
        try:
            from google.genai import types
            
            config = types.GenerateContentConfig(
                temperature=self.TEMPERATURE,
                max_output_tokens=self.MAX_TOKENS,
                response_mime_type="application/json",
                system_instruction=system_prompt,
            )
            
            # 非同期実行
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=self.model,
                contents=user_prompt,
                config=config,
            )
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            content = response.text
            
            # トークン使用量を取得（usage_metadataから実値を使用）
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                meta = response.usage_metadata
                input_tokens = getattr(meta, 'prompt_token_count', 0) or 0
                output_tokens = getattr(meta, 'candidates_token_count', 0) or 0
                # thinking tokens
                thoughts = getattr(meta, 'thoughts_token_count', 0) or 0
                output_tokens += thoughts
            
            # fallback: usage_metadataがない場合はtiktokenで推定
            if input_tokens == 0:
                token_counter = TokenCounter(self.model)
                input_tokens = token_counter.count_tokens(f"{system_prompt}\n\n{user_prompt}")
            if output_tokens == 0:
                token_counter = TokenCounter(self.model)
                output_tokens = token_counter.count_tokens(content)
            
            # JSONを解析
            translations = self._parse_json_response(content)
            
            return TranslationResponse(
                translations=translations,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=0,
                cache_write_tokens=0,
                response_time_ms=elapsed_ms,
                provider=self.provider_name,
                model=self.model,
                raw_response=content,
            )
            
        except Exception as e:
            error_str = str(e).lower()
            
            if "quota" in error_str or "rate" in error_str or "429" in error_str:
                retry_after = None
                import re
                match = re.search(r"'retrydelay':\s*'(\d+(\.\d+)?)s'", error_str)
                if match:
                    retry_after = float(match.group(1))
                raise RateLimitError(str(e), retry_after)
            
            if "api_key" in error_str or "authentication" in error_str or "401" in error_str:
                raise AuthenticationError(f"Gemini認証エラー: {e}")
            
            raise APIError(f"Gemini APIエラー: {e}")


class GPT4Provider(TranslationProvider):
    """
    OpenAI GPT-4 APIプロバイダー
    
    GPT-4 Turbo/GPT-4oをサポート。
    """
    
    DEFAULT_MODEL = "gpt-4o"
    
    def __init__(self, api_key: str, model: Optional[str] = None):
        super().__init__(api_key, model)
        self._client = None
    
    @property
    def provider_name(self) -> str:
        return "gpt4"
    
    @property
    def default_model(self) -> str:
        return self.DEFAULT_MODEL
    
    def _get_client(self):
        """OpenAIクライアントを取得（遅延初期化）"""
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                raise TranslationError("openaiパッケージがインストールされていません")
        return self._client
    
    async def translate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> TranslationResponse:
        """OpenAI APIで翻訳を実行"""
        client = self._get_client()
        
        start_time = time.time()
        
        try:
            response = await client.chat.completions.create(
                model=self.model,
                max_tokens=self.MAX_TOKENS,
                temperature=self.TEMPERATURE,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            # レスポンスからテキストを抽出
            content = response.choices[0].message.content or ""
            
            # トークン使用量を取得
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            
            # JSONを解析
            translations = self._parse_json_response(content)
            
            return TranslationResponse(
                translations=translations,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=0,
                cache_write_tokens=0,
                response_time_ms=elapsed_ms,
                provider=self.provider_name,
                model=self.model,
                raw_response=content,
            )
            
        except Exception as e:
            error_str = str(e).lower()
            
            if "rate_limit" in error_str or "429" in error_str:
                retry_after = None
                if hasattr(e, "response") and hasattr(e.response, "headers"):
                    retry_after = e.response.headers.get("retry-after")
                    if retry_after:
                        retry_after = float(retry_after)
                raise RateLimitError(str(e), retry_after)
            
            if "authentication" in error_str or "401" in error_str or "invalid_api_key" in error_str:
                raise AuthenticationError(f"OpenAI認証エラー: {e}")
            
            raise APIError(f"OpenAI APIエラー: {e}")



@dataclass
class PassResult:
    """パス処理結果"""
    text_id: str
    result: str
    reason: Optional[str] = None
    changed: bool = False
    remarks: str = ""


@dataclass
class PipelineResult:
    """パイプライン処理結果"""
    text_id: str
    source_text: str
    pass_1: str
    pass_2: Optional[str] = None
    pass_2_reason: Optional[str] = None
    pass_3: Optional[str] = None
    pass_3_reason: Optional[str] = None
    pass_4_backtrans: Optional[str] = None
    character_id: Optional[str] = None
    provider: str = ""
    response_time_ms: int = 0
    remarks: str = ""
    
    @property
    def final(self) -> str:
        """最新のパスの結果を返す"""
        return self.pass_3 or self.pass_2 or self.pass_1 or ""
    
    @property
    def alternative(self) -> str:
        """修正があった場合、初期翻訳(pass_1)を代替案として返す"""
        if self.final and self.pass_1 and self.final != self.pass_1:
            return self.pass_1
        return ""


class TranslationService:
    """
    翻訳サービスクラス
    
    マルチパスパイプライン（1st〜4th pass）を実行し、
    品質モード（Draft/Standard/Thorough）に応じた翻訳を提供する。
    """
    
    # リトライ設定
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_BASE_DELAY = 1.0
    DEFAULT_MAX_DELAY = 60.0
    
    def __init__(
        self,
        providers: Dict[str, TranslationProvider],
        default_provider: str = "claude",
        max_retries: int = DEFAULT_MAX_RETRIES,
        pass_routing: Optional[Dict[str, Any]] = None,
        source_lang: str = "ja",
        target_lang: str = "en",
    ):
        """
        TranslationServiceを初期化
        
        Args:
            providers: プロバイダー辞書 {name: TranslationProvider}
            default_provider: デフォルトプロバイダー名
            max_retries: 最大リトライ回数
        """
        self.providers = providers
        self.default_provider = default_provider
        self.max_retries = max_retries
        self.pass_routing = pass_routing or {}
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # 統計情報
        self._total_requests = 0
        self._total_retries = 0
        self._errors: List[Dict[str, Any]] = []
    
    def get_provider(self, name: Optional[str] = None) -> TranslationProvider:
        """
        プロバイダーを取得
        
        Args:
            name: プロバイダー名（省略時はデフォルト）
            
        Returns:
            TranslationProvider
            
        Raises:
            ValueError: プロバイダーが見つからない場合
        """
        provider_name = name or self.default_provider
        
        if provider_name not in self.providers:
            available = list(self.providers.keys())
            raise ValueError(
                f"プロバイダー '{provider_name}' が見つかりません。"
                f"利用可能: {available}"
            )
        
        return self.providers[provider_name]
    
    def get_provider_for_pass(self, pass_name: str) -> TranslationProvider:
        """
        指定されたパスに設定されたプロバイダー（およびモデル）を取得する
        """
        if pass_name in self.pass_routing:
            route = self.pass_routing[pass_name]
            # PassRouteはdataclassなので属性に直接アクセス
            provider_name = getattr(route, 'provider', self.default_provider)
            model_name = getattr(route, 'model', '')
            
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                if model_name:
                    provider.model = model_name
                return provider
                
        return self.get_provider()
    
    async def translate_with_retry(
        self,
        provider: TranslationProvider,
        system_prompt: str,
        user_prompt: str,
        max_retries: Optional[int] = None,
    ) -> TranslationResponse:
        """
        指数バックオフでリトライしながら翻訳を実行
        
        Args:
            provider: 使用するプロバイダー
            system_prompt: システムプロンプト
            user_prompt: ユーザープロンプト
            max_retries: 最大リトライ回数（省略時はインスタンス設定）
            
        Returns:
            TranslationResponse
            
        Raises:
            TranslationError: リトライ上限に達した場合
        """
        retries = max_retries if max_retries is not None else self.max_retries
        last_error = None
        
        for attempt in range(retries + 1):
            try:
                self._total_requests += 1
                response = await provider.translate(system_prompt, user_prompt)
                return response
                
            except RateLimitError as e:
                last_error = e
                self._total_retries += 1
                
                if attempt >= retries:
                    break
                
                # Retry-Afterヘッダーがあればそれを使用
                if e.retry_after:
                    wait_time = e.retry_after
                else:
                    # 指数バックオフ
                    wait_time = min(
                        self.DEFAULT_BASE_DELAY * (2 ** attempt),
                        self.DEFAULT_MAX_DELAY
                    )
                
                logger.warning(
                    f"レート制限エラー (attempt {attempt + 1}/{retries + 1}), "
                    f"{wait_time:.1f}秒待機: {e}"
                )
                await asyncio.sleep(wait_time)
                
            except APIError as e:
                last_error = e
                self._total_retries += 1
                
                if attempt >= retries:
                    break
                
                # 指数バックオフ
                wait_time = min(
                    self.DEFAULT_BASE_DELAY * (2 ** attempt),
                    self.DEFAULT_MAX_DELAY
                )
                
                logger.warning(
                    f"APIエラー (attempt {attempt + 1}/{retries + 1}), "
                    f"{wait_time:.1f}秒待機: {e}"
                )
                await asyncio.sleep(wait_time)
                
            except (AuthenticationError, ResponseParseError) as e:
                # 認証エラーと解析エラーはリトライしない
                self._errors.append({
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "provider": provider.provider_name,
                })
                raise
        
        # リトライ上限に達した
        self._errors.append({
            "error_type": type(last_error).__name__,
            "message": str(last_error),
            "provider": provider.provider_name,
            "retries": retries,
        })
        
        raise TranslationError(
            f"リトライ上限（{retries}回）に達しました: {last_error}"
        )
    
    def _get_passes_for_mode(self, mode: str) -> List[int]:
        """
        品質モードに応じた実行パスを取得
        
        Args:
            mode: 品質モード（draft/standard/thorough）
            
        Returns:
            実行するパス番号のリスト
        """
        mode_lower = mode.lower()
        
        if mode_lower == QualityMode.DRAFT.value:
            return [1]  # 1st passのみ
        elif mode_lower == QualityMode.STANDARD.value:
            return [1, 2, 3]  # 1st + 2nd + 3rd pass
        elif mode_lower == QualityMode.THOROUGH.value:
            return [1, 2, 3, 4]  # 全パス
        else:
            logger.warning(f"不明な品質モード: {mode}, Standardを使用")
            return [1, 2, 3]
    
    async def run_pipeline(
        self,
        chunk: Chunk,
        prompt_builder,  # PromptBuilder
        mode: str = "standard",
        provider_name: Optional[str] = None,
        glossary_entries: Optional[List] = None,
        character_profiles: Optional[List] = None,
    ) -> Tuple[List[PipelineResult], Dict[str, Any]]:
        """
        マルチパスパイプラインを実行
        
        Args:
            chunk: 処理対象チャンク
            prompt_builder: PromptBuilderインスタンス
            mode: 品質モード（draft/standard/thorough）
            provider_name: 使用するプロバイダー名
            glossary_entries: 用語集エントリ
            character_profiles: キャラクタープロファイル
            
        Returns:
            (PipelineResult リスト, 統計情報)
        """
        provider = self.get_provider(provider_name)
        passes = self._get_passes_for_mode(mode)
        
        # 翻訳対象テキストを準備
        texts = [
            {
                "text_id": row.text_id,
                "character": row.character,
                "source_text": row.source_text,
            }
            for row in chunk.rows
        ]
        
        # 文脈参照を準備
        context_lines = [
            {
                "text_id": row.text_id,
                "character": row.character,
                "source_text": row.source_text,
            }
            for row in chunk.context_rows
        ]
        
        # 統計情報
        stats = {
            "mode": mode,
            "provider": provider.provider_name,
            "model": provider.model,
            "passes_executed": [],
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_read_tokens": 0,
            "total_cache_write_tokens": 0,
            "total_response_time_ms": 0,
            "pass_stats": {}
        }
        if not texts:
            return [], stats
        
        # 結果を格納
        results: Dict[str, PipelineResult] = {}
        
        # 1st pass: 初期翻訳
        if 1 in passes:
            provider = self.get_provider_for_pass("pass_1")
            stats["provider"] = provider.provider_name # 実際に使用されるプロバイダーを更新
            stats["model"] = provider.model # 実際に使用されるモデルを更新

            system_prompt = prompt_builder.build_system_prompt()
            user_prompt = prompt_builder.build_translation_prompt(
                texts=texts,
                glossary_entries=glossary_entries,
                character_profiles=character_profiles,
                context_lines=context_lines,
            )
            
            try:
                response = await self.translate_with_retry(
                    provider, system_prompt, user_prompt
                )
                
                stats["passes_executed"].append(1)
                stats["pass_stats"][1] = {
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cache_read_tokens": response.cache_read_tokens,
                    "cache_write_tokens": response.cache_write_tokens,
                    "response_time_ms": response.response_time_ms,
                }
                stats["total_input_tokens"] += response.input_tokens
                stats["total_output_tokens"] += response.output_tokens
                stats["total_cache_read_tokens"] += response.cache_read_tokens
                stats["total_cache_write_tokens"] += response.cache_write_tokens
                stats["total_response_time_ms"] += response.response_time_ms
                
                # 結果をマッピング
                for trans in response.translations:
                    text_id = trans.get("text_id", "")
                    translated_text = trans.get("translated_text", "")
                    remarks = trans.get("remarks", "")
                    
                    # 元のテキスト情報を取得
                    source_info = next(
                        (t for t in texts if t["text_id"] == text_id),
                        {}
                    )
                    
                    results[text_id] = PipelineResult(
                        text_id=text_id,
                        source_text=source_info.get("source_text", ""),
                        pass_1=translated_text,
                        character_id=source_info.get("character"),
                        provider=provider.provider_name,
                        response_time_ms=response.response_time_ms,
                        remarks=remarks,
                    )
                
                # Pass 1 stats
                stats["pass_stats"][1]["modified_rows"] = len(results)
                stats["pass_stats"][1]["total_rows"] = len(results)
                    
            except TranslationError as e:
                logger.error(f"1st pass失敗: {e}")
                raise e
        
        from ..review import ReviewPipeline
        review_pipeline = ReviewPipeline(
            translation_provider=provider,
            prompt_builder=prompt_builder,
            source_lang="ja",  # Hardcoding standard or use config if available passed in
            target_lang="en",
        )
        
        # 2nd pass: セルフレビュー
        if 2 in passes and results:
            review_provider = self.get_provider_for_pass("pass_2")
            review_pipeline.translation_provider = review_provider
            
            source_texts = texts
            pass1_results = [
                {"text_id": r.text_id, "translated_text": r.pass_1}
                for r in results.values()
            ]
            
            try:
                pass2_res, pass2_stats = await review_pipeline.run_2nd_pass(
                    source_texts=source_texts,
                    pass1_results=pass1_results,
                    character_profiles=character_profiles,
                )
                
                stats["passes_executed"].append(2)
                stats["pass_stats"][2] = pass2_stats
                stats["total_input_tokens"] += pass2_stats["input_tokens"]
                stats["total_output_tokens"] += pass2_stats["output_tokens"]
                stats["total_cache_read_tokens"] += pass2_stats["cache_read_tokens"]
                stats["total_cache_write_tokens"] += pass2_stats["cache_write_tokens"]
                stats["total_response_time_ms"] += pass2_stats["response_time_ms"]
                
                # 修正を適用
                modified_count = 0
                for r in pass2_res:
                    if r.text_id in results:
                        if r.changed:
                            results[r.text_id].pass_2 = r.result
                            results[r.text_id].pass_2_reason = r.reason
                            modified_count += 1
                        
                        if r.remarks:
                            prefix = "【2nd Pass】"
                            if results[r.text_id].remarks:
                                results[r.text_id].remarks += f" / {prefix}{r.remarks}"
                            else:
                                results[r.text_id].remarks = f"{prefix}{r.remarks}"
                
                stats["pass_stats"][2]["modified_rows"] = modified_count
                stats["pass_stats"][2]["total_rows"] = len(results)
                        
            except Exception as e:
                logger.warning(f"2nd pass失敗（続行）: {e}")
        
        # 3rd pass: 一貫性チェック
        if 3 in passes and results:
            consistency_provider = self.get_provider_for_pass("pass_3")
            review_pipeline.translation_provider = consistency_provider
            
            all_translations = [
                {
                    "text_id": r.text_id,
                    "source_text": r.source_text,
                    "translated_text": r.pass_2 or r.pass_1 or "",
                    "character": r.character_id,
                }
                for r in results.values()
            ]
            
            try:
                # 前処理（ルールベース機械的修正）
                preprocess_res = review_pipeline.run_3rd_pass_preprocess(all_translations)
                for r in preprocess_res:
                    if r.text_id in results:
                        results[r.text_id].pass_3 = r.result
                        results[r.text_id].pass_3_reason = r.reason
                
                rule_issues = review_pipeline.run_3rd_pass_rules(all_translations)
                pass3_res, pass3_stats = await review_pipeline.run_3rd_pass_ai(
                    all_translations=all_translations,
                    rule_issues=rule_issues,
                )
                
                stats["passes_executed"].append(3)
                stats["pass_stats"][3] = pass3_stats
                stats["total_input_tokens"] += pass3_stats["input_tokens"]
                stats["total_output_tokens"] += pass3_stats["output_tokens"]
                stats["total_cache_read_tokens"] += pass3_stats["cache_read_tokens"]
                stats["total_cache_write_tokens"] += pass3_stats["cache_write_tokens"]
                stats["total_response_time_ms"] += pass3_stats["response_time_ms"]
                
                # 修正を適用
                modified_count = 0
                for r in pass3_res:
                    if r.text_id in results and r.changed:
                        results[r.text_id].pass_3 = r.result
                        results[r.text_id].pass_3_reason = r.reason
                        modified_count += 1
                        
                        if r.remarks:
                            prefix = "【3rd Pass】"
                            if results[r.text_id].remarks:
                                results[r.text_id].remarks += f" / {prefix}{r.remarks}"
                            else:
                                results[r.text_id].remarks = f"{prefix}{r.remarks}"
                
                stats["pass_stats"][3]["modified_rows"] = modified_count
                stats["pass_stats"][3]["total_rows"] = len(results)
                        
            except Exception as e:
                logger.warning(f"3rd pass失敗（続行）: {e}")
        
        # 4th pass: バックトランスレーション
        if 4 in passes and results:
            # 修正された行・曖昧な行のみを対象
            target_translations = [
                {
                    "text_id": r.text_id,
                    "source_text": r.source_text,
                    "translated_text": r.pass_3 or r.pass_2 or r.pass_1 or "",
                }
                for r in results.values()
            ]
            
            if target_translations:
                try:
                    # Pass 4用のプロバイダーを設定
                    bt_provider = self.get_provider_for_pass("pass_4")
                    review_pipeline.translation_provider = bt_provider
                    
                    pass4_res, pass4_stats = await review_pipeline.run_4th_pass(
                        translations=target_translations
                    )
                    
                    stats["passes_executed"].append(4)
                    stats["pass_stats"][4] = pass4_stats
                    stats["total_input_tokens"] += pass4_stats["input_tokens"]
                    stats["total_output_tokens"] += pass4_stats["output_tokens"]
                    stats["total_cache_read_tokens"] += pass4_stats["cache_read_tokens"]
                    stats["total_cache_write_tokens"] += pass4_stats["cache_write_tokens"]
                    stats["total_response_time_ms"] += pass4_stats["response_time_ms"]
                    
                    stats["passes_executed"].append(4)
                    stats["pass_stats"][4]["total_rows"] = len(results)
                    stats["pass_stats"][4]["modified_rows"] = self._stats.get("4th_pass_discrepancies", 0) # 乖離検出数を便宜上記録
                    
                    # バックトランスレーション結果を記録
                    for bt in pass4_res:
                        if bt.text_id in results:
                            results[bt.text_id].pass_4_backtrans = bt.backtranslation
                            
                            # 乖離がある場合やスコアが低い場合はremarksに要確認フラグを付与
                            score = getattr(bt, 'similarity_score', 0)
                            has_disc = getattr(bt, 'has_discrepancy', False)
                            desc = getattr(bt, 'discrepancy_description', '')
                            
                            if has_disc or (0 < score < 7):
                                prefix = "【4th Pass 乖離大】"
                                remark = f"{prefix}Score: {score}/10"
                                if desc:
                                    remark += f" ({desc})"
                                
                                if results[bt.text_id].remarks:
                                    results[bt.text_id].remarks += f" / {remark}"
                                else:
                                    results[bt.text_id].remarks = remark
                            
                except Exception as e:
                    logger.warning(f"4th pass失敗（続行）: {e}")
        
        return list(results.values()), stats
    
    async def run_comparison(
        self,
        chunk: Chunk,
        prompt_builder,
        provider_names: List[str],
        mode: str = "standard",
        glossary_entries: Optional[List] = None,
        character_profiles: Optional[List] = None,
    ) -> Dict[str, Tuple[List[PipelineResult], Dict[str, Any]]]:
        """
        複数プロバイダーで並列翻訳を実行（比較モード）
        
        Args:
            chunk: 処理対象チャンク
            prompt_builder: PromptBuilderインスタンス
            provider_names: 比較するプロバイダー名リスト
            mode: 品質モード
            glossary_entries: 用語集エントリ
            character_profiles: キャラクタープロファイル
            
        Returns:
            {provider_name: (results, stats)} の辞書
        """
        tasks = []
        
        for provider_name in provider_names:
            task = self.run_pipeline(
                chunk=chunk,
                prompt_builder=prompt_builder,
                mode=mode,
                provider_name=provider_name,
                glossary_entries=glossary_entries,
                character_profiles=character_profiles,
            )
            tasks.append((provider_name, task))
        
        # 並列実行
        comparison_results = {}
        
        for provider_name, task in tasks:
            try:
                results, stats = await task
                comparison_results[provider_name] = (results, stats)
            except Exception as e:
                logger.error(f"プロバイダー '{provider_name}' でエラー: {e}")
                comparison_results[provider_name] = ([], {"error": str(e)})
        
        return comparison_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得"""
        return {
            "total_requests": self._total_requests,
            "total_retries": self._total_retries,
            "errors": self._errors,
        }
    
    def reset_statistics(self) -> None:
        """統計情報をリセット"""
        self._total_requests = 0
        self._total_retries = 0
        self._errors = []


def create_provider(
    provider_name: str,
    api_key: str,
    model: Optional[str] = None,
) -> TranslationProvider:
    """
    プロバイダーを作成
    
    Args:
        provider_name: プロバイダー名（claude/gemini/gpt4）
        api_key: APIキー
        model: モデル名（省略時はデフォルト）
        
    Returns:
        TranslationProvider
        
    Raises:
        ValueError: 不明なプロバイダー名
    """
    provider_map = {
        "claude": ClaudeProvider,
        "gemini": GeminiProvider,
        "gpt4": GPT4Provider,
    }
    
    provider_name_lower = provider_name.lower()
    
    if provider_name_lower not in provider_map:
        raise ValueError(
            f"不明なプロバイダー: {provider_name}. "
            f"利用可能: {list(provider_map.keys())}"
        )
    
    provider_class = provider_map[provider_name_lower]
    return provider_class(api_key=api_key, model=model)


def create_service_from_config(config) -> TranslationService:
    """
    設定からTranslationServiceを作成
    
    Args:
        config: TranslationConfig
        
    Returns:
        TranslationService
    """
    providers = {}
    
    # Claude
    if config.api.anthropic_api_key:
        providers["claude"] = ClaudeProvider(
            api_key=config.api.anthropic_api_key
        )
    
    # Gemini
    if config.api.google_api_key:
        providers["gemini"] = GeminiProvider(
            api_key=config.api.google_api_key
        )
    
    # GPT-4
    if config.api.openai_api_key:
        providers["gpt4"] = GPT4Provider(
            api_key=config.api.openai_api_key
        )
    
    if not providers:
        raise TranslationError(
            "利用可能なプロバイダーがありません。"
            "APIキーを設定してください。"
        )
    
    return TranslationService(
        providers=providers,
        default_provider=config.default_provider,
        max_retries=config.max_retries,
        pass_routing=config.pass_routing,
        source_lang=config.source_languages[0] if hasattr(config, "source_languages") and config.source_languages else "ja",
        target_lang=config.target_languages[0] if config.target_languages else "en",
    )
