"""設定管理モジュール - TranslationConfig dataclass と設定読み込み機能"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict

import yaml


class QualityMode(str, Enum):
    """品質モード定義"""
    DRAFT = "draft"        # 1st passのみ
    STANDARD = "standard"  # 1st + 2nd + 3rd pass
    THOROUGH = "thorough"  # 全パス（1st〜4th）


class LLMProvider(str, Enum):
    """LLMプロバイダー定義"""
    CLAUDE = "claude"
    GEMINI = "gemini"
    GPT4 = "gpt4"


@dataclass
class APIConfig:
    """API設定"""
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    def validate(self, provider: str) -> None:
        """指定されたプロバイダーのAPIキーが設定されているか検証"""
        key_map = {
            LLMProvider.CLAUDE.value: ("anthropic_api_key", "ANTHROPIC_API_KEY"),
            LLMProvider.GEMINI.value: ("google_api_key", "GOOGLE_API_KEY"),
            LLMProvider.GPT4.value: ("openai_api_key", "OPENAI_API_KEY"),
        }
        
        if provider not in key_map:
            raise ValueError(f"Unknown provider: {provider}")
        
        attr_name, env_name = key_map[provider]
        if not getattr(self, attr_name):
            raise ConfigurationError(
                f"APIキーが設定されていません: {env_name} 環境変数を設定してください"
            )


@dataclass
class PassRoute:
    """パスごとのルーター定義"""
    provider: str
    model: str


@dataclass
class TranslationConfig:
    """翻訳設定 dataclass"""
    # 翻訳対象言語
    target_languages: List[str] = field(default_factory=lambda: ["en"])
    
    # デフォルトLLMプロバイダー
    default_provider: str = LLMProvider.CLAUDE.value
    
    # 品質モード（draft/standard/thorough）
    quality_mode: str = QualityMode.STANDARD.value
    
    # チャンクサイズ（トークン数）
    chunk_size_tokens: int = 2000
    
    # リトライ回数上限
    max_retries: int = 3
    
    # コスト上限（USD）
    cost_limit_usd: Optional[float] = None
    
    # 文字数制限
    char_limit: Optional[int] = None
    
    # API設定
    api: APIConfig = field(default_factory=APIConfig)
    
    # パスごとのモデル指定 (pass_1, pass_2, ...)
    pass_routing: Dict[str, PassRoute] = field(default_factory=dict)
    
    # 進捗保存ディレクトリ
    progress_dir: str = ".translation_progress"
    
    # 文脈参照行数
    context_lines: int = 3
    
    def validate(self) -> None:
        """設定の妥当性を検証"""
        # 品質モードの検証
        valid_modes = [m.value for m in QualityMode]
        if self.quality_mode not in valid_modes:
            raise ConfigurationError(
                f"無効な品質モード: {self.quality_mode}. "
                f"有効な値: {', '.join(valid_modes)}"
            )
        
        # プロバイダーの検証
        valid_providers = [p.value for p in LLMProvider]
        if self.default_provider not in valid_providers:
            raise ConfigurationError(
                f"無効なプロバイダー: {self.default_provider}. "
                f"有効な値: {', '.join(valid_providers)}"
            )
        
        # APIキーの検証
        self.api.validate(self.default_provider)


class ConfigurationError(Exception):
    """設定エラー"""
    pass


def load_api_keys_from_env() -> APIConfig:
    """環境変数からAPIキーを読み込む"""
    return APIConfig(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )


def load_config_from_yaml(file_path: str) -> dict:
    """YAMLファイルから設定を読み込む"""
    path = Path(file_path)
    if not path.exists():
        raise ConfigurationError(f"設定ファイルが見つかりません: {file_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(settings_path: Optional[str] = None) -> TranslationConfig:
    """
    設定を読み込む
    
    優先順位:
    1. 設定ファイル（settings.yaml）
    2. 環境変数（APIキー）
    3. デフォルト値
    """
    config_dict = {}
    
    # 設定ファイルから読み込み
    if settings_path:
        config_dict = load_config_from_yaml(settings_path)
    else:
        # デフォルトパスを試行
        default_paths = ["settings.yaml", "config/settings.yaml"]
        for path in default_paths:
            if Path(path).exists():
                config_dict = load_config_from_yaml(path)
                break
    
    # 環境変数からAPIキーを読み込み
    api_config = load_api_keys_from_env()
    
    # 設定ファイルのAPI設定とマージ
    if "api" in config_dict:
        api_dict = config_dict.pop("api")
        # 環境変数が優先
        if not api_config.anthropic_api_key and api_dict.get("anthropic_api_key"):
            api_config.anthropic_api_key = api_dict["anthropic_api_key"]
        if not api_config.google_api_key and api_dict.get("google_api_key"):
            api_config.google_api_key = api_dict["google_api_key"]
        if not api_config.openai_api_key and api_dict.get("openai_api_key"):
            api_config.openai_api_key = api_dict["openai_api_key"]
    
    # パスルーターの構築
    pass_routing_dict = config_dict.get("pass_routing", {})
    pass_routing = {}
    for pass_name, route_info in pass_routing_dict.items():
        if isinstance(route_info, dict):
            pass_routing[pass_name] = PassRoute(
                provider=route_info.get("provider", config_dict.get("default_provider", LLMProvider.CLAUDE.value)),
                model=route_info.get("model", "")
            )
            
    # TranslationConfigを構築
    config = TranslationConfig(
        target_languages=config_dict.get("target_languages", ["en"]),
        default_provider=config_dict.get("default_provider", LLMProvider.CLAUDE.value),
        quality_mode=config_dict.get("quality_mode", QualityMode.STANDARD.value),
        chunk_size_tokens=config_dict.get("chunk_size_tokens", 2000),
        max_retries=config_dict.get("max_retries", 3),
        cost_limit_usd=config_dict.get("cost_limit_usd"),
        char_limit=config_dict.get("char_limit"),
        api=api_config,
        pass_routing=pass_routing,
        progress_dir=config_dict.get("progress_dir", ".translation_progress"),
        context_lines=config_dict.get("context_lines", 3),
    )
    
    return config
