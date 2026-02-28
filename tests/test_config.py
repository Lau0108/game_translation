"""設定管理モジュールのユニットテスト"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from excel_translator.config import (
    APIConfig,
    ConfigurationError,
    LLMProvider,
    QualityMode,
    TranslationConfig,
    load_api_keys_from_env,
    load_config,
    load_config_from_yaml,
)


class TestQualityMode:
    """品質モードのテスト"""
    
    def test_quality_mode_values(self):
        """品質モードの値が正しいことを確認"""
        assert QualityMode.DRAFT.value == "draft"
        assert QualityMode.STANDARD.value == "standard"
        assert QualityMode.THOROUGH.value == "thorough"


class TestAPIConfig:
    """API設定のテスト"""
    
    def test_validate_missing_anthropic_key(self):
        """Anthropic APIキー未設定時のエラーメッセージ検証"""
        api_config = APIConfig(anthropic_api_key=None)
        
        with pytest.raises(ConfigurationError) as exc_info:
            api_config.validate(LLMProvider.CLAUDE.value)
        
        assert "ANTHROPIC_API_KEY" in str(exc_info.value)
        assert "環境変数を設定してください" in str(exc_info.value)
    
    def test_validate_missing_google_key(self):
        """Google APIキー未設定時のエラーメッセージ検証"""
        api_config = APIConfig(google_api_key=None)
        
        with pytest.raises(ConfigurationError) as exc_info:
            api_config.validate(LLMProvider.GEMINI.value)
        
        assert "GOOGLE_API_KEY" in str(exc_info.value)
    
    def test_validate_missing_openai_key(self):
        """OpenAI APIキー未設定時のエラーメッセージ検証"""
        api_config = APIConfig(openai_api_key=None)
        
        with pytest.raises(ConfigurationError) as exc_info:
            api_config.validate(LLMProvider.GPT4.value)
        
        assert "OPENAI_API_KEY" in str(exc_info.value)
    
    def test_validate_with_valid_key(self):
        """APIキーが設定されている場合は検証が通る"""
        api_config = APIConfig(anthropic_api_key="test-key")
        api_config.validate(LLMProvider.CLAUDE.value)  # 例外が発生しない
    
    def test_validate_unknown_provider(self):
        """不明なプロバイダーでエラー"""
        api_config = APIConfig()
        
        with pytest.raises(ValueError) as exc_info:
            api_config.validate("unknown")
        
        assert "Unknown provider" in str(exc_info.value)


class TestTranslationConfig:
    """TranslationConfigのテスト"""
    
    def test_default_values(self):
        """デフォルト値の確認"""
        config = TranslationConfig()
        
        assert config.target_languages == ["en"]
        assert config.default_provider == "claude"
        assert config.quality_mode == "standard"
        assert config.chunk_size_tokens == 2000
        assert config.max_retries == 3
        assert config.cost_limit_usd is None
        assert config.char_limit is None
    
    def test_validate_invalid_quality_mode(self):
        """無効な品質モードでエラー"""
        config = TranslationConfig(
            quality_mode="invalid",
            api=APIConfig(anthropic_api_key="test")
        )
        
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        
        assert "無効な品質モード" in str(exc_info.value)
    
    def test_validate_invalid_provider(self):
        """無効なプロバイダーでエラー"""
        config = TranslationConfig(
            default_provider="invalid",
            api=APIConfig(anthropic_api_key="test")
        )
        
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        
        assert "無効なプロバイダー" in str(exc_info.value)


class TestLoadApiKeysFromEnv:
    """環境変数からのAPIキー読み込みテスト"""
    
    def test_load_from_env(self, monkeypatch):
        """環境変数からAPIキーを読み込む"""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google")
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        
        api_config = load_api_keys_from_env()
        
        assert api_config.anthropic_api_key == "test-anthropic"
        assert api_config.google_api_key == "test-google"
        assert api_config.openai_api_key == "test-openai"
    
    def test_load_missing_env(self, monkeypatch):
        """環境変数が未設定の場合はNone"""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        
        api_config = load_api_keys_from_env()
        
        assert api_config.anthropic_api_key is None
        assert api_config.google_api_key is None
        assert api_config.openai_api_key is None


class TestLoadConfigFromYaml:
    """YAML設定ファイル読み込みテスト"""
    
    def test_load_valid_yaml(self):
        """有効なYAMLファイルを読み込む"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump({
                "target_languages": ["en", "zh"],
                "quality_mode": "thorough",
            }, f)
            f.flush()
            
            config_dict = load_config_from_yaml(f.name)
            
            assert config_dict["target_languages"] == ["en", "zh"]
            assert config_dict["quality_mode"] == "thorough"
        
        os.unlink(f.name)
    
    def test_load_missing_file(self):
        """存在しないファイルでエラー"""
        with pytest.raises(ConfigurationError) as exc_info:
            load_config_from_yaml("nonexistent.yaml")
        
        assert "設定ファイルが見つかりません" in str(exc_info.value)


class TestLoadConfig:
    """設定読み込み統合テスト"""
    
    def test_load_with_yaml_file(self, monkeypatch):
        """YAMLファイルから設定を読み込む"""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump({
                "target_languages": ["ja", "ko"],
                "quality_mode": "draft",
                "chunk_size_tokens": 1500,
            }, f)
            f.flush()
            
            config = load_config(f.name)
            
            assert config.target_languages == ["ja", "ko"]
            assert config.quality_mode == "draft"
            assert config.chunk_size_tokens == 1500
            assert config.api.anthropic_api_key == "test-key"
        
        os.unlink(f.name)
    
    def test_env_overrides_yaml_api_keys(self, monkeypatch):
        """環境変数がYAMLのAPIキーより優先される"""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump({
                "api": {
                    "anthropic_api_key": "yaml-key"
                }
            }, f)
            f.flush()
            
            config = load_config(f.name)
            
            # 環境変数が優先
            assert config.api.anthropic_api_key == "env-key"
        
        os.unlink(f.name)
    
    def test_load_defaults_without_file(self, monkeypatch):
        """設定ファイルなしでデフォルト値を使用"""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        
        config = load_config(None)
        
        assert config.target_languages == ["en"]
        assert config.quality_mode == "standard"
        assert config.default_provider == "claude"
