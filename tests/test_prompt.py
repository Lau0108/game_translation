"""プロンプト構築モジュールのテスト"""

import json
from typing import Any, Dict, List, Optional

import pytest
from hypothesis import given, settings, strategies as st

from excel_translator.character import CharacterProfile
from excel_translator.glossary import GlossaryEntry
from excel_translator.prompt import PromptBuilder


# ============================================================================
# テストユーティリティ
# ============================================================================

def create_text_entry(
    text_id: str,
    source_text: str,
    character: Optional[str] = None,
) -> Dict[str, Any]:
    """テスト用テキストエントリを作成"""
    return {
        "text_id": text_id,
        "character": character,
        "source_text": source_text,
    }


def create_character_profile(
    character_id: str,
    name_source: str,
    name_target: str,
    personality: str = "",
    speech_style: str = "",
    speech_examples: Optional[List[tuple]] = None,
) -> CharacterProfile:
    """テスト用キャラクタープロファイルを作成"""
    return CharacterProfile(
        character_id=character_id,
        name_source=name_source,
        name_target=name_target,
        personality=personality,
        speech_style=speech_style,
        speech_examples=speech_examples or [],
    )


# ============================================================================
# ユニットテスト
# ============================================================================

class TestPromptBuilderBasic:
    """PromptBuilder基本機能のテスト"""
    
    def test_init_default(self):
        """デフォルト初期化"""
        builder = PromptBuilder()
        assert builder.source_lang == "ja"
        assert builder.target_lang == "en"
    
    def test_init_custom_langs(self):
        """カスタム言語設定"""
        builder = PromptBuilder(source_lang="en", target_lang="zh")
        assert builder.source_lang == "en"
        assert builder.target_lang == "zh"
    
    def test_get_lang_name(self):
        """言語名取得"""
        builder = PromptBuilder()
        assert builder.get_lang_name("ja") == "日本語"
        assert builder.get_lang_name("en") == "英語"
        assert builder.get_lang_name("zh") == "中国語"
        assert builder.get_lang_name("unknown") == "unknown"
    
    def test_fixed_parameters(self):
        """固定パラメータの確認"""
        assert PromptBuilder.TEMPERATURE == 0.0
        assert PromptBuilder.MAX_TOKENS == 4096
        assert PromptBuilder.TOP_P == 1.0
    
    def test_get_api_parameters(self):
        """APIパラメータ取得"""
        builder = PromptBuilder()
        params = builder.get_api_parameters()
        
        assert params["temperature"] == 0.0
        assert params["max_tokens"] == 4096
        assert params["top_p"] == 1.0


class TestBuildSystemPrompt:
    """build_system_promptのテスト"""
    
    def test_system_prompt_contains_role(self):
        """System Promptに翻訳者ロールが含まれる"""
        builder = PromptBuilder()
        prompt = builder.build_system_prompt()
        
        assert "翻訳者" in prompt
        assert "日本語" in prompt
        assert "英語" in prompt
    
    def test_system_prompt_contains_rules(self):
        """System Promptに翻訳ルールが含まれる"""
        builder = PromptBuilder()
        prompt = builder.build_system_prompt()
        
        assert "自然さ優先" in prompt
        assert "原文忠実" in prompt
        assert "用語統一" in prompt
        assert "タグ保護" in prompt
    
    def test_system_prompt_contains_output_format(self):
        """System PromptにJSON出力フォーマットが含まれる"""
        builder = PromptBuilder()
        prompt = builder.build_system_prompt()
        
        assert "JSON" in prompt
        assert "translations" in prompt
        assert "translated_text" in prompt
        assert "remarks" in prompt
    
    def test_system_prompt_contains_remarks_rules(self):
        """System Promptにremarks運用ルールが含まれる"""
        builder = PromptBuilder()
        prompt = builder.build_system_prompt()
        
        assert "remarks" in prompt
        assert "原則として空文字" in prompt
        assert "言葉遊び" in prompt
    
    def test_system_prompt_custom_target_lang(self):
        """カスタムターゲット言語でのSystem Prompt"""
        builder = PromptBuilder(target_lang="zh")
        prompt = builder.build_system_prompt()
        
        assert "中国語" in prompt


class TestBuildTranslationPrompt:
    """build_translation_promptのテスト"""
    
    def test_translation_prompt_basic(self):
        """基本的な翻訳プロンプト"""
        builder = PromptBuilder()
        texts = [
            create_text_entry("001", "こんにちは", "勇者"),
        ]
        
        prompt = builder.build_translation_prompt(texts)
        
        assert "翻訳対象テキスト" in prompt
        assert "001" in prompt
        assert "こんにちは" in prompt
        assert "勇者" in prompt
    
    def test_translation_prompt_with_glossary(self):
        """用語集付き翻訳プロンプト"""
        builder = PromptBuilder()
        texts = [create_text_entry("001", "魔王を倒す")]
        glossary = [
            GlossaryEntry(term_source="魔王", term_target="Demon Lord"),
        ]
        
        prompt = builder.build_translation_prompt(texts, glossary_entries=glossary)
        
        assert "用語集" in prompt
        assert "魔王" in prompt
        assert "Demon Lord" in prompt
    
    def test_translation_prompt_with_character_profiles(self):
        """キャラプロファイル付き翻訳プロンプト"""
        builder = PromptBuilder()
        texts = [create_text_entry("001", "こんにちは", "勇者")]
        profiles = [
            create_character_profile(
                "hero", "勇者", "Hero",
                personality="勇敢",
                speech_style="丁寧語",
                speech_examples=[("こんにちは", "Hello")],
            ),
        ]
        
        prompt = builder.build_translation_prompt(texts, character_profiles=profiles)
        
        assert "キャラクタープロファイル" in prompt
        assert "勇者（Hero）" in prompt
        assert "勇敢" in prompt
        assert "丁寧語" in prompt
    
    def test_translation_prompt_with_context(self):
        """文脈参照付き翻訳プロンプト"""
        builder = PromptBuilder()
        texts = [create_text_entry("003", "行くぞ！", "勇者")]
        context = [
            create_text_entry("001", "魔王城が見えてきた"),
            create_text_entry("002", "いよいよだな", "騎士"),
        ]
        
        prompt = builder.build_translation_prompt(texts, context_lines=context)
        
        assert "文脈参照" in prompt
        assert "魔王城が見えてきた" in prompt
        assert "いよいよだな" in prompt


class TestBuildReviewPrompt:
    """build_review_promptのテスト"""
    
    def test_review_prompt_basic(self):
        """基本的なレビュープロンプト"""
        builder = PromptBuilder()
        source_texts = [create_text_entry("001", "こんにちは", "勇者")]
        translations = [{"text_id": "001", "translated_text": "Hello"}]
        
        prompt = builder.build_review_prompt(source_texts, translations)
        
        assert "レビュアー" in prompt
        assert "修正が必要な行のみ" in prompt
        assert "reviews" in prompt
    
    def test_review_prompt_contains_review_criteria(self):
        """レビュー観点が含まれる"""
        builder = PromptBuilder()
        source_texts = [create_text_entry("001", "こんにちは")]
        translations = [{"text_id": "001", "translated_text": "Hello"}]
        
        prompt = builder.build_review_prompt(source_texts, translations)
        
        assert "自然さ" in prompt
        assert "正確性" in prompt
        assert "キャラクター性" in prompt
        assert "一貫性" in prompt


class TestBuildConsistencyPrompt:
    """build_consistency_promptのテスト"""
    
    def test_consistency_prompt_basic(self):
        """基本的な一貫性チェックプロンプト"""
        builder = PromptBuilder()
        translations = [
            {"text_id": "001", "source_text": "こんにちは", "translated_text": "Hello", "character": "勇者"},
            {"text_id": "002", "source_text": "さようなら", "translated_text": "Goodbye", "character": "勇者"},
        ]
        
        prompt = builder.build_consistency_prompt(translations)
        
        assert "一貫性" in prompt
        assert "consistency_fixes" in prompt
    
    def test_consistency_prompt_contains_check_criteria(self):
        """チェック観点が含まれる"""
        builder = PromptBuilder()
        translations = [{"text_id": "001", "source_text": "test", "translated_text": "test"}]
        
        prompt = builder.build_consistency_prompt(translations)
        
        assert "オノマトペ" in prompt
        assert "括弧" in prompt
        assert "キャラ名表記" in prompt
        assert "文体" in prompt


class TestBuildBacktransPrompt:
    """build_backtrans_promptのテスト"""
    
    def test_backtrans_prompt_basic(self):
        """基本的なバックトランスレーションプロンプト"""
        builder = PromptBuilder()
        translations = [
            {"text_id": "001", "source_text": "こんにちは", "translated_text": "Hello"},
        ]
        
        prompt = builder.build_backtrans_prompt(translations)
        
        assert "バックトランスレーション" in prompt
        assert "backtranslations" in prompt
        assert "has_discrepancy" in prompt


class TestIsDialogue:
    """is_dialogueのテスト"""
    
    def test_is_dialogue_with_character(self):
        """キャラクターありは台詞"""
        builder = PromptBuilder()
        text = create_text_entry("001", "こんにちは", "勇者")
        
        assert builder.is_dialogue(text) is True
    
    def test_is_dialogue_without_character(self):
        """キャラクターなしは地の文"""
        builder = PromptBuilder()
        text = create_text_entry("001", "魔王城が見えてきた")
        
        assert builder.is_dialogue(text) is False
    
    def test_is_dialogue_empty_character(self):
        """空文字キャラクターは地の文"""
        builder = PromptBuilder()
        text = create_text_entry("001", "魔王城が見えてきた", "")
        
        assert builder.is_dialogue(text) is False


class TestBuildPromptForDialogue:
    """build_prompt_for_dialogueのテスト"""
    
    def test_dialogue_prompt_basic(self):
        """基本的な台詞プロンプト"""
        builder = PromptBuilder()
        text = create_text_entry("001", "こんにちは", "勇者")
        
        prompt = builder.build_prompt_for_dialogue(text)
        
        assert "台詞翻訳指示" in prompt
        assert "勇者" in prompt
        assert "こんにちは" in prompt
    
    def test_dialogue_prompt_with_profile(self):
        """プロファイル付き台詞プロンプト"""
        builder = PromptBuilder()
        text = create_text_entry("001", "こんにちは", "勇者")
        profile = create_character_profile(
            "hero", "勇者", "Hero",
            personality="勇敢",
            speech_style="丁寧語",
            speech_examples=[("こんにちは", "Hello")],
        )
        
        prompt = builder.build_prompt_for_dialogue(text, profile)
        
        assert "キャラクター情報" in prompt
        assert "勇者（Hero）" in prompt
        assert "勇敢" in prompt
        assert "翻訳例" in prompt


class TestBuildPromptForNarration:
    """build_prompt_for_narrationのテスト"""
    
    def test_narration_prompt_basic(self):
        """基本的な地の文プロンプト"""
        builder = PromptBuilder()
        text = create_text_entry("001", "魔王城が見えてきた")
        
        prompt = builder.build_prompt_for_narration(text)
        
        assert "地の文翻訳指示" in prompt
        assert "ナレーション" in prompt
        assert "魔王城が見えてきた" in prompt
    
    def test_narration_prompt_contains_rules(self):
        """地の文プロンプトに翻訳ルールが含まれる"""
        builder = PromptBuilder()
        text = create_text_entry("001", "test")
        
        prompt = builder.build_prompt_for_narration(text)
        
        assert "客観的" in prompt
        assert "情景描写" in prompt


# ============================================================================
# Property-Based Tests
# ============================================================================

@st.composite
def text_entry_strategy(draw, with_character: Optional[bool] = None):
    """テキストエントリを生成するストラテジー"""
    text_id = draw(st.text(
        alphabet=st.sampled_from("0123456789"),
        min_size=3,
        max_size=6,
    ))
    
    source_text = draw(st.sampled_from([
        "こんにちは",
        "さようなら",
        "ありがとう",
        "魔王城が見えてきた",
        "行くぞ！",
        "助けて！",
        "静かな夜だった",
        "風が吹いている",
    ]))
    
    if with_character is True:
        character = draw(st.sampled_from([
            "勇者", "魔王", "騎士", "姫", "村人A", "魔法使い"
        ]))
    elif with_character is False:
        character = None
    else:
        character = draw(st.one_of(
            st.none(),
            st.just(""),
            st.sampled_from(["勇者", "魔王", "騎士", "姫", "村人A", "魔法使い"]),
        ))
    
    return {
        "text_id": text_id,
        "character": character,
        "source_text": source_text,
    }


@st.composite
def character_profile_strategy(draw):
    """キャラクタープロファイルを生成するストラテジー"""
    character_id = draw(st.sampled_from([
        "hero", "villain", "knight", "princess", "npc1"
    ]))
    
    name_source = draw(st.sampled_from([
        "勇者", "魔王", "騎士", "姫", "村人A"
    ]))
    
    name_target = draw(st.sampled_from([
        "Hero", "Demon Lord", "Knight", "Princess", "Villager A"
    ]))
    
    personality = draw(st.sampled_from([
        "勇敢", "冷酷", "優しい", "高貴", "普通", ""
    ]))
    
    speech_style = draw(st.sampled_from([
        "丁寧語", "尊大", "普通", "敬語", ""
    ]))
    
    num_examples = draw(st.integers(min_value=0, max_value=3))
    speech_examples = []
    for _ in range(num_examples):
        source = draw(st.sampled_from([
            "こんにちは", "ありがとう", "さようなら"
        ]))
        target = draw(st.sampled_from([
            "Hello", "Thank you", "Goodbye"
        ]))
        speech_examples.append((source, target))
    
    first_person = draw(st.one_of(
        st.none(),
        st.sampled_from(["私", "俺", "僕", "わたくし"])
    ))
    
    return CharacterProfile(
        character_id=character_id,
        name_source=name_source,
        name_target=name_target,
        personality=personality,
        speech_style=speech_style,
        speech_examples=speech_examples,
        first_person=first_person,
    )


class TestProperty13DialogueNarrationSwitch:
    """
    Property 13: 地の文/台詞切替
    
    *For any* テキスト行について、characterが空（地の文）の場合と非空（台詞）の場合で
    異なる翻訳指示がプロンプトに含まれる
    
    **Feature: excel-translation-api, Property 13: 地の文/台詞切替**
    **Validates: Requirements 4.4**
    """
    
    @given(text=text_entry_strategy(with_character=True))
    @settings(max_examples=100, deadline=None)
    def test_dialogue_text_has_dialogue_instruction(self, text: Dict[str, Any]):
        """台詞テキストには台詞用の翻訳指示が含まれる"""
        builder = PromptBuilder()
        
        # 台詞として判定される
        assert builder.is_dialogue(text) is True
        
        # 翻訳プロンプトを構築
        prompt = builder.build_translation_prompt([text])
        
        # 台詞用の指示が含まれる
        assert "台詞として翻訳" in prompt, \
            f"台詞テキストに台詞用指示が含まれていません: {text}"
        
        # キャラクター名が含まれる
        assert text["character"] in prompt, \
            f"キャラクター名 '{text['character']}' がプロンプトに含まれていません"
    
    @given(text=text_entry_strategy(with_character=False))
    @settings(max_examples=100, deadline=None)
    def test_narration_text_has_narration_instruction(self, text: Dict[str, Any]):
        """地の文テキストには地の文用の翻訳指示が含まれる"""
        builder = PromptBuilder()
        
        # 地の文として判定される
        assert builder.is_dialogue(text) is False
        
        # 翻訳プロンプトを構築
        prompt = builder.build_translation_prompt([text])
        
        # 地の文用の指示が含まれる
        assert "地の文" in prompt or "ナレーション" in prompt, \
            f"地の文テキストに地の文用指示が含まれていません: {text}"
    
    @given(
        dialogue_text=text_entry_strategy(with_character=True),
        narration_text=text_entry_strategy(with_character=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_dialogue_and_narration_have_different_instructions(
        self,
        dialogue_text: Dict[str, Any],
        narration_text: Dict[str, Any],
    ):
        """台詞と地の文で異なる翻訳指示が生成される"""
        builder = PromptBuilder()
        
        # 個別にプロンプトを構築
        dialogue_prompt = builder.build_prompt_for_dialogue(dialogue_text)
        narration_prompt = builder.build_prompt_for_narration(narration_text)
        
        # 異なる指示が含まれる
        assert "台詞翻訳指示" in dialogue_prompt
        assert "地の文翻訳指示" in narration_prompt
        
        # 台詞プロンプトにはキャラクター情報が含まれる可能性がある
        assert dialogue_text["character"] in dialogue_prompt
        
        # 地の文プロンプトには地の文特有のルールが含まれる
        assert "客観的" in narration_prompt or "ナレーション" in narration_prompt
    
    @given(
        texts=st.lists(
            text_entry_strategy(),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_mixed_texts_have_appropriate_instructions(self, texts: List[Dict[str, Any]]):
        """混合テキストリストで各テキストに適切な指示が付与される"""
        builder = PromptBuilder()
        
        # 翻訳プロンプトを構築
        prompt = builder.build_translation_prompt(texts)
        
        # 各テキストについて確認
        for text in texts:
            text_id = text["text_id"]
            character = text.get("character")
            
            # text_idがプロンプトに含まれる
            assert text_id in prompt, \
                f"text_id '{text_id}' がプロンプトに含まれていません"
            
            # 台詞/地の文の判定に応じた指示が含まれる
            if builder.is_dialogue(text):
                # 台詞の場合、キャラクター名が含まれる
                assert character in prompt, \
                    f"台詞のキャラクター名 '{character}' がプロンプトに含まれていません"
    
    @given(
        text=text_entry_strategy(with_character=True),
        profile=character_profile_strategy(),
    )
    @settings(max_examples=100, deadline=None)
    def test_dialogue_with_profile_includes_character_info(
        self,
        text: Dict[str, Any],
        profile: CharacterProfile,
    ):
        """台詞プロンプトにキャラクタープロファイル情報が含まれる"""
        builder = PromptBuilder()
        
        # プロファイル付きで台詞プロンプトを構築
        prompt = builder.build_prompt_for_dialogue(text, profile)
        
        # キャラクター情報が含まれる
        assert profile.name_source in prompt, \
            f"キャラクター原文名 '{profile.name_source}' がプロンプトに含まれていません"
        assert profile.name_target in prompt, \
            f"キャラクター訳語名 '{profile.name_target}' がプロンプトに含まれていません"
        
        # 性格・口調が設定されている場合は含まれる
        if profile.personality:
            assert profile.personality in prompt, \
                f"性格 '{profile.personality}' がプロンプトに含まれていません"
        
        if profile.speech_style:
            assert profile.speech_style in prompt, \
                f"口調 '{profile.speech_style}' がプロンプトに含まれていません"
        
        # speech_examplesが設定されている場合は含まれる
        if profile.speech_examples:
            for source, target in profile.speech_examples[:3]:
                assert source in prompt, \
                    f"例文原文 '{source}' がプロンプトに含まれていません"
                assert target in prompt, \
                    f"例文訳文 '{target}' がプロンプトに含まれていません"
    
    @given(text=text_entry_strategy(with_character=False))
    @settings(max_examples=100, deadline=None)
    def test_narration_prompt_has_specific_rules(self, text: Dict[str, Any]):
        """地の文プロンプトには地の文特有のルールが含まれる"""
        builder = PromptBuilder()
        
        prompt = builder.build_prompt_for_narration(text)
        
        # 地の文特有のルールが含まれる
        assert "地の文" in prompt
        assert "ナレーション" in prompt
        assert "客観的" in prompt
        assert "情景描写" in prompt
    
    @given(text=text_entry_strategy())
    @settings(max_examples=100, deadline=None)
    def test_is_dialogue_consistent_with_character_field(self, text: Dict[str, Any]):
        """is_dialogueの判定がcharacterフィールドと一致する"""
        builder = PromptBuilder()
        
        character = text.get("character")
        is_dialogue = builder.is_dialogue(text)
        
        # characterが非空の場合は台詞
        if character is not None and character != "":
            assert is_dialogue is True, \
                f"character='{character}' なのに台詞と判定されていません"
        else:
            assert is_dialogue is False, \
                f"character='{character}' なのに地の文と判定されていません"


class TestPromptBuilderEdgeCases:
    """PromptBuilderのエッジケーステスト"""
    
    def test_empty_texts_list(self):
        """空のテキストリスト"""
        builder = PromptBuilder()
        prompt = builder.build_translation_prompt([])
        
        assert "翻訳対象テキスト" in prompt
    
    def test_empty_glossary(self):
        """空の用語集"""
        builder = PromptBuilder()
        texts = [create_text_entry("001", "test")]
        
        prompt = builder.build_translation_prompt(texts, glossary_entries=[])
        
        # 用語集セクションは含まれない
        assert "用語集" not in prompt
    
    def test_empty_character_profiles(self):
        """空のキャラクタープロファイル"""
        builder = PromptBuilder()
        texts = [create_text_entry("001", "test")]
        
        prompt = builder.build_translation_prompt(texts, character_profiles=[])
        
        # キャラクタープロファイルセクションは含まれない
        assert "キャラクタープロファイル" not in prompt
    
    def test_glossary_with_do_not_translate(self):
        """翻訳禁止用語を含む用語集"""
        builder = PromptBuilder()
        texts = [create_text_entry("001", "test")]
        glossary = [
            GlossaryEntry(
                term_source="HP",
                term_target="HP",
                do_not_translate=True,
            ),
        ]
        
        prompt = builder.build_translation_prompt(texts, glossary_entries=glossary)
        
        assert "翻訳禁止" in prompt
    
    def test_glossary_with_context_note(self):
        """文脈メモ付き用語集"""
        builder = PromptBuilder()
        texts = [create_text_entry("001", "test")]
        glossary = [
            GlossaryEntry(
                term_source="魔王",
                term_target="Demon Lord",
                context_note="ラスボスの称号",
            ),
        ]
        
        prompt = builder.build_translation_prompt(texts, glossary_entries=glossary)
        
        assert "ラスボスの称号" in prompt
