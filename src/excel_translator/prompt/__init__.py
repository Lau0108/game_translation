"""プロンプト構築モジュール

PromptBuilderクラスを提供し、マルチパス翻訳パイプラインの各パス用プロンプトを構築する。
- 1st pass: 翻訳生成
- 2nd pass: セルフレビュー（レビュアーロール）
- 3rd pass: 一貫性チェック
- 4th pass: バックトランスレーション
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..character import CharacterProfile
from ..glossary import GlossaryEntry

logger = logging.getLogger(__name__)


@dataclass
class PromptConfig:
    """プロンプト設定（固定パラメータ）"""
    # 固定パラメータ - ユーザーによる変更禁止
    TEMPERATURE: float = 0.0
    MAX_TOKENS: int = 4096
    TOP_P: float = 1.0


class PromptBuilder:
    """
    プロンプト構築クラス
    
    マルチパス翻訳パイプラインの各パス用プロンプトを構築する。
    固定パラメータ（temperature=0等）を使用し、一貫した翻訳品質を保証する。
    """
    
    # 固定パラメータ（クラス定数）
    TEMPERATURE = 0.0
    MAX_TOKENS = 4096
    TOP_P = 1.0
    
    # remarksに書いてよいケース
    REMARKS_ALLOWED_CASES = [
        "言葉遊びの再現が不可能な場合",
        "文脈情報が不足している場合",
        "造語や新語の場合",
        "原文に誤りがあると思われる場合",
        "文字数制限を満たせない場合",
    ]
    
    def __init__(self, source_lang: str = "ja", target_lang: str = "en"):
        """
        PromptBuilderを初期化
        
        Args:
            source_lang: 原文言語コード（デフォルト: ja）
            target_lang: 翻訳先言語コード（デフォルト: en）
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self._lang_names = {
            "ja": "日本語",
            "en": "英語",
            "zh": "中国語",
            "ko": "韓国語",
        }
    
    def get_lang_name(self, lang_code: str) -> str:
        """言語コードから言語名を取得"""
        return self._lang_names.get(lang_code, lang_code)
    
    def build_system_prompt(self, target_lang: Optional[str] = None, pass_type: str = "translation") -> str:
        """
        固定のSystem Promptを構築
        
        Args:
            target_lang: 翻訳先言語コード（省略時はインスタンスのtarget_langを使用）
            pass_type: "translation", "review", "consistency", "backtranslation" のいずれか
            
        Returns:
            System Prompt文字列
        """
        target = target_lang or self.target_lang
        target_name = self.get_lang_name(target)
        source_name = self.get_lang_name(self.source_lang)
        
        if pass_type == "translation":
            json_format = f"""```json
{{
  "translations": [
    {{
      "text_id": "元のtext_id",
      "translated_text": "翻訳結果",
      "remarks": "",
      "alternative": null
    }}
  ]
}}
```

## remarksの運用ルール
remarksは**原則として空文字**としてください。以下のケースのみ記入を許可します：
{self._format_remarks_cases()}

remarksに記入する場合は、簡潔に理由を説明してください。"""
        elif pass_type == "review":
            json_format = """```json
{
  "reviews": [
    {
      "text_id": "修正が必要な行のtext_id",
      "original_translation": "元の翻訳",
      "revised_translation": "修正後の翻訳",
      "reason": "修正理由"
    }
  ]
}
```"""
        elif pass_type == "consistency":
            json_format = """```json
{
  "consistency_fixes": [
    {
      "text_id": "修正が必要な行のtext_id",
      "original_translation": "元の翻訳",
      "revised_translation": "修正後の翻訳",
      "reason": "用語または表現の統一理由"
    }
  ]
}
```"""
        elif pass_type == "backtranslation":
            json_format = """```json
{
  "backtranslations": [
    {
      "text_id": "元のtext_id",
      "translated_text": "翻訳結果",
      "backtranslation": "逆翻訳"
    }
  ]
}
```"""
        else:
            json_format = "{}"
            
        system_prompt = f"""あなたはゲームテキストの専門翻訳者です。{source_name}から{target_name}への翻訳を行います。

## 翻訳者としての役割
- ゲームの世界観とキャラクター性を深く理解し、自然で違和感のない翻訳を提供する
- 原文の意図とニュアンスを正確に伝えながら、ターゲット言語として自然な表現を使用する
- キャラクターの口調、性格、年齢、性別を考慮した翻訳を行う

## 最重要原則: 原文の文体特徴の保存
翻訳において最も重要なルールは「原文の文体特徴をそのまま保存すること」です。
AIが「英語として自然か」を過度に気にして、文体レベルを勝手に変えることは絶対に許されません。

- 原文が幼い言葉遣い → 英訳も幼い言葉遣い（語彙を簡単に）
- 原文が粗い・乱暴な言葉遣い → 英訳も粗い・乱暴な言葉遣い（スラング・砕けた表現を使用）
- 原文が詩的・文学的 → 英訳も詩的・文学的（リズム・比喩・修辞を保存）
- 原文がぶっきらぼう → 英訳もぶっきらぼう（短文・省略形を使用）
- 原文が丁寧・フォーマル → 英訳も丁寧・フォーマル
- 原文に感情が込められている → 英訳にも同等の感情表現を反映

文体の「格」を上げたり下げたりしてはいけません。原文のテクスチャをそのまま翻訳先言語で再現してください。

## 翻訳ルール
1. **文体保存最優先**: 原文の語調・レジスター・文体レベルを忠実に再現する
2. **原文忠実**: 意味やニュアンスを変えない範囲で自然な翻訳を行う
3. **キャラクター性維持**: 各キャラクターの口調・性格を一貫して反映する
4. **用語統一**: 提供された用語集の訳語を必ず使用する
5. **タグ保護**: プレースホルダー（<<VAR_N>>, <<TAG_N>>等）や絵文字（(heart)等）などの特殊記号は絶対に変更しない
6. **擬音・オノマトペの分類**:
    - **台詞（発話）内**: 原則として声・叫びとして訳す（例: 「ああっ」→"Ah!", 「ううっ」→"Ngh!"）。
    - **地の文（叙述）内**: 原則として描写・効果音として訳す（例: 「ドキドキ」→*thump-thump*, 「カサッ」→*rustle*）。
7. **Remarks（備考）の生成**: 訳語選択で迷った点、レジスタの調整が必要だった箇所などを `remarks` フィールドに記載する。

## 出力フォーマット
必ず以下のJSON形式で出力してください：
{json_format}
"""
        return system_prompt
    
    def _format_remarks_cases(self) -> str:
        """remarksに書いてよいケースをフォーマット"""
        lines = []
        for i, case in enumerate(self.REMARKS_ALLOWED_CASES, 1):
            lines.append(f"- {case}")
        return "\n".join(lines)
    
    def build_translation_prompt(
        self,
        texts: List[Dict[str, Any]],
        glossary_entries: Optional[List[GlossaryEntry]] = None,
        character_profiles: Optional[List[CharacterProfile]] = None,
        context_lines: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        1st pass: 翻訳生成用プロンプトを構築
        
        Args:
            texts: 翻訳対象テキストのリスト
                   [{"text_id": str, "character": str|None, "source_text": str}, ...]
            glossary_entries: マッチした用語集エントリのリスト
            character_profiles: チャンク内登場キャラのプロファイルリスト
            context_lines: 文脈参照用の前後行（翻訳対象外）
            
        Returns:
            User Prompt文字列
        """
        sections = []
        
        # 用語集セクション
        if glossary_entries:
            sections.append(self._format_glossary_section(glossary_entries))
        
        # キャラクタープロファイルセクション
        if character_profiles:
            sections.append(self._format_character_section(character_profiles))
        
        # 文脈参照セクション
        if context_lines:
            sections.append(self._format_context_section(context_lines))
        
        # 翻訳対象テキストセクション
        sections.append(self._format_translation_target_section(texts))
        
        return "\n\n".join(sections)
    
    def build_review_prompt(
        self,
        source_texts: List[Dict[str, Any]],
        translations: List[Dict[str, Any]],
        character_profiles: Optional[List[CharacterProfile]] = None,
    ) -> str:
        """
        2nd pass: セルフレビュー用プロンプトを構築（レビュアーロール）
        
        Args:
            source_texts: 原文テキストのリスト
                          [{"text_id": str, "character": str|None, "source_text": str}, ...]
            translations: 1st pass翻訳結果のリスト
                          [{"text_id": str, "translated_text": str}, ...]
            character_profiles: キャラクタープロファイルリスト
            
        Returns:
            User Prompt文字列
        """
        target_name = self.get_lang_name(self.target_lang)
        source_name = self.get_lang_name(self.source_lang)
        
        sections = []
        
        # レビュアーロール指示
        review_instruction = f"""## レビュアーとしての役割
あなたは今、翻訳レビュアーとして、別の翻訳者が行った{source_name}から{target_name}への翻訳をレビューします。

## レベル別チェック観点
0. **文体保存・レジスタの忠実性（最重要）**: 
   - 幼い言葉が粗雑な言葉（"pee-pee" が "cock" など）に無断変換されていないか？
   - 表現のトーン（詩的、粗野、幼さ）が原文の意図と合致しているか？
1. **抽象化逃げの検知**: 
   - （特に性的・直接的な描写において）AI特有の検閲・回避行動により、具体的な行為や表現が抽象的な描写に置き換えられていないか？原文が直接的なら、訳文も直接的にすること。
2. **感情トーンとニュアンス**: 
   - 感情の起伏、キャラクター間の関係性、シチュエーション特有のニュアンス（「抱きしめてもらう」などの受け身のニュアンス、「粘膜で繋がって」といった生々しい表現）が欠落していないか？
3. **正確性と自然さ**: 原文の意味が正確に伝わり、かつ各キャラクターの口調に適合しているか。

## 人間からの指示
「この1st passの翻訳結果には、AI特有の無難な表現への抽象化や、レジスタ（文体・語彙レベル）の違反が含まれている可能性が非常に高いです。全行に対して徹底的なレビューを行い、**本当に原文のニュアンスと語彙レベルに忠実か**を疑ってください。81行などの多数の行を与えられたにも関わらず、修正（`revised_translation`の変更）が0件であることは実質的にあり得ません。問題を見逃さないでください。」

## レビュー対象のテキスト
以下のJSONフォーマットで原文と現在の翻訳結果を提供します。
これらを比較し、問題があれば修正案を提示してください。問題がなくても出力フォーマットに従って返してください。

## 出力フォーマット
```json
{{
  "reviews": [
    {{
      "text_id": "text_id",
      "revised_translation": "修正が必要な場合は修正後のテキスト。不要な場合は現在の翻訳",
      "reason": "修正した場合はその理由。感情トーンやレジスタ違反をどう修正したかを具体的に。修正しない場合は「要件を満たしているため」など",
      "remarks": "（必須）人間による最終確認が推奨される箇所（翻訳上の判断が際どい、複数の解釈が可能な箇所など）があれば、ここに日本語で申し送り事項を記載。特になければ空文字"
    }}
  ]
}}
```
"""
        sections.append(review_instruction)
        
        # キャラクタープロファイルセクション
        if character_profiles:
            sections.append(self._format_character_section(character_profiles))
        
        # レビュー対象セクション
        sections.append(self._format_review_target_section(source_texts, translations))
        
        return "\n\n".join(sections)
    
    def build_consistency_prompt(
        self,
        all_translations: List[Dict[str, Any]],
    ) -> str:
        """
        3rd pass: 一貫性チェック用プロンプトを構築
        
        Args:
            all_translations: 全翻訳結果のリスト
                              [{"text_id": str, "source_text": str, "translated_text": str, "character": str|None}, ...]
            
        Returns:
            User Prompt文字列
        """
        target_name = self.get_lang_name(self.target_lang)
        
        consistency_instruction = f"""## 一貫性チェッカーとしての役割
あなたは翻訳の一貫性をチェックする専門家です。ファイル全体の翻訳を確認し、表記揺れや文体の不統一を検出・修正します。

## チェック観点
1. **特定語彙・表現の統一**: 頻出する特定の動詞や表現（例: 「だめになる」「イク」「気持ちいい」等）の訳語が、同じ文脈や特定キャラクター内で不自然にブレていないか
2. **オノマトペと感嘆詞の統一**: 「Ngh」「Ah」「Haa...」などの吐息・感嘆詞や、擬音語のアルファベット表記が一貫しているか
3. **時制の一貫性**: 地の文やシーン描写において、現在形（Present tense）と過去形（Past tense）が意図せず混在していないか
4. **キャラ名表記の統一**: キャラクター名のスペルや表記スタイルが一貫しているか
5. **文体の統一**: 丁寧語/タメ口、あるいは客観描写/主観描写のトーンが統一されているか
6. **記号・フォーマットの統一**: 括弧の種類（「」『』""等）、数字表記（半角/全角）が統一されているか

## 出力ルール
- **修正が必要な行のみ**を出力してください
- 修正不要な行は出力に含めないでください
- 修正理由を必ず記載してください

## 出力フォーマット
```json
{{
  "consistency_fixes": [
    {{
      "text_id": "修正が必要な行のtext_id",
      "original_translation": "元の翻訳",
      "revised_translation": "修正後の翻訳",
      "reason": "修正理由（どの観点での修正か）",
      "remarks": "（必須）特記事項があればここに日本語で記載。特になければ空文字"
    }}
  ]
}}
```

修正が不要な場合は空の配列を返してください：
```json
{{"consistency_fixes": []}}
```

## チェック対象の翻訳一覧
"""
        
        # 翻訳一覧をフォーマット
        translation_list = self._format_consistency_target_section(all_translations)
        
        return consistency_instruction + "\n" + translation_list
    
    def build_backtrans_step1_prompt(
        self,
        translations: List[Dict[str, Any]],
        source_lang: Optional[str] = None,
    ) -> str:
        """
        4th pass Step 1: ブラインド・バックトランスレーション用プロンプト
        
        Args:
            translations: 検証対象の翻訳結果リスト [{"text_id": str, "translated_text": str}, ...]
            source_lang: 原文言語コード
        """
        source = source_lang or self.source_lang
        source_name = self.get_lang_name(source)
        target_name = self.get_lang_name(self.target_lang)
        
        instruction = f"""## 役割
あなたは翻訳の専門家です。与えられた以下の{target_name}のテキストを、原文の情報を知らない状態で、純粋に{source_name}に翻訳（バックトランスレーション）してください。

## 出力フォーマット
```json
{{
  "backtranslations": [
    {{
      "text_id": "text_id",
      "backtranslation": "再翻訳結果（{source_name}）"
    }}
  ]
}}
```

## 注意事項
- **コンテキストの保持**: 文脈から推測される役割やトーンを維持してください。
- **直訳と意訳のバランス**: 意味が最も正確に伝わるように訳してください。

## 翻訳対象（{target_name}）
"""
        lines = ["```json"]
        items = [{"text_id": t["text_id"], "translated_text": t["translated_text"]} for t in translations]
        lines.append(json.dumps(items, ensure_ascii=False, indent=2))
        lines.append("```")
        
        return instruction + "\n" + "\n".join(lines)

    def build_backtrans_step2_prompt(
        self,
        comparisons: List[Dict[str, Any]],
        source_lang: Optional[str] = None,
    ) -> str:
        """
        4th pass Step 2: 意味的乖離チェック用プロンプト
        
        Args:
            comparisons: 比較対象リスト [{"text_id": str, "original_source": str, "backtranslation": str}, ...]
        """
        source = source_lang or self.source_lang
        source_name = self.get_lang_name(source)
        
        instruction = f"""## 役割
あなたはQA（品質保証）の専門家です。原文と「ブラインド・バックトランスレーション（別のAIが原文を伏せて再翻訳したもの）」を比較し、意味の乖離やニュアンスの損失がないか検証してください。

## 検証プロセス
1. 原文（{source_name}）と再翻訳結果（{source_name}）を比較する
2. 意味内容が致命的に変わっていないか、重要なニュアンスが抜け落ちていないかを確認する
3. 乖離がある場合は修正案を提示する

## 出力フォーマット
```json
{{
  "results": [
    {{
      "text_id": "text_id",
      "similarity_score": 1-10,
      "has_discrepancy": true/false,
      "discrepancy_description": "乖離の説明（乖離がある場合のみ）",
      "suggested_revision": "修正案（乖離がある場合のみ。原文の意味をより正確に反映する翻訳案）"
    }}
  ]
}}
```

- **similarity_score**: 原文と再翻訳の意味的一致度を1（全く別物）〜10（完全一致）で評価してください。
    - **重要：擬音・オノマトペ（例: Ngh, Ah, Haa, 吐息など）の表記の軽微な差異は無視してください。** これらが異なっていても、地の文や台詞の核心的な意味が一致していれば、高いスコアを付けてください。
- **has_discrepancy**: スコアが7以下の場合は `true` にしてください。

## 検証対象
"""
        lines = ["```json"]
        lines.append(json.dumps(comparisons, ensure_ascii=False, indent=2))
        lines.append("```")
        
        return instruction + "\n" + "\n".join(lines)
    
    def _format_glossary_section(self, entries: List[GlossaryEntry]) -> str:
        """用語集セクションをフォーマット"""
        lines = ["## 用語集（必ず以下の訳語を使用してください）", ""]
        
        for entry in entries:
            if entry.do_not_translate:
                line = f"- {entry.term_source} → {entry.term_source}（翻訳禁止）"
            else:
                line = f"- {entry.term_source} → {entry.term_target}"
            
            if entry.context_note:
                line += f"（{entry.context_note}）"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def _format_character_section(self, profiles: List[CharacterProfile]) -> str:
        """キャラクタープロファイルセクションをフォーマット"""
        lines = [
            "## キャラクタープロファイル",
            "",
            "**重要**: 各キャラクターの口調・文体レベルは原文に忠実に再現してください。",
            "キャラクターが幼い話し方をしている場合は英語でも幼い表現を、",
            "粗野な話し方をしている場合は英語でも粗野な表現を使用してください。",
            "AIの判断で文体の格を上げたり洗練させたりしないでください。",
            "",
        ]
        
        for profile in profiles:
            lines.append(f"### {profile.name_source}（{profile.name_target}）")
            
            if profile.personality:
                lines.append(f"- 性格: {profile.personality}")
            
            if profile.speech_style:
                lines.append(f"- 口調: {profile.speech_style}")
            
            if profile.first_person:
                lines.append(f"- 一人称: {profile.first_person}")
            
            if profile.second_person:
                lines.append(f"- 二人称: {profile.second_person}")
            
            if profile.age:
                lines.append(f"- 年齢: {profile.age}")
            
            if profile.gender:
                lines.append(f"- 性別: {profile.gender}")
            
            if profile.role:
                lines.append(f"- 役割: {profile.role}")
            
            # speech_examples（Few-shot例文）を必ず含める
            if profile.speech_examples:
                lines.append("- 例文:")
                for source, target in profile.speech_examples[:3]:  # 最大3件
                    lines.append(f"  - 「{source}」→「{target}」")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_context_section(self, context_lines: List[Dict[str, Any]]) -> str:
        """文脈参照セクションをフォーマット"""
        lines = ["## 文脈参照（翻訳対象外・参考情報）", ""]
        
        for ctx in context_lines:
            char = ctx.get("character", "")
            text = ctx.get("source_text", "")
            char_display = f"[{char}] " if char else "[地の文] "
            lines.append(f"- {char_display}{text}")
        
        return "\n".join(lines)
    
    def _format_translation_target_section(self, texts: List[Dict[str, Any]]) -> str:
        """翻訳対象テキストセクションをフォーマット"""
        lines = ["## 翻訳対象テキスト", ""]
        lines.append("以下のテキストを翻訳してください：")
        lines.append("")
        lines.append("```json")
        
        items = []
        for text in texts:
            text_id = text.get("text_id", "")
            character = text.get("character")
            source_text = text.get("source_text", "")
            
            item = {
                "text_id": text_id,
                "character": character,
                "source_text": source_text,
            }
            
            # 地の文/台詞の指示を追加
            if character:
                item["instruction"] = f"キャラクター「{character}」の台詞として翻訳"
            else:
                item["instruction"] = "地の文（ナレーション）として翻訳"
            
            items.append(item)
        
        lines.append(json.dumps(items, ensure_ascii=False, indent=2))
        lines.append("```")
        
        return "\n".join(lines)
    
    def _format_review_target_section(
        self,
        source_texts: List[Dict[str, Any]],
        translations: List[Dict[str, Any]],
    ) -> str:
        """レビュー対象セクションをフォーマット"""
        lines = ["## レビュー対象", ""]
        
        # text_idでマッピング
        trans_map = {t.get("text_id"): t for t in translations}
        
        lines.append("```json")
        items = []
        
        for src in source_texts:
            text_id = src.get("text_id", "")
            character = src.get("character")
            source_text = src.get("source_text", "")
            
            trans = trans_map.get(text_id, {})
            translated_text = trans.get("translated_text", "")
            
            item = {
                "text_id": text_id,
                "character": character,
                "source_text": source_text,
                "translated_text": translated_text,
            }
            items.append(item)
        
        lines.append(json.dumps(items, ensure_ascii=False, indent=2))
        lines.append("```")
        
        return "\n".join(lines)
    
    def _format_consistency_target_section(self, all_translations: List[Dict[str, Any]]) -> str:
        """一貫性チェック対象セクションをフォーマット"""
        lines = ["```json"]
        
        items = []
        for trans in all_translations:
            item = {
                "text_id": trans.get("text_id", ""),
                "character": trans.get("character"),
                "source_text": trans.get("source_text", ""),
                "translated_text": trans.get("translated_text", ""),
            }
            items.append(item)
        
        lines.append(json.dumps(items, ensure_ascii=False, indent=2))
        lines.append("```")
        
        return "\n".join(lines)
    
    def _format_backtrans_target_section(self, translations: List[Dict[str, Any]]) -> str:
        """バックトランスレーション対象セクションをフォーマット (DEPRECATED: Use Step 1/2 prompts)"""
        return ""
    
    def get_api_parameters(self) -> Dict[str, Any]:
        """
        固定APIパラメータを取得
        
        Returns:
            APIパラメータの辞書
        """
        return {
            "temperature": self.TEMPERATURE,
            "max_tokens": self.MAX_TOKENS,
            "top_p": self.TOP_P,
        }
    
    def build_prompt_for_dialogue(
        self,
        text: Dict[str, Any],
        character_profile: Optional[CharacterProfile] = None,
    ) -> str:
        """
        台詞用の翻訳指示を構築
        
        Args:
            text: テキスト情報 {"text_id": str, "character": str, "source_text": str}
            character_profile: キャラクタープロファイル
            
        Returns:
            台詞用翻訳指示文字列
        """
        character = text.get("character", "")
        source_text = text.get("source_text", "")
        
        lines = [f"## 台詞翻訳指示"]
        lines.append(f"キャラクター「{character}」の台詞を翻訳してください。")
        lines.append("")
        
        if character_profile:
            lines.append(f"### キャラクター情報")
            lines.append(f"- 名前: {character_profile.name_source}（{character_profile.name_target}）")
            if character_profile.personality:
                lines.append(f"- 性格: {character_profile.personality}")
            if character_profile.speech_style:
                lines.append(f"- 口調: {character_profile.speech_style}")
            if character_profile.first_person:
                lines.append(f"- 一人称: {character_profile.first_person}")
            lines.append("")
            
            if character_profile.speech_examples:
                lines.append("### 翻訳例")
                for src, tgt in character_profile.speech_examples[:3]:
                    lines.append(f"- 「{src}」→「{tgt}」")
                lines.append("")
        
        lines.append(f"### 原文")
        lines.append(f"「{source_text}」")
        
        return "\n".join(lines)
    
    def build_prompt_for_narration(
        self,
        text: Dict[str, Any],
    ) -> str:
        """
        地の文（ナレーション）用の翻訳指示を構築
        
        Args:
            text: テキスト情報 {"text_id": str, "source_text": str}
            
        Returns:
            地の文用翻訳指示文字列
        """
        source_text = text.get("source_text", "")
        
        lines = ["## 地の文翻訳指示"]
        lines.append("以下の地の文（ナレーション）を翻訳してください。")
        lines.append("")
        lines.append("### 翻訳ルール")
        lines.append("- 原文の文体、雰囲気、テンポを可能な限り維持する")
        lines.append("- 原文が詩的な場合は詩的に、粗野な場合は粗野に、幼い場合は幼く翻訳するなど、元のトーンを尊重する")
        lines.append("- 情景描写は臨場感を保ちながら自然な表現にする")
        lines.append("")
        lines.append("### 原文")
        lines.append(source_text)
        
        return "\n".join(lines)
    
    def is_dialogue(self, text: Dict[str, Any]) -> bool:
        """
        テキストが台詞かどうかを判定
        
        Args:
            text: テキスト情報 {"character": str|None, ...}
            
        Returns:
            台詞の場合True、地の文の場合False
        """
        character = text.get("character")
        return character is not None and character != ""
