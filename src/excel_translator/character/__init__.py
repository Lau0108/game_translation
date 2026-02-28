"""キャラクタープロファイル管理モジュール"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CharacterProfile:
    """キャラクタープロファイル"""
    character_id: str  # キャラクターID
    name_source: str  # 原文名
    name_target: str  # 訳語名
    personality: str = ""  # 性格
    speech_style: str = ""  # 口調
    speech_examples: List[Tuple[str, str]] = field(default_factory=list)  # [(source, target), ...]
    first_person: Optional[str] = None  # 一人称
    second_person: Optional[str] = None  # 二人称
    age: Optional[str] = None  # 年齢
    gender: Optional[str] = None  # 性別
    role: Optional[str] = None  # 役割


class CharacterProfileManagerError(Exception):
    """キャラクタープロファイル管理エラー"""
    pass


# カラム名エイリアス辞書
CHARACTER_COLUMN_ALIASES = {
    "character_id": ["character_id", "キャラID", "id", "char_id", "キャラクターID"],
    "name_source": ["name_source", "原文名", "日本語名", "jp_name", "ja_name", "name_jp", "name_ja"],
    "name_target": ["name_target", "訳語名", "英語名", "en_name", "name_en", "translation"],
    "personality": ["personality", "性格", "キャラ性格", "character"],
    "speech_style": ["speech_style", "口調", "話し方", "tone", "style"],
    "speech_examples": ["speech_examples", "例文", "サンプル", "examples", "sample"],
    "first_person": ["first_person", "一人称", "1st_person", "自称"],
    "second_person": ["second_person", "二人称", "2nd_person", "相手呼称"],
    "age": ["age", "年齢", "歳"],
    "gender": ["gender", "性別", "sex"],
    "role": ["role", "役割", "立場", "position"],
}


class CharacterProfileManager:
    """キャラクタープロファイル管理クラス"""
    
    def __init__(self):
        self._profiles: Dict[str, CharacterProfile] = {}
        self._column_aliases = CHARACTER_COLUMN_ALIASES.copy()
    
    def load(self, file_path: str, sheet_name: Optional[str] = None) -> Dict[str, CharacterProfile]:
        """
        キャラクタープロファイルを読み込む
        
        Args:
            file_path: キャラクターシートファイルパス（Excel）
            sheet_name: シート名（省略時は最初のシート）
            
        Returns:
            {character_id: CharacterProfile} の辞書
        """
        path = Path(file_path)
        if not path.exists():
            raise CharacterProfileManagerError(f"キャラクターファイルが見つかりません: {file_path}")
        
        if not path.suffix.lower() in [".xlsx", ".xls", ".xlsm"]:
            raise CharacterProfileManagerError(f"サポートされていないファイル形式: {path.suffix}")
        
        try:
            # Excelファイルを読み込み
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name, dtype=str)
            else:
                df = pd.read_excel(file_path, dtype=str)
            
            # カラムマッピング
            df = self._map_columns(df)
            
            # プロファイルに変換
            profiles = self._parse_profiles(df)
            
            self._profiles = profiles
            logger.info(f"キャラクタープロファイルを読み込みました: {len(profiles)} 件")
            
            return profiles
            
        except CharacterProfileManagerError:
            raise
        except Exception as e:
            raise CharacterProfileManagerError(f"キャラクタープロファイルの読み込みに失敗しました: {e}")

    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        カラムを正規名にマッピング
        
        Args:
            df: DataFrame
            
        Returns:
            マッピング済みDataFrame
        """
        new_columns = {}
        
        for col in df.columns:
            col_str = str(col).strip().lower() if pd.notna(col) else ""
            mapped = False
            
            for canonical, aliases in self._column_aliases.items():
                aliases_lower = [a.lower() for a in aliases]
                if col_str in aliases_lower:
                    new_columns[col] = canonical
                    mapped = True
                    break
            
            if not mapped:
                new_columns[col] = str(col)
        
        df = df.rename(columns=new_columns)
        
        # 必須カラムの存在確認
        required = ["character_id", "name_source", "name_target"]
        for col in required:
            if col not in df.columns:
                raise CharacterProfileManagerError(
                    f"必須カラム '{col}' が見つかりません。"
                    f"利用可能なカラム: {list(df.columns)}"
                )
        
        return df
    
    def _parse_profiles(self, df: pd.DataFrame) -> Dict[str, CharacterProfile]:
        """
        DataFrameからCharacterProfileの辞書を生成
        
        Args:
            df: DataFrame
            
        Returns:
            {character_id: CharacterProfile} の辞書
        """
        profiles = {}
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            character_id = str(row.get("character_id", "")).strip()
            name_source = str(row.get("name_source", "")).strip()
            name_target = str(row.get("name_target", "")).strip()
            
            # 空の行はスキップ
            if not character_id or not name_source or not name_target:
                continue
            
            # NaN値の処理
            if character_id.lower() == "nan" or name_source.lower() == "nan" or name_target.lower() == "nan":
                continue
            
            # オプションフィールドの取得
            personality = self._get_str_value(row, "personality")
            speech_style = self._get_str_value(row, "speech_style")
            first_person = self._get_optional_str(row, "first_person")
            second_person = self._get_optional_str(row, "second_person")
            age = self._get_optional_str(row, "age")
            gender = self._get_optional_str(row, "gender")
            role = self._get_optional_str(row, "role")
            
            # speech_examplesの解析
            speech_examples = self._parse_speech_examples(row.get("speech_examples", ""))
            
            profile = CharacterProfile(
                character_id=character_id,
                name_source=name_source,
                name_target=name_target,
                personality=personality,
                speech_style=speech_style,
                speech_examples=speech_examples,
                first_person=first_person,
                second_person=second_person,
                age=age,
                gender=gender,
                role=role,
            )
            profiles[character_id] = profile
        
        return profiles
    
    def _get_str_value(self, row: pd.Series, key: str) -> str:
        """文字列値を取得（NaN対応）"""
        value = row.get(key, "")
        if pd.isna(value):
            return ""
        str_value = str(value).strip()
        if str_value.lower() == "nan":
            return ""
        return str_value
    
    def _get_optional_str(self, row: pd.Series, key: str) -> Optional[str]:
        """オプション文字列値を取得（NaN対応）"""
        value = row.get(key, None)
        if pd.isna(value):
            return None
        str_value = str(value).strip()
        if str_value.lower() == "nan" or str_value == "":
            return None
        return str_value
    
    def _parse_speech_examples(self, value) -> List[Tuple[str, str]]:
        """
        speech_examplesを解析
        
        サポートするフォーマット:
        1. JSON配列: [["原文1", "訳文1"], ["原文2", "訳文2"]]
        2. 改行区切り: "原文1|訳文1\n原文2|訳文2"
        3. セミコロン区切り: "原文1|訳文1;原文2|訳文2"
        
        Args:
            value: speech_examplesの値
            
        Returns:
            [(source, target), ...] のリスト
        """
        if pd.isna(value):
            return []
        
        str_value = str(value).strip()
        if not str_value or str_value.lower() == "nan":
            return []
        
        examples = []
        
        # JSON形式を試す
        try:
            parsed = json.loads(str_value)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        examples.append((str(item[0]), str(item[1])))
                return examples
        except (json.JSONDecodeError, TypeError):
            pass
        
        # 改行区切りを試す
        if "\n" in str_value:
            lines = str_value.split("\n")
            for line in lines:
                line = line.strip()
                if "|" in line:
                    parts = line.split("|", 1)
                    if len(parts) == 2:
                        examples.append((parts[0].strip(), parts[1].strip()))
            if examples:
                return examples
        
        # セミコロン区切りを試す
        if ";" in str_value:
            items = str_value.split(";")
            for item in items:
                item = item.strip()
                if "|" in item:
                    parts = item.split("|", 1)
                    if len(parts) == 2:
                        examples.append((parts[0].strip(), parts[1].strip()))
            if examples:
                return examples
        
        # 単一の例文（|区切り）
        if "|" in str_value:
            parts = str_value.split("|", 1)
            if len(parts) == 2:
                examples.append((parts[0].strip(), parts[1].strip()))
        
        return examples

    def get_profiles_for_chunk(
        self,
        character_ids: Set[str],
        profiles: Optional[Dict[str, CharacterProfile]] = None
    ) -> List[CharacterProfile]:
        """
        チャンク内の登場キャラのプロファイルのみを返す
        
        Args:
            character_ids: チャンク内に登場するキャラクターIDのセット
            profiles: プロファイル辞書（省略時は読み込み済みプロファイルを使用）
            
        Returns:
            マッチしたCharacterProfileのリスト
        """
        if profiles is None:
            profiles = self._profiles
        
        if not character_ids or not profiles:
            return []
        
        matched = []
        
        for char_id in character_ids:
            if char_id in profiles:
                matched.append(profiles[char_id])
        
        logger.debug(f"キャラプロファイルフィルタリング: {len(matched)}/{len(profiles)} 件マッチ")
        
        return matched
    
    def get_profile(self, character_id: str) -> Optional[CharacterProfile]:
        """
        指定されたキャラクターIDのプロファイルを取得
        
        Args:
            character_id: キャラクターID
            
        Returns:
            CharacterProfile または None
        """
        return self._profiles.get(character_id)
    
    def get_all_profiles(self) -> Dict[str, CharacterProfile]:
        """
        読み込み済みの全プロファイルを取得
        
        Returns:
            {character_id: CharacterProfile} の辞書
        """
        return self._profiles.copy()
    
    def add_profile(self, profile: CharacterProfile) -> None:
        """
        プロファイルを追加
        
        Args:
            profile: 追加するプロファイル
        """
        self._profiles[profile.character_id] = profile
    
    def clear(self) -> None:
        """プロファイルをクリア"""
        self._profiles = {}
    
    def format_for_prompt(
        self,
        profiles: List[CharacterProfile],
        include_examples: bool = True
    ) -> str:
        """
        プロンプト用にプロファイルをフォーマット
        
        Args:
            profiles: フォーマットするプロファイルのリスト
            include_examples: speech_examplesを含めるか
            
        Returns:
            フォーマットされた文字列
        """
        if not profiles:
            return ""
        
        lines = ["## キャラクタープロファイル", ""]
        
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
            
            if include_examples and profile.speech_examples:
                lines.append("- 例文:")
                for source, target in profile.speech_examples:
                    lines.append(f"  - 「{source}」→「{target}」")
            
            lines.append("")
        
        return "\n".join(lines)
