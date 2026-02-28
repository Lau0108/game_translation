# Design Document

## Overview

ゲームテキスト翻訳システムは、Excelファイルを入力として受け取り、LLM APIを使用して日本語から英語・中国語への高品質な翻訳を行うPythonバックエンドアプリケーションである。マルチパス翻訳パイプライン（1st pass翻訳生成 → 2nd passセルフレビュー → 3rd pass一貫性チェック → 4th passバックトランスレーション）により、キャラクター性・固有名詞の一貫性を保ちながら高品質な翻訳を実現する。

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Main Controller                          │
│                    (translation_app.py)                         │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐    ┌───────────────────┐    ┌───────────────┐
│  Excel_Parser │    │  Config_Manager   │    │ Excel_Writer  │
│               │    │                   │    │               │
│ - read_excel  │    │ - glossary.xlsx   │    │ - write_main  │
│ - preprocess  │    │ - characters.xlsx │    │ - write_pass  │
│ - normalize   │    │ - settings.yaml   │    │ - write_cost  │
└───────────────┘    └───────────────────┘    └───────────────┘
        │                       │
        ▼                       ▼
┌───────────────────────────────────────────────────────────────┐
│                      Chunk_Processor                          │
│                                                               │
│  - split_by_file() → split_by_tokens() → process_chunks()    │
└───────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                      Prompt_Builder                           │
│                                                               │
│  - build_system_prompt()                                      │
│  - build_translation_prompt() (1st pass)                      │
│  - build_review_prompt() (2nd pass)                           │
│  - build_consistency_prompt() (3rd pass)                      │
│  - build_backtrans_prompt() (4th pass)                        │
└───────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                    Translation_Service                        │
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │   Claude    │  │   Gemini    │  │   GPT-4     │           │
│  │   Provider  │  │   Provider  │  │   Provider  │           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
└───────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                     Review_Pipeline                           │
│                                                               │
│  1st pass ──▶ 2nd pass ──▶ 3rd pass ──▶ 4th pass             │
│  (翻訳)      (セルフ     (一貫性      (バック                 │
│              レビュー)    チェック)    トランス)               │
└───────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                       Cost_Tracker                            │
│                                                               │
│  - track_tokens() - calculate_cost() - generate_report()     │
└───────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                       QA_Checker                              │
│                                                               │
│  - check_glossary_consistency()                               │
│  - check_char_count()                                         │
│  - validate_placeholders()                                    │
└───────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Excel_Parser

```python
class ExcelParser:
    def read_excel(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """
        Excelファイルを読み込み、全シートをDataFrameとして返す
        Returns: {sheet_name: DataFrame}
        """
        pass
    
    def detect_header(self, df: pd.DataFrame) -> Optional[int]:
        """
        ヘッダー行を自動検出。存在しない場合はNone
        """
        pass
    
    def map_columns(self, df: pd.DataFrame, header_row: Optional[int]) -> pd.DataFrame:
        """
        カラムを正規名（text_id, character, source_text）にマッピング
        """
        pass
    
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        セル結合解除、空行除外、キャラ名トリム、text_id生成
        """
        pass
    
    def detect_skip_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        翻訳不要行（既に翻訳済み、空テキスト、変数のみ）にフラグ付与
        """
        pass
    
    def protect_tags(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        変数・制御タグをプレースホルダーに置換
        Returns: (protected_text, placeholder_map)
        """
        pass
    
    def restore_tags(self, text: str, placeholder_map: Dict[str, str]) -> str:
        """
        プレースホルダーを元の値に復元
        """
        pass
```

### 2. Glossary_Manager

```python
@dataclass
class GlossaryEntry:
    term_source: str
    term_target: str
    category: str
    context_note: Optional[str]
    do_not_translate: bool

class GlossaryManager:
    def load(self, file_path: str) -> List[GlossaryEntry]:
        """用語集を読み込む"""
        pass
    
    def filter_by_text(self, text: str, entries: List[GlossaryEntry]) -> List[GlossaryEntry]:
        """テキスト内にマッチする用語のみを抽出"""
        pass
    
    def verify_usage(self, source: str, translated: str, entries: List[GlossaryEntry]) -> List[str]:
        """翻訳結果で用語が正しく使用されているか検証。不一致のリストを返す"""
        pass
```

### 3. Character_Profile_Manager

```python
@dataclass
class CharacterProfile:
    character_id: str
    name_source: str
    name_target: str
    personality: str
    speech_style: str
    speech_examples: List[Tuple[str, str]]  # [(source, target), ...]
    first_person: Optional[str]
    second_person: Optional[str]
    age: Optional[str]
    gender: Optional[str]
    role: Optional[str]

class CharacterProfileManager:
    def load(self, file_path: str) -> Dict[str, CharacterProfile]:
        """キャラクタープロファイルを読み込む"""
        pass
    
    def get_profiles_for_chunk(self, character_ids: Set[str]) -> List[CharacterProfile]:
        """チャンク内の登場キャラのプロファイルのみを返す"""
        pass
```

### 4. Prompt_Builder

```python
class PromptBuilder:
    # 固定パラメータ
    TEMPERATURE = 0.0
    MAX_TOKENS = 4096
    
    def build_system_prompt(self, target_lang: str) -> str:
        """
        固定のSystem Promptを構築
        - 翻訳者ロール定義
        - 翻訳ルール（自然さ優先、原文忠実）
        - 出力フォーマット（JSON）
        - remarks運用ルール
        """
        pass
    
    def build_translation_prompt(
        self,
        texts: List[Dict],
        glossary_entries: List[GlossaryEntry],
        character_profiles: List[CharacterProfile],
        context_lines: List[Dict]
    ) -> str:
        """1st pass: 翻訳生成用プロンプト"""
        pass
    
    def build_review_prompt(
        self,
        source_texts: List[Dict],
        translations: List[Dict],
        character_profiles: List[CharacterProfile]
    ) -> str:
        """2nd pass: セルフレビュー用プロンプト（レビュアーロール）"""
        pass
    
    def build_consistency_prompt(
        self,
        all_translations: List[Dict]
    ) -> str:
        """3rd pass: 一貫性チェック用プロンプト"""
        pass
    
    def build_backtrans_prompt(
        self,
        translations: List[Dict],
        source_lang: str
    ) -> str:
        """4th pass: バックトランスレーション用プロンプト"""
        pass
```

### 5. Review_Pipeline

```python
@dataclass
class PassResult:
    text_id: str
    result: str
    reason: Optional[str]
    changed: bool

class ReviewPipeline:
    def run_2nd_pass(
        self,
        source_texts: List[Dict],
        pass1_results: List[Dict],
        character_profiles: List[CharacterProfile]
    ) -> List[PassResult]:
        """
        セルフレビュー: 修正が必要な行のみを返す
        """
        pass
    
    def run_3rd_pass_rules(
        self,
        all_translations: List[Dict]
    ) -> List[Dict]:
        """
        ルールベース一貫性チェック（正規表現）
        - オノマトペの方針統一
        - 括弧・記号の統一
        - キャラ名表記の統一
        """
        pass
    
    def run_3rd_pass_ai(
        self,
        all_translations: List[Dict],
        rule_issues: List[Dict]
    ) -> List[PassResult]:
        """
        AI一貫性チェック（文体・ニュアンス系）
        """
        pass
    
    def run_4th_pass(
        self,
        translations: List[Dict],
        target_text_ids: List[str]
    ) -> List[PassResult]:
        """
        バックトランスレーション検証（対象行のみ）
        """
        pass
```

### 6. Cost_Tracker

```python
@dataclass
class PassCost:
    pass_name: str
    input_tokens: int
    output_tokens: int
    cache_hit_tokens: int
    api_calls: int
    processing_time_ms: int
    cost_usd: float
    modified_rows: int
    total_rows: int

class CostTracker:
    def track_request(
        self,
        pass_name: str,
        input_tokens: int,
        output_tokens: int,
        cache_hit_tokens: int,
        processing_time_ms: int
    ) -> None:
        """リクエストのトークン消費を記録"""
        pass
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int, cache_hit_tokens: int) -> float:
        """料金を計算"""
        pass
    
    def get_pass_summary(self, pass_name: str) -> PassCost:
        """パスごとのサマリを取得"""
        pass
    
    def generate_report(self) -> Dict:
        """コストレポートを生成"""
        pass
    
    def generate_mode_comparison(self, results: Dict[str, List]) -> Dict:
        """モード間比較レポートを生成"""
        pass
```

### 7. Chunk_Processor

```python
@dataclass
class Chunk:
    chunk_id: str
    file_name: str
    sheet_name: str
    rows: List[Dict]
    context_rows: List[Dict]  # 文脈参照用（翻訳対象外）

class ChunkProcessor:
    MAX_TOKENS_PER_CHUNK = 2000
    CONTEXT_LINES = 3
    
    def split_into_chunks(self, data: Dict[str, pd.DataFrame]) -> List[Chunk]:
        """
        ファイル・シート単位でグループ化し、トークン上限でさらに分割
        """
        pass
    
    def merge_results(self, chunks: List[Chunk], results: List[Dict]) -> pd.DataFrame:
        """
        翻訳結果をtext_id順にマージ
        """
        pass
    
    def save_progress(self, chunk: Chunk, result: Dict) -> None:
        """チャンク単位で結果を保存"""
        pass
    
    def load_progress(self) -> Dict[str, Dict]:
        """保存済みの進捗を読み込む"""
        pass
```

### 8. Translation_Service

```python
class TranslationProvider(ABC):
    @abstractmethod
    async def translate(self, system_prompt: str, user_prompt: str) -> Dict:
        """翻訳を実行し、結果を返す"""
        pass

class ClaudeProvider(TranslationProvider):
    async def translate(self, system_prompt: str, user_prompt: str) -> Dict:
        pass

class GeminiProvider(TranslationProvider):
    async def translate(self, system_prompt: str, user_prompt: str) -> Dict:
        pass

class GPT4Provider(TranslationProvider):
    async def translate(self, system_prompt: str, user_prompt: str) -> Dict:
        pass

class TranslationService:
    def __init__(self, providers: Dict[str, TranslationProvider]):
        self.providers = providers
        self.default_provider = "claude"
    
    async def run_pipeline(
        self,
        chunk: Chunk,
        mode: str = "standard",  # draft, standard, thorough
        provider_name: str = None
    ) -> Dict:
        """
        マルチパスパイプラインを実行
        """
        pass
    
    async def translate_with_retry(
        self,
        provider: TranslationProvider,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3
    ) -> Dict:
        """指数バックオフでリトライ"""
        pass
```

### 9. Excel_Writer

```python
class ExcelWriter:
    def write_main_sheet(
        self,
        results: pd.DataFrame,
        output_path: str
    ) -> None:
        """
        メイン翻訳結果シートを出力
        カラム: text_id, source_text, translated_text, character_id,
                char_count, length_ok, glossary_check, remarks, alternative
        """
        pass
    
    def write_pass_comparison_sheet(
        self,
        pass_results: Dict[str, List],
        output_path: str
    ) -> None:
        """
        パス別比較シートを出力
        カラム: text_id, source_text, pass_1, pass_2, pass_2_reason,
                pass_3, pass_3_reason, pass_4_backtrans, final
        """
        pass
    
    def write_cost_report_sheet(
        self,
        cost_report: Dict,
        output_path: str
    ) -> None:
        """
        コストレポートシートを出力
        """
        pass
    
    def write_mode_comparison_sheet(
        self,
        mode_results: Dict[str, List],
        output_path: str
    ) -> None:
        """
        モード間比較シートを出力
        """
        pass
    
    def apply_conditional_formatting(self, ws) -> None:
        """
        条件付き書式を適用（2nd pass: 青、3rd pass: 緑、4th pass: 黄）
        """
        pass
```

## Data Models

### 入力データ構造

```python
@dataclass
class InputRow:
    text_id: str
    character: Optional[str]  # 地の文はNone
    source_text: str
    sheet_name: str
    file_name: str
    row_number: int
    skip: bool  # 翻訳不要フラグ

@dataclass
class ParsedExcel:
    rows: List[InputRow]
    summary: ParseSummary

@dataclass
class ParseSummary:
    total_sheets: int
    total_rows: int
    skipped_rows: int
    skip_reasons: Dict[str, int]  # {reason: count}
    column_mapping: Dict[str, str]
    characters_found: Set[str]
```

### 翻訳結果データ構造

```python
@dataclass
class TranslationResult:
    text_id: str
    source_text: str
    pass_1: str  # 1st pass結果
    pass_2: Optional[str]  # 2nd pass結果（変更時のみ）
    pass_2_reason: Optional[str]
    pass_3: Optional[str]  # 3rd pass結果（変更時のみ）
    pass_3_reason: Optional[str]
    pass_4_backtrans: Optional[str]  # 4th pass結果
    final: str  # 最終採用テキスト
    character_id: Optional[str]
    char_count: int
    length_ok: str
    glossary_check: str
    remarks: str
    alternative: Optional[str]
    provider: str
    response_time_ms: int
```

### 設定データ構造

```python
@dataclass
class TranslationConfig:
    target_languages: List[str]  # ["en", "zh"]
    default_provider: str  # "claude"
    quality_mode: str  # "draft", "standard", "thorough"
    chunk_size_tokens: int  # 2000
    max_retries: int  # 3
    cost_limit_usd: Optional[float]
    char_limit: Optional[int]
```



## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: 複数シート読み込み
*For any* Excelファイルに含まれる全てのシートについて、Excel_Parserで読み込んだ結果には全シートのデータが含まれ、各シート名が識別子として保持されている
**Validates: Requirements 1.1, 1.10**

### Property 2: カラムマッピング
*For any* Excelデータについて、ヘッダーの有無に関わらず、text_id・character・source_textの3カラムが正しくマッピングされる
**Validates: Requirements 1.2, 1.3, 1.4**

### Property 3: セル結合展開
*For any* セル結合を含むExcelデータについて、結合解除後は結合範囲の全セルに同じ値が展開されている
**Validates: Requirements 1.5**

### Property 4: 不要行除外
*For any* 空行またはセクション見出し行を含むデータについて、これらの行は結果から除外され、除外された行数がログに記録される
**Validates: Requirements 1.6**

### Property 5: text_id自動生成
*For any* text_idが空または存在しないデータについて、自動生成されたtext_idは全て一意であり、ファイル名+行番号の形式に従う
**Validates: Requirements 1.7**

### Property 6: キャラ名正規化
*For any* 末尾にスペースを含むキャラ名について、正規化後は末尾スペースが除去されている
**Validates: Requirements 1.8**

### Property 7: 翻訳不要行フラグ
*For any* 翻訳不要行（既に翻訳済み、空テキスト、変数のみ）について、skipフラグがTrueに設定される
**Validates: Requirements 1.11**

### Property 8: タグ保護
*For any* 変数（{...}）、制御タグ（\n, <br>等）、ゲーム固有マーカー（(heart)等）を含むテキストについて、protect_tags実行後はこれらが全てプレースホルダー（<<VAR_N>>または<<TAG_N>>）に置換されている
**Validates: Requirements 2.1, 2.2, 2.3**

### Property 9: タグ復元ラウンドトリップ
*For any* タグを含むテキストについて、protect_tags → restore_tags を実行すると元のテキストと完全に一致する
**Validates: Requirements 2.4, 2.5**

### Property 10: 用語集動的フィルタリング
*For any* テキストと用語集について、filter_by_textで抽出された用語は全てテキスト内に出現し、テキスト内に出現しない用語は含まれない
**Validates: Requirements 3.2, 3.3**

### Property 11: 用語一貫性検証
*For any* 用語集エントリと翻訳結果について、原文にterm_sourceが含まれる場合、翻訳結果にterm_targetが含まれていなければ不一致として検出される
**Validates: Requirements 3.4, 3.5**

### Property 12: キャラプロファイル動的注入
*For any* チャンクと全キャラプロファイルについて、プロンプトに注入されるプロファイルはチャンク内に登場するキャラクターのもののみであり、各プロファイルにはspeech_examplesが含まれる
**Validates: Requirements 4.2, 4.3**

### Property 13: 地の文/台詞切替
*For any* テキスト行について、characterが空（地の文）の場合と非空（台詞）の場合で異なる翻訳指示がプロンプトに含まれる
**Validates: Requirements 4.4**

### Property 14: 2nd passセルフレビュー
*For any* 1st pass翻訳結果について、2nd pass実行後は修正が必要な行のみが返され、修正理由が記録される
**Validates: Requirements 6.2, 6.3**

### Property 15: 3rd pass一貫性チェック
*For any* 翻訳結果セットについて、3rd pass実行時はルールベースチェック（正規表現）が先に実行され、その後AIチェックが実行される
**Validates: Requirements 6.4, 6.5**

### Property 16: 4th passバックトランスレーション
*For any* 翻訳結果について、4th pass実行時は修正された行・曖昧な行のみが対象となり、バックトランスレーション結果が記録される
**Validates: Requirements 6.6, 6.7**

### Property 17: パス修正理由記録
*For any* パスで修正が行われた場合、修正理由が対応するreasonフィールドに記録される
**Validates: Requirements 6.8**

### Property 18: 品質モード実行
*For any* 品質モード（Draft/Standard/Thorough）について、Draftは1st passのみ、Standardは1st+2nd+3rd pass、Thoroughは全パスが実行される
**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**

### Property 19: チャンク分割
*For any* 複数ファイル・大規模データについて、チャンクはファイル単位でグループ化され、トークン上限を超える場合は分割される。分割時は前チャンク末尾の行が次チャンクの文脈参照として含まれる
**Validates: Requirements 8.1, 8.2, 8.3**

### Property 20: チャンクマージ
*For any* 分割されたチャンクの翻訳結果について、マージ後のデータはtext_id順に並び、元のデータと同じ行数を持つ（重複・欠落がない）
**Validates: Requirements 8.4, 8.5**

### Property 21: 進捗永続化ラウンドトリップ
*For any* チャンク処理結果について、save_progress → load_progress を実行すると元の結果と完全に一致する
**Validates: Requirements 8.6**

### Property 22: トークン計測・再分割
*For any* プロンプトについて、トークン数が上限を超える場合は自動的にチャンクが再分割され、再分割後の各チャンクは上限以下になる
**Validates: Requirements 9.3**

### Property 23: コスト追跡
*For any* APIリクエストについて、入力トークン数、出力トークン数、キャッシュヒット数、処理時間が記録される
**Validates: Requirements 10.1, 10.2, 10.3**

### Property 24: コストレポート生成
*For any* 翻訳処理について、パスごとのコストサマリとプロジェクト全体のダッシュボードが生成される
**Validates: Requirements 10.4, 10.5**

### Property 25: モード間比較
*For any* 同一ファイルを異なるモードで処理した結果について、モード間比較シートが生成され、差分と追加コストが計算される
**Validates: Requirements 11.1, 11.2, 11.3, 11.4**

### Property 26: 出力Excel生成
*For any* 翻訳結果について、出力Excelにはメイン翻訳結果シート、パス別比較シート、コストレポートシートが含まれ、条件付き書式が適用される
**Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7**

### Property 27: リトライ
*For any* APIエラーが発生した場合、指数バックオフでリトライが実行され、リトライ上限に達した場合はエラーが記録されて次の行に進む
**Validates: Requirements 13.1, 13.2**

### Property 28: レート制限待機
*For any* 429エラー（レート制限）が発生した場合、適切な待機時間を設けて再試行される
**Validates: Requirements 13.3**

### Property 29: LLM比較
*For any* 比較モードでの翻訳について、選択された全てのLLMプロバイダーから結果が取得され、出力には各LLMの結果が別列として含まれる
**Validates: Requirements 15.3, 15.4, 15.5**

## Error Handling

### エラー種別と対応

| エラー種別               | 対応                                     |
| ------------------------ | ---------------------------------------- |
| ファイル読み込みエラー   | 明確なエラーメッセージを返し処理を中断   |
| カラムマッピング失敗     | ユーザーに確認を求める                   |
| APIキー未設定            | 明確なエラーメッセージを表示し処理を中断 |
| API呼び出しエラー        | 指数バックオフで最大3回リトライ          |
| レート制限（429）        | 動的待機後に再試行                       |
| JSON解析エラー           | エラーを記録し次の行に進む               |
| 用語不一致               | glossary_checkカラムに警告を記録         |
| プレースホルダー復元失敗 | エラーを記録し手動確認を促す             |
| コスト上限超過           | 処理を一時停止しユーザーに確認を求める   |

### リトライ戦略

```python
async def retry_with_backoff(func, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            return await func()
        except RateLimitError:
            wait_time = base_delay * (2 ** attempt)
            await asyncio.sleep(wait_time)
        except APIError as e:
            if attempt == max_retries - 1:
                raise
            wait_time = base_delay * (2 ** attempt)
            await asyncio.sleep(wait_time)
```

## Testing Strategy

### Unit Tests
- 各コンポーネントの個別機能をテスト
- エッジケース（空ファイル、単一行、特殊文字等）をカバー
- モックを使用したAPI呼び出しテスト

### Property-Based Tests
- 最小100イテレーションで実行
- 各プロパティは単一のテストとして実装
- タグ形式: **Feature: excel-translation-api, Property {number}: {property_text}**

### テストフレームワーク
- pytest: ユニットテスト
- hypothesis: プロパティベーステスト
- pytest-asyncio: 非同期テスト

### テスト対象の優先度

1. **高優先度（データ整合性）**
 - Property 9: タグ復元ラウンドトリップ
 - Property 20: チャンクマージ
 - Property 21: 進捗永続化ラウンドトリップ

2. **中優先度（コア機能）**
 - Property 2: カラムマッピング
 - Property 8: タグ保護
 - Property 10: 用語集動的フィルタリング
 - Property 18: 品質モード実行
 - Property 19: チャンク分割

3. **低優先度（補助機能）**
 - Property 23: コスト追跡
 - Property 26: 出力Excel生成
