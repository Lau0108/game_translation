# Requirements Document

## Introduction

ゲームのテキストデータ（Excel）をLLM APIで翻訳し、キャラクター性・固有名詞の一貫性を保ちながら高品質な翻訳Excelを出力するPythonバックエンドアプリケーション。マルチパス翻訳パイプライン（翻訳生成→セルフレビュー→一貫性チェック→バックトランスレーション）、パス別比較・コストレポート、品質モード選択を備える。

## Glossary

- **Excel_Parser**: 入力Excelを読み込み、前処理・正規化を行うコンポーネント
- **Glossary_Manager**: 用語集（固有名詞・専門用語）を管理し、翻訳時に制約として注入するコンポーネント
- **Character_Profile_Manager**: キャラクターの口調・性格を管理し、Few-shot例文を提供するコンポーネント
- **Prompt_Builder**: 動的にプロンプトを組み立てるコンポーネント
- **Translation_Service**: LLM APIを呼び出し翻訳を実行するコンポーネント
- **Review_Pipeline**: マルチパスレビュー（セルフレビュー、一貫性チェック、バックトランスレーション）を実行するコンポーネント
- **Chunk_Processor**: 大規模ファイルを分割処理し、結果をマージするコンポーネント
- **QA_Checker**: 翻訳結果の品質チェック（用語一貫性、文字数等）を行うコンポーネント
- **Cost_Tracker**: トークン消費量・料金を追跡し、コストレポートを生成するコンポーネント
- **Excel_Writer**: 翻訳結果をExcel形式で出力するコンポーネント

## Requirements

### Requirement 1: 入力Excel解析と前処理

**User Story:** As a ユーザー, I want to 様々なフォーマットのExcelファイルを自動で解析・正規化する, so that 手動でのデータ整形なしに翻訳処理を開始できる。

#### Acceptance Criteria

1. WHEN Excelファイルが読み込まれる THEN THE Excel_Parser SHALL 全シートを読み込み、シートごとに処理する
2. WHEN Excelファイルが読み込まれる THEN THE Excel_Parser SHALL ヘッダー行の有無を自動検出する
3. WHEN ヘッダー行が存在しない場合（B000.xlsxのように） THEN THE Excel_Parser SHALL 列位置ベースでカラムをマッピングする（A列=text_id, B列=character, C列=source_text）
4. WHEN カラム名が存在する場合 THEN THE Excel_Parser SHALL エイリアス辞書（原文/テキスト/text/japanese等）で正規カラム名にマッピングする
5. WHEN セル結合が検出される THEN THE Excel_Parser SHALL 結合範囲全体に同じ値を展開する
6. WHEN 空行・セクション見出し行が検出される THEN THE Excel_Parser SHALL これらを除外しログに記録する
7. WHEN text_idカラムが無いまたは空の場合 THEN THE Excel_Parser SHALL ファイル名+行番号で自動生成する
8. WHEN キャラ名に末尾スペースがある場合 THEN THE Excel_Parser SHALL トリムして正規化する
9. THE Excel_Parser SHALL 前処理結果のサマリ（カラムマッピング、シート数、行数、スキップ行数）をユーザーに確認させる
10. WHEN 複数シートが存在する場合 THEN THE Excel_Parser SHALL シート名を識別子として保持し、出力時にシート単位で分離可能とする
11. WHEN 翻訳不要行（既に翻訳済み、空テキスト、変数のみ）が検出される THEN THE Excel_Parser SHALL スキップ対象としてフラグ付与する

### Requirement 2: 制御文字・タグの保護

**User Story:** As a ユーザー, I want to テキスト内の変数・制御タグを翻訳から保護する, so that ゲームエンジンの制御コードが破壊されない。

#### Acceptance Criteria

1. WHEN テキスト内に変数（{player_name}等）が含まれる THEN THE Excel_Parser SHALL プレースホルダー（<<VAR_N>>）に置換する
2. WHEN テキスト内に制御タグ（\n, <br>, [ruby]等）が含まれる THEN THE Excel_Parser SHALL プレースホルダー（<<TAG_N>>）に置換する
3. WHEN テキスト内にゲーム固有マーカー（(heart)等）が含まれる THEN THE Excel_Parser SHALL プレースホルダーに置換する
4. WHEN 翻訳が完了する THEN THE Excel_Parser SHALL プレースホルダーを元の値に復元する
5. THE Excel_Parser SHALL 復元後にプレースホルダーの欠落・重複がないか検証する

### Requirement 3: 用語集（グロッサリー）管理

**User Story:** As a ユーザー, I want to 固有名詞・専門用語の訳語を一貫させる, so that ゲーム全体で統一された翻訳を得られる。

#### Acceptance Criteria

1. THE Glossary_Manager SHALL 用語集シート（term_source, term_target, category, context_note, do_not_translate）を読み込む
2. WHEN 翻訳前に原文をスキャンする THEN THE Glossary_Manager SHALL 用語集のterm_sourceと照合しマッチした用語を抽出する
3. WHEN プロンプトを構築する THEN THE Prompt_Builder SHALL マッチした用語のみを制約として注入する（全件送らない）
4. WHEN 翻訳が完了する THEN THE QA_Checker SHALL term_targetが正しく使用されているか自動検証する
5. IF 用語の不一致が検出される THEN THE QA_Checker SHALL glossary_checkカラムに⚠️と詳細を記録する

### Requirement 4: キャラクタープロファイル管理

**User Story:** As a ユーザー, I want to 各キャラクターの口調・性格を定義する, so that 一貫したキャラクター性のある翻訳を得られる。

#### Acceptance Criteria

1. THE Character_Profile_Manager SHALL キャラクターシート（character_id, name_source, name_target, personality, speech_style, speech_examples, first_person, second_person, age, gender, role）を読み込む
2. WHEN プロンプトを構築する THEN THE Prompt_Builder SHALL チャンク内に登場するキャラクターのプロファイルのみを注入する
3. THE Prompt_Builder SHALL キャラクターのspeech_examples（Few-shot例文2〜3件）を必ず含める
4. WHEN 発話キャラが空欄（地の文）の場合 THEN THE Prompt_Builder SHALL 地の文用の翻訳指示に切り替える

### Requirement 5: プロンプト設計と品質制御

**User Story:** As a ユーザー, I want to 違和感のない自然な翻訳を得る, so that そのまま使用できる品質の翻訳を得られる。

#### Acceptance Criteria

1. THE Prompt_Builder SHALL System Prompt（翻訳者ロール、世界観、翻訳ルール、出力フォーマット）をハードコードする
2. THE Prompt_Builder SHALL APIパラメータ（temperature=0等）を固定値で設定しユーザーによる変更を禁止する
3. WHEN プロンプトを構築する THEN THE Prompt_Builder SHALL 用語集制約、キャラプロファイル、前後文脈を動的に注入する
4. THE Prompt_Builder SHALL 出力をJSON形式（translated_text, remarks, alternative）で要求する
5. THE Prompt_Builder SHALL remarksは原則空文字とし、書いてよいケース（言葉遊び再現不能、文脈不足、造語、原文誤り疑い、文字数制限不可）を明示的に指示する

### Requirement 6: マルチパス翻訳パイプライン

**User Story:** As a ユーザー, I want to 翻訳後に自動レビュー・修正を行う, so that 高品質な翻訳を得られる。

#### Acceptance Criteria

1. THE Translation_Service SHALL 1st pass（翻訳生成）を実行する
2. WHEN 2nd passが有効な場合 THEN THE Review_Pipeline SHALL セルフレビュー（レビュアーロールで再評価・修正）を実行する
3. THE Review_Pipeline SHALL 2nd passで修正が必要な行のみを返し、不要な行は含めない
4. WHEN 3rd passが有効な場合 THEN THE Review_Pipeline SHALL 一貫性チェック（ファイル全体の表記揺れ・文体統一）を実行する
5. THE Review_Pipeline SHALL 3rd passでルールベースチェック（正規表現）を先に実行し、AIチェックは必要な場合のみ実行する
6. WHEN 4th passが有効な場合 THEN THE Review_Pipeline SHALL バックトランスレーション検証（翻訳結果を原文言語に再翻訳し比較）を実行する
7. THE Review_Pipeline SHALL 4th passは修正された行・曖昧な行のみを対象とする
8. THE Review_Pipeline SHALL 各パスの修正理由を記録する

### Requirement 7: 品質モード選択

**User Story:** As a ユーザー, I want to プロジェクトの重要度・予算に応じてレビュー深度を選ぶ, so that コストと品質のバランスを取れる。

#### Acceptance Criteria

1. THE Translation_Service SHALL Draft（1st passのみ）、Standard（1st+2nd+3rd pass）、Thorough（全パス）の3モードをサポートする
2. WHEN Draftモードが選択される THEN THE Translation_Service SHALL 1st passのみを実行する
3. WHEN Standardモードが選択される THEN THE Translation_Service SHALL 1st + 2nd + 3rd passを実行する
4. WHEN Thoroughモードが選択される THEN THE Translation_Service SHALL 全パス（1st〜4th）を実行する
5. THE Translation_Service SHALL デフォルトでStandardモードを使用する

### Requirement 8: 大規模ファイルのチャンク分割処理

**User Story:** As a ユーザー, I want to 大規模なファイルでも処理できる, so that ファイルサイズの制限なく翻訳できる。

#### Acceptance Criteria

1. THE Chunk_Processor SHALL ファイル単位でグループ化し、意味的なまとまりを最大限維持する
2. WHEN チャンクサイズが上限（原文2,000トークン目安）を超える THEN THE Chunk_Processor SHALL ファイル内で分割する
3. WHEN ファイル内で分割する場合 THEN THE Chunk_Processor SHALL 前チャンク末尾2〜3行を次チャンクの冒頭に文脈参照として含める
4. WHEN 全チャンクの処理が完了する THEN THE Chunk_Processor SHALL text_idをキーに結果を元の順序で再統合する
5. THE Chunk_Processor SHALL マージ時に重複・欠落がないかカウントチェックする
6. THE Chunk_Processor SHALL チャンク単位で結果を都度ディスクに保存し、途中クラッシュ時も翻訳済み分を保持する

### Requirement 9: トークン量の最適化

**User Story:** As a ユーザー, I want to APIコストを最小化する, so that 大量のテキストを効率的に翻訳できる。

#### Acceptance Criteria

1. THE Prompt_Builder SHALL 用語集を動的フィルタリングし、チャンク内でマッチした用語のみを含める
2. THE Prompt_Builder SHALL キャラプロファイルをチャンク内の登場キャラのみに絞る
3. THE Translation_Service SHALL 送信前にトークン数を計測し、上限超過ならチャンクを自動再分割する
4. THE Translation_Service SHALL 推定コストが設定上限を超えたら処理を一時停止しユーザーに確認を求める
5. THE Translation_Service SHALL Claude APIのPrompt Cachingを活用し、System Prompt部分のコストを削減する

### Requirement 10: コスト追跡とレポート

**User Story:** As a ユーザー, I want to パスごとのコストと効果を把握する, so that 品質設定の判断ができる。

#### Acceptance Criteria

1. THE Cost_Tracker SHALL 各パスの入力トークン数、出力トークン数、キャッシュヒット数を記録する
2. THE Cost_Tracker SHALL 各パスのAPI呼び出し回数、処理時間、料金を計算する
3. THE Cost_Tracker SHALL 各パスの修正行数、修正率を記録する
4. WHEN 出力Excelを生成する THEN THE Excel_Writer SHALL コストレポートシートを含める
5. WHEN 複数ファイルを処理した場合 THEN THE Cost_Tracker SHALL プロジェクト全体のダッシュボードを生成する

### Requirement 11: モード間比較機能

**User Story:** As a ユーザー, I want to 異なる品質モードの結果を比較する, so that 最適なモードを選択できる。

#### Acceptance Criteria

1. THE Translation_Service SHALL 同一ファイルを異なるモードで処理した結果をキャッシュする
2. WHEN 比較が要求される THEN THE Excel_Writer SHALL モード間比較シート（Draft/Standard/Thorough列）を生成する
3. THE Excel_Writer SHALL 各モード間の差分（変更有無、変更理由）を記録する
4. THE Cost_Tracker SHALL モード間のコスト比較（追加コスト、修正率）を計算する

### Requirement 12: 出力Excel生成

**User Story:** As a ユーザー, I want to 翻訳結果をExcel形式で受け取る, so that 既存のワークフローに統合できる。

#### Acceptance Criteria

1. THE Excel_Writer SHALL メイン翻訳結果シート（text_id, source_text, translated_text, character_id, char_count, length_ok, glossary_check, remarks, alternative）を生成する
2. THE Excel_Writer SHALL パス別比較シート（pass_1, pass_2, pass_2_reason, pass_3, pass_3_reason, pass_4_backtrans, final）を生成する
3. THE Excel_Writer SHALL コストレポートシート（パスごとのトークン数、料金、修正率）を生成する
4. WHEN 文字数制限が指定されている場合 THEN THE Excel_Writer SHALL length_okカラムに✅または⚠️を記録する
5. THE Excel_Writer SHALL remarksカラムは原則空欄とし、AIが判断不能な場合のみ記入する
6. WHEN 複数ファイルを処理した場合 THEN THE Excel_Writer SHALL ファイル単位または統合した1つのExcelとして出力する
7. THE Excel_Writer SHALL 条件付き書式で変更があったセルに色を付ける（2nd pass: 青、3rd pass: 緑、4th pass: 黄）

### Requirement 13: エラーハンドリングとリトライ

**User Story:** As a ユーザー, I want to API障害時も処理が継続される, so that 安定した翻訳処理を実行できる。

#### Acceptance Criteria

1. IF 翻訳APIがエラーを返す THEN THE Translation_Service SHALL 指数バックオフでリトライする
2. WHEN リトライ上限（デフォルト3回）に達する THEN THE Translation_Service SHALL エラーを記録し次の行に進む
3. IF APIレート制限（429エラー）に達する THEN THE Translation_Service SHALL RPM/TPMの上限に合わせて動的に待機する
4. THE Translation_Service SHALL 処理完了後にエラーが発生した行のサマリーを出力する

### Requirement 14: API設定と認証

**User Story:** As a 開発者, I want to APIキーを安全に管理する, so that セキュアにAPIを利用できる。

#### Acceptance Criteria

1. THE Translation_Service SHALL 環境変数からAPIキー（ANTHROPIC_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY）を読み込む
2. IF APIキーが未設定の場合 THEN THE Translation_Service SHALL 明確なエラーメッセージを表示する
3. THE Translation_Service SHALL APIエンドポイントを設定ファイルで管理する

### Requirement 15: 複数LLMの比較機能

**User Story:** As a ユーザー, I want to 複数のLLMの翻訳結果を比較する, so that 最適な翻訳結果を選択できる。

#### Acceptance Criteria

1. THE Translation_Service SHALL Claude、Gemini、GPT-4の3つのLLMをサポートする
2. THE Translation_Service SHALL デフォルトでClaudeを使用する
3. WHEN 比較モードが有効な場合 THEN THE Translation_Service SHALL 選択されたLLMの翻訳結果を並列で取得する
4. WHEN 比較結果を出力する THEN THE Excel_Writer SHALL 各LLMの結果を別列として出力する
5. THE Translation_Service SHALL 各LLMのレスポンス時間を記録する
