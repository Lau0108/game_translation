# Implementation Plan: Excel Translation API

## Overview

ゲームテキスト翻訳システムをPythonで実装する。マルチパス翻訳パイプライン（1st pass翻訳 → 2nd passセルフレビュー → 3rd pass一貫性チェック → 4th passバックトランスレーション）、パス別比較・コストレポート、品質モード選択を備える。

## Tasks

- [x] 1. プロジェクト構造とコア設定
  - [x] 1.1 プロジェクトディレクトリ構造を作成
    - src/excel_translator/ 配下にモジュールを配置
    - requirements.txt, pyproject.toml を作成
    - _Requirements: 14.3_
  - [x] 1.2 設定管理モジュール（config.py）を実装
    - TranslationConfig dataclass を定義
    - 環境変数からAPIキーを読み込む機能
    - 設定ファイル（settings.yaml）の読み込み
    - 品質モード（draft/standard/thorough）の定義
    - _Requirements: 14.1, 14.2, 14.3, 7.1_
  - [x] 1.3 設定読み込みのユニットテストを作成
    - 環境変数未設定時のエラーメッセージ検証
    - _Requirements: 14.2_

- [x] 2. Excel解析・前処理モジュール
  - [x] 2.1 ExcelParser クラスを実装
    - read_excel: 全シート読み込み
    - detect_header: ヘッダー行自動検出
    - map_columns: エイリアス辞書によるカラムマッピング
    - normalize: セル結合解除、空行除外、text_id生成、キャラ名トリム
    - detect_skip_rows: 翻訳不要行のフラグ付与
    - _Requirements: 1.1-1.11_
  - [x] 2.2 Property 1: 複数シート読み込みプロパティテスト
    - **Property 1: 複数シート読み込み**
    - **Validates: Requirements 1.1, 1.10**
  - [x] 2.3 Property 2: カラムマッピングプロパティテスト
    - **Property 2: カラムマッピング**
    - **Validates: Requirements 1.2, 1.3, 1.4**
  - [x] 2.4 Property 5: text_id自動生成プロパティテスト
    - **Property 5: text_id自動生成**
    - **Validates: Requirements 1.7**
  - [x] 2.5 Property 7: 翻訳不要行フラグプロパティテスト
    - **Property 7: 翻訳不要行フラグ**
    - **Validates: Requirements 1.11**

- [x] 3. タグ保護・復元モジュール（オプション機能）
  - [x] 3.1 protect_tags / restore_tags 関数を実装
    - 変数（{...}）、制御タグ（\n, <br>等）、ゲーム固有マーカー（(heart)等）の検出
    - プレースホルダー置換・復元ロジック
    - _Requirements: 2.1-2.5_
  - [x] 3.2 Property 8: タグ保護プロパティテスト
    - **Property 8: タグ保護**
    - **Validates: Requirements 2.1, 2.2, 2.3**
  - [x] 3.3 Property 9: タグ復元ラウンドトリッププロパティテスト
    - **Property 9: タグ復元ラウンドトリップ**
    - **Validates: Requirements 2.4, 2.5**

- [x] 4. Checkpoint - Excel解析モジュール完了確認
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. 用語集管理モジュール（オプション機能）
  - [x] 5.1 GlossaryManager クラスを実装
    - load: 用語集シート読み込み
    - filter_by_text: テキスト内マッチ用語抽出
    - verify_usage: 翻訳結果の用語一貫性検証
    - _Requirements: 3.1-3.5_
  - [x] 5.2 Property 10: 用語集動的フィルタリングプロパティテスト
    - **Property 10: 用語集動的フィルタリング**
    - **Validates: Requirements 3.2, 3.3**
  - [x] 5.3 Property 11: 用語一貫性検証プロパティテスト
    - **Property 11: 用語一貫性検証**
    - **Validates: Requirements 3.4, 3.5**

- [x] 6. キャラクタープロファイル管理モジュール（オプション機能）
  - [x] 6.1 CharacterProfileManager クラスを実装
    - load: キャラクターシート読み込み（age, gender, role含む）
    - get_profiles_for_chunk: チャンク内登場キャラのプロファイル取得
    - _Requirements: 4.1-4.3_
  - [x] 6.2 Property 12: キャラプロファイル動的注入プロパティテスト
    - **Property 12: キャラプロファイル動的注入**
    - **Validates: Requirements 4.2, 4.3**

- [x] 7. プロンプト構築モジュール
  - [x] 7.1 PromptBuilder クラスを実装
    - build_system_prompt: 固定System Prompt生成
    - build_translation_prompt: 1st pass翻訳用プロンプト
    - build_review_prompt: 2nd passセルフレビュー用プロンプト（レビュアーロール）
    - build_consistency_prompt: 3rd pass一貫性チェック用プロンプト
    - build_backtrans_prompt: 4th passバックトランスレーション用プロンプト
    - 固定パラメータ（temperature=0等）の定義
    - remarks運用ルールの指示
    - _Requirements: 5.1-5.5, 6.2, 6.4, 6.6_
  - [x] 7.2 Property 13: 地の文/台詞切替プロパティテスト
    - **Property 13: 地の文/台詞切替**
    - **Validates: Requirements 4.4**

- [x] 8. Checkpoint - コア機能モジュール完了確認
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. チャンク分割・マージモジュール
  - [x] 9.1 ChunkProcessor クラスを実装
    - split_into_chunks: ファイル単位グループ化、トークン上限分割、文脈参照追加
    - merge_results: text_id順マージ、重複・欠落チェック
    - save_progress / load_progress: 進捗永続化
    - _Requirements: 8.1-8.6_
  - [x] 9.2 Property 19: チャンク分割プロパティテスト
    - **Property 19: チャンク分割**
    - **Validates: Requirements 8.1, 8.2, 8.3**
  - [x] 9.3 Property 20: チャンクマージプロパティテスト
    - **Property 20: チャンクマージ**
    - **Validates: Requirements 8.4, 8.5**
  - [x] 9.4 Property 21: 進捗永続化ラウンドトリッププロパティテスト
    - **Property 21: 進捗永続化ラウンドトリップ**
    - **Validates: Requirements 8.6**

- [x] 10. トークン計測モジュール
  - [x] 10.1 トークン計測・再分割機能を実装
    - tiktokenを使用したトークン数計測
    - 上限超過時の自動再分割
    - コスト推定・上限チェック
    - Prompt Caching対応
    - _Requirements: 9.1-9.5_
  - [x] 10.2 Property 22: トークン計測・再分割プロパティテスト
    - **Property 22: トークン計測・再分割**
    - **Validates: Requirements 9.3**

- [x] 11. Checkpoint - チャンク処理モジュール完了確認
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. 翻訳サービスモジュール
  - [x] 12.1 TranslationProvider 抽象クラスと各プロバイダーを実装
    - ClaudeProvider: Anthropic API呼び出し
    - GeminiProvider: Google Generative AI API呼び出し
    - GPT4Provider: OpenAI API呼び出し
    - _Requirements: 15.1, 15.2_
  - [x] 12.2 TranslationService クラスを実装
    - run_pipeline: マルチパスパイプライン実行（品質モード対応）
    - translate_with_retry: 指数バックオフリトライ
    - レート制限対応（429エラー時の動的待機）
    - _Requirements: 6.1, 7.2-7.5, 13.1-13.4, 15.3-15.5_
  - [x] 12.3 Property 18: 品質モード実行プロパティテスト
    - **Property 18: 品質モード実行**
    - **Validates: Requirements 7.1-7.5**
  - [x] 12.4 Property 27: リトライプロパティテスト
    - **Property 27: リトライ**
    - **Validates: Requirements 13.1, 13.2**
  - [x] 12.5 Property 29: LLM比較プロパティテスト
    - **Property 29: LLM比較**
    - **Validates: Requirements 15.3, 15.4, 15.5**

- [x] 13. レビューパイプラインモジュール
  - [x] 13.1 ReviewPipeline クラスを実装
    - run_2nd_pass: セルフレビュー（修正が必要な行のみ返す）
    - run_3rd_pass_rules: ルールベース一貫性チェック（正規表現）
    - run_3rd_pass_ai: AI一貫性チェック（文体・ニュアンス系）
    - run_4th_pass: バックトランスレーション検証（対象行のみ）
    - _Requirements: 6.2-6.8_
  - [x] 13.2 Property 14: 2nd passセルフレビュープロパティテスト
    - **Property 14: 2nd passセルフレビュー**
    - **Validates: Requirements 6.2, 6.3**
  - [x] 13.3 Property 15: 3rd pass一貫性チェックプロパティテスト
    - **Property 15: 3rd pass一貫性チェック**
    - **Validates: Requirements 6.4, 6.5**
  - [x] 13.4 Property 16: 4th passバックトランスレーションプロパティテスト
    - **Property 16: 4th passバックトランスレーション**
    - **Validates: Requirements 6.6, 6.7**
  - [x] 13.5 Property 17: パス修正理由記録プロパティテスト
    - **Property 17: パス修正理由記録**
    - **Validates: Requirements 6.8**

- [x] 14. コスト追跡モジュール
  - [x] 14.1 CostTracker クラスを実装
    - track_request: トークン消費記録
    - calculate_cost: 料金計算
    - get_pass_summary: パスごとのサマリ取得
    - generate_report: コストレポート生成
    - generate_mode_comparison: モード間比較レポート生成
    - _Requirements: 10.1-10.5, 11.1-11.4_
  - [x] 14.2 Property 23: コスト追跡プロパティテスト
    - **Property 23: コスト追跡**
    - **Validates: Requirements 10.1, 10.2, 10.3**
  - [x] 14.3 Property 24: コストレポート生成プロパティテスト
    - **Property 24: コストレポート生成**
    - **Validates: Requirements 10.4, 10.5**
  - [x] 14.4 Property 25: モード間比較プロパティテスト
    - **Property 25: モード間比較**
    - **Validates: Requirements 11.1-11.4**

- [x] 15. Checkpoint - 翻訳パイプライン完了確認
  - Ensure all tests pass, ask the user if questions arise.

- [x] 16. QAチェックモジュール
  - [x] 16.1 QAChecker クラスを実装
    - check_glossary_consistency: 用語一貫性チェック
    - check_char_count: 文字数チェック
    - validate_placeholders: プレースホルダー検証
    - _Requirements: 3.4, 3.5, 12.4_

- [x] 17. Excel出力モジュール
  - [x] 17.1 ExcelWriter クラスを実装
    - write_main_sheet: メイン翻訳結果シート出力
    - write_pass_comparison_sheet: パス別比較シート出力
    - write_cost_report_sheet: コストレポートシート出力
    - write_mode_comparison_sheet: モード間比較シート出力
    - apply_conditional_formatting: 条件付き書式適用（2nd: 青、3rd: 緑、4th: 黄）
    - _Requirements: 12.1-12.7_
  - [x] 17.2 Property 26: 出力Excel生成プロパティテスト
    - **Property 26: 出力Excel生成**
    - **Validates: Requirements 12.1-12.7**

- [x] 18. Checkpoint - 全モジュール完了確認
  - Ensure all tests pass, ask the user if questions arise.

- [x] 19. メインコントローラーと統合
  - [x] 19.1 TranslationApp メインクラスを実装
    - run: 全体処理フロー（読み込み→前処理→翻訳パイプライン→QA→出力）
    - ユーザー確認ステップ（前処理サマリ表示）
    - コスト上限チェック・一時停止
    - エラーサマリ出力
    - _Requirements: 1.9, 9.4, 13.4_
  - [x] 19.2 CLIエントリーポイントを実装
    - argparseによるコマンドライン引数処理
    - 入力ファイル/フォルダ、出力ファイル、設定ファイルの指定
    - 品質モード（--mode draft/standard/thorough）
    - LLM比較モード（--compare）
    - _Requirements: 7.1-7.5, 15.3_

- [x] 20. Final Checkpoint - 全体統合テスト
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- All tasks are required for comprehensive implementation
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- マルチパスパイプライン: 1st pass（翻訳）→ 2nd pass（セルフレビュー）→ 3rd pass（一貫性チェック）→ 4th pass（バックトランスレーション）
- 品質モード: Draft（1st passのみ）、Standard（1st+2nd+3rd）、Thorough（全パス）
