"""メインコントローラーモジュール

TranslationAppメインクラスとCLIエントリーポイントを提供する。
- 全体処理フロー（読み込み→前処理→翻訳パイプライン→QA→出力）
- ユーザー確認ステップ（前処理サマリ表示）
- コスト上限チェック・一時停止
- エラーサマリ出力
"""

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..config import (
    ConfigurationError,
    QualityMode,
    TranslationConfig,
    load_config,
)
from ..parser import ExcelParser, ParsedExcel, InputRow, protect_tags, restore_tags
from ..glossary import GlossaryManager, GlossaryEntry
from ..character import CharacterProfileManager, CharacterProfile
from ..prompt import PromptBuilder
from ..chunk import Chunk, ChunkProcessor, ChunkResult
from ..translation import (
    TranslationService,
    PipelineResult,
    create_service_from_config,
    TranslationError,
)
from ..review import ReviewPipeline
from ..cost import CostTracker, CostLimitChecker, TokenCounter
from ..qa import QAChecker
from ..writer import ExcelWriter

logger = logging.getLogger(__name__)






@dataclass
class ProcessingSummary:
    """処理サマリ"""
    total_files: int = 0
    total_rows: int = 0
    translated_rows: int = 0
    skipped_rows: int = 0
    error_rows: int = 0
    total_cost_usd: float = 0.0
    processing_time_ms: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)


class TranslationAppError(Exception):
    """翻訳アプリケーションエラー"""
    pass


class TranslationApp:
    """
    翻訳アプリケーションメインクラス
    
    全体処理フロー:
    1. 設定読み込み
    2. Excel解析・前処理
    3. ユーザー確認（前処理サマリ表示）
    4. チャンク分割
    5. 翻訳パイプライン実行
    6. QAチェック
    7. Excel出力
    8. エラーサマリ出力
    """
    
    def __init__(
        self,
        config: Optional[TranslationConfig] = None,
        settings_path: Optional[str] = None,
    ):
        """
        TranslationAppを初期化
        
        Args:
            config: 翻訳設定（省略時はsettings_pathから読み込み）
            settings_path: 設定ファイルパス
        """
        self.config = config or load_config(settings_path)
        
        # コンポーネント初期化
        self._parser = ExcelParser()
        self._glossary_manager = GlossaryManager()
        self._character_manager = CharacterProfileManager()
        self._prompt_builder = PromptBuilder()
        self._chunk_processor = ChunkProcessor(
            max_tokens_per_chunk=self.config.chunk_size_tokens,
            context_lines=self.config.context_lines,
            progress_dir=self.config.progress_dir,
        )
        self._qa_checker = QAChecker(self._glossary_manager)
        self._writer = ExcelWriter(char_limit=self.config.char_limit)
        
        # 翻訳サービス（遅延初期化）
        self._translation_service: Optional[TranslationService] = None
        
        # コスト追跡（プロバイダーのデフォルトモデルに基づく）
        provider_model_map = {
            "claude": "claude-3-5-sonnet-20241022",
            "gemini": "gemini-2.5-pro",
            "gpt4": "gpt-4o",
        }
        cost_model = provider_model_map.get(self.config.default_provider, "claude-3-5-sonnet-20241022")
        self._cost_tracker = CostTracker(model=cost_model)
        self._cost_checker = CostLimitChecker(
            cost_limit_usd=self.config.cost_limit_usd,
            model=cost_model,
        )
        
        # 処理状態
        self._summary = ProcessingSummary()
        self._glossary_entries: List[GlossaryEntry] = []
        self._character_profiles: Dict[str, CharacterProfile] = {}
        self._is_paused = False
    
    def _get_translation_service(self) -> TranslationService:
        """翻訳サービスを取得（遅延初期化）"""
        if self._translation_service is None:
            self._translation_service = create_service_from_config(self.config)
        return self._translation_service
    
    def load_glossary(self, file_path: str, sheet_name: Optional[str] = None) -> None:
        """
        用語集を読み込む
        
        Args:
            file_path: 用語集ファイルパス
            sheet_name: シート名
        """
        self._glossary_entries = self._glossary_manager.load(file_path, sheet_name)
        logger.info(f"用語集を読み込みました: {len(self._glossary_entries)} 件")
    
    def load_characters(self, file_path: str, sheet_name: Optional[str] = None) -> None:
        """
        キャラクタープロファイルを読み込む
        
        Args:
            file_path: キャラクターファイルパス
            sheet_name: シート名
        """
        self._character_profiles = self._character_manager.load(file_path, sheet_name)
        logger.info(f"キャラクタープロファイルを読み込みました: {len(self._character_profiles)} 件")
    
    def parse_excel(self, file_path: str) -> ParsedExcel:
        """
        Excelファイルを解析
        
        Args:
            file_path: Excelファイルパス
            
        Returns:
            ParsedExcel
        """
        return self._parser.parse(file_path)
    
    def get_preprocessing_summary(self, parsed: ParsedExcel) -> str:
        """
        前処理サマリを取得（ユーザー確認用）
        
        Args:
            parsed: 解析済みExcelデータ
            
        Returns:
            サマリテキスト
        """
        return self._parser.get_summary_text(parsed)
    
    def _estimate_cost(self, rows: List[InputRow]) -> Tuple[float, str]:
        """
        翻訳コストを推定
        
        Args:
            rows: 翻訳対象行
            
        Returns:
            (推定コスト, メッセージ)
        """
        # 翻訳対象行のみ
        translatable = [r for r in rows if not r.skip]
        
        # トークン数を推定
        token_counter = TokenCounter()
        total_tokens = 0
        for row in translatable:
            total_tokens += token_counter.count_tokens(row.source_text)
            if row.character:
                total_tokens += token_counter.count_tokens(row.character)
        
        # コストを推定
        estimate, can_continue, message = self._cost_checker.estimate_and_check(
            input_tokens=total_tokens,
            estimated_output_ratio=1.2,
            use_cache=True,
        )
        
        return estimate.total_usd, message

    async def run(
        self,
        input_path: str,
        output_path: str,
        glossary_path: Optional[str] = None,
        character_path: Optional[str] = None,
        mode: Optional[str] = None,
        skip_confirmation: bool = False,
    ) -> ProcessingSummary:
        """
        翻訳処理を実行
        
        Args:
            input_path: 入力ファイル/フォルダパス
            output_path: 出力ファイルパス
            glossary_path: 用語集ファイルパス（オプション）
            character_path: キャラクターファイルパス（オプション）
            mode: 品質モード（省略時は設定ファイルの値）
            skip_confirmation: ユーザー確認をスキップするか
            
        Returns:
            ProcessingSummary
        """
        import time
        start_time = time.time()
        
        quality_mode = mode or self.config.quality_mode
        
        logger.info(f"翻訳処理を開始: mode={quality_mode}")
        
        try:
            # 1. 用語集・キャラクター読み込み
            if glossary_path:
                self.load_glossary(glossary_path)
            
            if character_path:
                self.load_characters(character_path)
            
            # 2. 入力ファイル解析
            input_files = self._get_input_files(input_path)
            self._summary.total_files = len(input_files)
            
            all_results: List[Dict[str, Any]] = []
            all_pass_results: List[Dict[str, Any]] = []
            
            for file_path in input_files:
                logger.info(f"処理中: {file_path}")
                
                # Excel解析
                parsed = self.parse_excel(file_path)
                self._summary.total_rows += parsed.summary.total_rows
                self._summary.skipped_rows += parsed.summary.skipped_rows
                
                # 3. ユーザー確認（前処理サマリ表示）
                if not skip_confirmation:
                    summary_text = self.get_preprocessing_summary(parsed)
                    print("\n" + summary_text)
                    
                    # コスト推定
                    estimated_cost, cost_message = self._estimate_cost(parsed.rows)
                    print(f"\n推定コスト: ${estimated_cost:.4f}")
                    print(f"コストチェック: {cost_message}")
                    
                    # 確認プロンプト
                    response = input("\n処理を続行しますか？ (y/n): ")
                    if response.lower() not in ["y", "yes", "はい"]:
                        logger.info("ユーザーによりキャンセルされました")
                        return self._summary
                
                # 4. チャンク分割
                chunks = self._chunk_processor.split_into_chunks(parsed.rows)
                
                # 5. 保存済み進捗を読み込み
                progress = self._chunk_processor.load_progress()
                pending_chunks = self._chunk_processor.get_pending_chunks(chunks, progress)
                
                # 6. 翻訳パイプライン実行
                file_results, file_pass_results = await self._process_chunks(
                    chunks=pending_chunks,
                    progress=progress,
                    quality_mode=quality_mode,
                )
                
                all_results.extend(file_results)
                all_pass_results.extend(file_pass_results)
            
            # 7. QAチェック
            if all_results:
                glossary_results, char_results, placeholder_results, qa_summary = \
                    self._qa_checker.run_all_checks(
                        rows=all_results,
                        glossary_entries=self._glossary_entries,
                        char_limit=self.config.char_limit,
                    )
                
                # QA結果をマージ
                qa_formatted = self._qa_checker.format_results_for_excel(
                    glossary_results, char_results, placeholder_results
                )
                for i, result in enumerate(all_results):
                    if i < len(qa_formatted):
                        result.update(qa_formatted[i])
            
            # 8. Excel出力
            if all_results:
                cost_report = self._cost_tracker.generate_report()
                
                self._writer.write_all_sheets(
                    results=all_results,
                    pass_results=all_pass_results if all_pass_results else None,
                    cost_report=cost_report,
                    output_path=output_path,
                )
                
                logger.info(f"出力完了: {output_path}")
            
            # 9. サマリ更新
            self._summary.translated_rows = len(all_results)
            self._summary.total_cost_usd = self._cost_tracker.get_total_cost()
            self._summary.processing_time_ms = int((time.time() - start_time) * 1000)
            
            # 10. エラーサマリ出力
            self._print_error_summary()
            
            return self._summary
            
        except Exception as e:
            logger.error(f"翻訳処理中にエラーが発生しました: {e}")
            self._summary.errors.append({
                "type": type(e).__name__,
                "message": str(e),
            })
            raise TranslationAppError(f"翻訳処理に失敗しました: {e}")
    
    def _get_input_files(self, input_path: str) -> List[str]:
        """
        入力ファイルリストを取得
        
        Args:
            input_path: 入力ファイル/フォルダパス
            
        Returns:
            ファイルパスのリスト
        """
        path = Path(input_path)
        
        if path.is_file():
            return [str(path)]
        
        if path.is_dir():
            files = []
            for ext in [".xlsx", ".xls", ".xlsm"]:
                files.extend(path.glob(f"*{ext}"))
            return [str(f) for f in sorted(files)]
        
        raise TranslationAppError(f"入力パスが見つかりません: {input_path}")
    
    async def _process_chunks(
        self,
        chunks: List[Chunk],
        progress: Dict[str, ChunkResult],
        quality_mode: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        チャンクを処理
        
        Args:
            chunks: 処理対象チャンク
            progress: 保存済み進捗
            quality_mode: 品質モード
            
        Returns:
            (翻訳結果リスト, パス別結果リスト)
        """
        service = self._get_translation_service()
        all_results: List[Dict[str, Any]] = []
        all_pass_results: List[Dict[str, Any]] = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"チャンク処理中: {i + 1}/{len(chunks)} ({chunk.chunk_id})")
            
            # コスト上限チェック
            if self._is_paused:
                logger.warning("コスト上限により処理を一時停止中")
                break
            
            # チャンク内のキャラクターを抽出
            character_ids = {r.character for r in chunk.rows if r.character}
            chunk_profiles = self._character_manager.get_profiles_for_chunk(
                character_ids, self._character_profiles
            )
            
            # チャンク内のテキストにマッチする用語を抽出
            chunk_text = " ".join(r.source_text for r in chunk.rows)
            chunk_glossary = self._glossary_manager.filter_by_text(
                chunk_text, self._glossary_entries
            )
            
            # タグ保護処理
            placeholder_maps = {}
            original_texts = {}
            for row in chunk.rows:
                original_texts[row.text_id] = row.source_text
                protected_text, p_map = protect_tags(row.source_text)
                row.source_text = protected_text
                placeholder_maps[row.text_id] = p_map
                
            try:
                # 翻訳パイプライン実行
                pipeline_results, stats = await service.run_pipeline(
                    chunk=chunk,
                    prompt_builder=self._prompt_builder,
                    mode=quality_mode,
                    glossary_entries=chunk_glossary,
                    character_profiles=chunk_profiles,
                )
                
                # コスト追跡
                pass_stats = stats.get("pass_stats", {})
                for pass_num in stats.get("passes_executed", []):
                    pass_name = f"{pass_num}st_pass" if pass_num == 1 else f"{pass_num}nd_pass" if pass_num == 2 else f"{pass_num}rd_pass" if pass_num == 3 else f"{pass_num}th_pass"
                    p_stat = pass_stats.get(pass_num, {})
                    if p_stat:
                        self._cost_tracker.track_request(
                            pass_name=pass_name,
                            input_tokens=p_stat.get("input_tokens", 0),
                            output_tokens=p_stat.get("output_tokens", 0),
                            cache_hit_tokens=p_stat.get("cache_read_tokens", 0),
                            processing_time_ms=p_stat.get("response_time_ms", 0),
                        )
                        # 修正行数を記録 (TASK-09)
                        if "modified_rows" in p_stat:
                            self._cost_tracker.record_modifications(
                                pass_name=pass_name,
                                modified_rows=p_stat["modified_rows"],
                                total_rows=p_stat.get("total_rows", len(chunk.rows)),
                            )
                
                # コスト上限チェック
                current_cost = self._cost_tracker.get_total_cost()
                can_continue, message = self._cost_checker.check_limit(0)
                if not can_continue:
                    logger.warning(f"コスト上限に達しました: {message}")
                    self._is_paused = True
                    
                    # ユーザー確認
                    print(f"\n⚠️ {message}")
                    response = input("処理を続行しますか？ (y/n): ")
                    if response.lower() in ["y", "yes", "はい"]:
                        self._is_paused = False
                    else:
                        break
                
                # 結果を変換
                for result in pipeline_results:
                    text_id = result.text_id
                    p_map = placeholder_maps.get(text_id, {})
                    
                    # 原文を復元
                    if text_id in original_texts:
                        result.source_text = original_texts[text_id]
                    
                    # 翻訳文のタグを復元
                    final_text = restore_tags(result.final, p_map) if result.final else result.final
                    pass_1_text = restore_tags(result.pass_1, p_map) if result.pass_1 else result.pass_1
                    pass_2_text = restore_tags(result.pass_2, p_map) if result.pass_2 else result.pass_2
                    pass_3_text = restore_tags(result.pass_3, p_map) if result.pass_3 else result.pass_3
                    
                    result_dict = {
                        "text_id": text_id,
                        "source_text": result.source_text,
                        "translated_text": final_text,
                        "character_id": result.character_id,
                        "provider": result.provider,
                        "remarks": result.remarks,
                        "alternative": restore_tags(result.alternative, p_map) if result.alternative else result.alternative,
                    }
                    all_results.append(result_dict)
                    
                    pass_dict = {
                        "text_id": text_id,
                        "source_text": result.source_text,
                        "pass_1": pass_1_text,
                        "pass_2": pass_2_text,
                        "pass_2_reason": result.pass_2_reason,
                        "pass_3": pass_3_text,
                        "pass_3_reason": result.pass_3_reason,
                        "pass_4_backtrans": result.pass_4_backtrans,
                        "final": final_text,
                        "remarks": result.remarks,
                    }
                    all_pass_results.append(pass_dict)
                    
                # chunk.rowsの原文も元に戻す
                for row in chunk.rows:
                    if row.text_id in original_texts:
                        row.source_text = original_texts[row.text_id]
                
                # 進捗保存
                chunk_result = ChunkResult(
                    chunk_id=chunk.chunk_id,
                    results=[r for r in all_results if any(
                        row.text_id == r["text_id"] for row in chunk.rows
                    )],
                )
                self._chunk_processor.save_progress(chunk, chunk_result)
                
            except TranslationError as e:
                logger.error(f"チャンク処理エラー: {chunk.chunk_id}, {e}")
                self._summary.errors.append({
                    "chunk_id": chunk.chunk_id,
                    "type": type(e).__name__,
                    "message": str(e),
                })
                self._summary.error_rows += len(chunk.rows)
        
        return all_results, all_pass_results
    
    def _print_error_summary(self) -> None:
        """エラーサマリを出力"""
        if not self._summary.errors:
            logger.info("エラーなしで処理が完了しました")
            return
        
        print("\n=== エラーサマリ ===")
        print(f"エラー件数: {len(self._summary.errors)}")
        
        for i, error in enumerate(self._summary.errors[:10], 1):
            print(f"{i}. [{error.get('type', 'Unknown')}] {error.get('message', 'No message')}")
            if "chunk_id" in error:
                print(f"   チャンク: {error['chunk_id']}")
        
        if len(self._summary.errors) > 10:
            print(f"... 他 {len(self._summary.errors) - 10} 件")

    async def run_comparison(
        self,
        input_path: str,
        output_path: str,
        provider_names: List[str],
        glossary_path: Optional[str] = None,
        character_path: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> ProcessingSummary:
        """
        LLM比較モードで翻訳を実行
        
        Args:
            input_path: 入力ファイルパス
            output_path: 出力ファイルパス
            provider_names: 比較するプロバイダー名リスト
            glossary_path: 用語集ファイルパス
            character_path: キャラクターファイルパス
            mode: 品質モード
            
        Returns:
            ProcessingSummary
        """
        import time
        start_time = time.time()
        
        quality_mode = mode or self.config.quality_mode
        
        logger.info(f"LLM比較モードで翻訳を開始: providers={provider_names}, mode={quality_mode}")
        
        try:
            # 用語集・キャラクター読み込み
            if glossary_path:
                self.load_glossary(glossary_path)
            
            if character_path:
                self.load_characters(character_path)
            
            # 入力ファイル解析
            parsed = self.parse_excel(input_path)
            self._summary.total_files = 1
            self._summary.total_rows = parsed.summary.total_rows
            self._summary.skipped_rows = parsed.summary.skipped_rows
            
            # チャンク分割
            chunks = self._chunk_processor.split_into_chunks(parsed.rows)
            
            # 翻訳サービス取得
            service = self._get_translation_service()
            
            # 各プロバイダーで翻訳
            comparison_results: Dict[str, List[Dict[str, Any]]] = {
                provider: [] for provider in provider_names
            }
            
            for chunk in chunks:
                # チャンク内のキャラクターを抽出
                character_ids = {r.character for r in chunk.rows if r.character}
                chunk_profiles = self._character_manager.get_profiles_for_chunk(
                    character_ids, self._character_profiles
                )
                
                # チャンク内のテキストにマッチする用語を抽出
                chunk_text = " ".join(r.source_text for r in chunk.rows)
                chunk_glossary = self._glossary_manager.filter_by_text(
                    chunk_text, self._glossary_entries
                )
                
                # 並列翻訳
                provider_results = await service.run_comparison(
                    chunk=chunk,
                    prompt_builder=self._prompt_builder,
                    provider_names=provider_names,
                    mode=quality_mode,
                    glossary_entries=chunk_glossary,
                    character_profiles=chunk_profiles,
                )
                
                # 結果を収集
                for provider_name, (results, stats) in provider_results.items():
                    for result in results:
                        comparison_results[provider_name].append({
                            "text_id": result.text_id,
                            "source_text": result.source_text,
                            "translated_text": result.final,
                        })
            
            # モード間比較レポート生成
            mode_comparison = self._cost_tracker.generate_mode_comparison(comparison_results)
            
            # Excel出力
            # メイン結果は最初のプロバイダーの結果を使用
            main_results = comparison_results.get(provider_names[0], [])
            
            self._writer.write_main_sheet(main_results, output_path)
            self._writer.write_mode_comparison_sheet(mode_comparison, output_path, "LLM比較")
            
            logger.info(f"比較結果を出力: {output_path}")
            
            # サマリ更新
            self._summary.translated_rows = len(main_results)
            self._summary.total_cost_usd = self._cost_tracker.get_total_cost()
            self._summary.processing_time_ms = int((time.time() - start_time) * 1000)
            
            return self._summary
            
        except Exception as e:
            logger.error(f"LLM比較処理中にエラーが発生しました: {e}")
            self._summary.errors.append({
                "type": type(e).__name__,
                "message": str(e),
            })
            raise TranslationAppError(f"LLM比較処理に失敗しました: {e}")
    
    def get_summary(self) -> ProcessingSummary:
        """処理サマリを取得"""
        return self._summary
    
    def reset(self) -> None:
        """状態をリセット"""
        self._summary = ProcessingSummary()
        self._cost_tracker.reset()
        self._cost_checker.reset()
        self._is_paused = False
        self._chunk_processor.clear_progress()


def create_argument_parser() -> argparse.ArgumentParser:
    """
    コマンドライン引数パーサーを作成
    
    Returns:
        ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="excel-translator",
        description="ゲームテキスト翻訳システム - ExcelファイルをLLM APIで翻訳",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な翻訳
  excel-translator input.xlsx -o output.xlsx

  # 用語集・キャラクター設定を使用
  excel-translator input.xlsx -o output.xlsx -g glossary.xlsx -c characters.xlsx

  # 品質モードを指定
  excel-translator input.xlsx -o output.xlsx --mode thorough

  # LLM比較モード
  excel-translator input.xlsx -o output.xlsx --compare claude gemini gpt4

品質モード:
  draft     - 1st passのみ（高速・低コスト）
  standard  - 1st + 2nd + 3rd pass（デフォルト）
  thorough  - 全パス（高品質・高コスト）
        """,
    )
    
    # 必須引数
    parser.add_argument(
        "input",
        help="入力Excelファイルまたはフォルダのパス",
    )
    
    # 出力オプション
    parser.add_argument(
        "-o", "--output",
        default="output.xlsx",
        help="出力Excelファイルのパス（デフォルト: output.xlsx）",
    )
    
    # 設定ファイル
    parser.add_argument(
        "-s", "--settings",
        help="設定ファイル（settings.yaml）のパス",
    )
    
    # 用語集
    parser.add_argument(
        "-g", "--glossary",
        help="用語集Excelファイルのパス",
    )
    
    # キャラクター設定
    parser.add_argument(
        "-c", "--characters",
        help="キャラクター設定Excelファイルのパス",
    )
    
    # 品質モード
    parser.add_argument(
        "-m", "--mode",
        choices=["draft", "standard", "thorough"],
        default=None,
        help="品質モード（draft/standard/thorough）",
    )
    
    # LLM比較モード
    parser.add_argument(
        "--compare",
        nargs="+",
        choices=["claude", "gemini", "gpt4"],
        help="LLM比較モード（複数のプロバイダーを指定）",
    )
    
    # 確認スキップ
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="確認プロンプトをスキップ",
    )
    
    # ログレベル
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="詳細ログを出力",
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="ログ出力を最小限に",
    )
    
    return parser


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """
    ロギングを設定
    
    Args:
        verbose: 詳細ログを出力するか
        quiet: ログ出力を最小限にするか
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


async def async_main(args: argparse.Namespace) -> int:
    """
    非同期メイン処理
    
    Args:
        args: コマンドライン引数
        
    Returns:
        終了コード
    """
    try:
        # アプリケーション初期化
        app = TranslationApp(settings_path=args.settings)
        
        # LLM比較モード
        if args.compare:
            summary = await app.run_comparison(
                input_path=args.input,
                output_path=args.output,
                provider_names=args.compare,
                glossary_path=args.glossary,
                character_path=args.characters,
                mode=args.mode,
            )
        else:
            # 通常モード
            summary = await app.run(
                input_path=args.input,
                output_path=args.output,
                glossary_path=args.glossary,
                character_path=args.characters,
                mode=args.mode,
                skip_confirmation=args.yes,
            )
        
        # 結果サマリを出力
        print("\n=== 処理完了 ===")
        print(f"ファイル数: {summary.total_files}")
        print(f"総行数: {summary.total_rows}")
        print(f"翻訳行数: {summary.translated_rows}")
        print(f"スキップ行数: {summary.skipped_rows}")
        print(f"エラー行数: {summary.error_rows}")
        print(f"総コスト: ${summary.total_cost_usd:.4f}")
        print(f"処理時間: {summary.processing_time_ms / 1000:.1f}秒")
        
        return 0 if not summary.errors else 1
        
    except ConfigurationError as e:
        logger.error(f"設定エラー: {e}")
        print(f"\n❌ 設定エラー: {e}")
        return 1
        
    except TranslationAppError as e:
        logger.error(f"翻訳エラー: {e}")
        print(f"\n❌ 翻訳エラー: {e}")
        return 1
        
    except KeyboardInterrupt:
        logger.info("ユーザーにより中断されました")
        print("\n⚠️ 処理が中断されました")
        return 130
        
    except Exception as e:
        logger.exception(f"予期しないエラー: {e}")
        print(f"\n❌ 予期しないエラー: {e}")
        return 1


def main() -> int:
    """
    CLIエントリーポイント
    
    Returns:
        終了コード
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # ロギング設定
    setup_logging(verbose=args.verbose, quiet=args.quiet)
    
    # 非同期処理を実行
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())
