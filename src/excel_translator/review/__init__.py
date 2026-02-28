"""レビューパイプラインモジュール

ReviewPipelineクラスを提供し、マルチパスレビューを実行する。
- 2nd pass: セルフレビュー（修正が必要な行のみ返す）
- 3rd pass: 一貫性チェック（ルールベース + AI）
- 4th pass: バックトランスレーション検証（対象行のみ）
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PassResult:
    """パス処理結果"""
    text_id: str
    result: str
    reason: Optional[str] = None
    changed: bool = False
    remarks: Optional[str] = None


@dataclass
class RuleCheckResult:
    """ルールベースチェック結果"""
    text_id: str
    issue_type: str
    original: str
    suggested: str
    description: str


@dataclass
class BacktranslationResult:
    """バックトランスレーション結果"""
    text_id: str
    original_source: str
    translation: str
    backtranslation: str
    similarity_score: int = 0
    has_discrepancy: bool = False
    discrepancy_description: Optional[str] = None
    suggested_revision: Optional[str] = None


class ReviewPipelineError(Exception):
    """レビューパイプラインエラー"""
    pass


# 一貫性チェック用の正規表現パターン
CONSISTENCY_PATTERNS = {
    # オノマトペパターン（カタカナの繰り返し）
    "onomatopoeia": re.compile(r'([ァ-ヶー]{2,})'),
    
    # 括弧パターン
    "brackets_jp": re.compile(r'[「」『』【】（）]'),
    "brackets_en": re.compile(r'[""\'\'()]'),
    
    # 数字パターン
    "numbers_half": re.compile(r'[0-9]+'),
    "numbers_full": re.compile(r'[０-９]+'),
    "numbers_kanji": re.compile(r'[一二三四五六七八九十百千万億兆]+'),
    
    # 句読点パターン
    "punctuation_jp": re.compile(r'[。、！？]'),
    "punctuation_en": re.compile(r'[.,!?]'),
    
    # 省略記号パターン
    "ellipsis_jp": re.compile(r'[…]+'),
    "ellipsis_dots": re.compile(r'\.{2,}'),
}


class ReviewPipeline:
    """
    レビューパイプラインクラス
    
    マルチパスレビュー（2nd pass〜4th pass）を実行し、
    翻訳品質を向上させる。
    """
    
    def __init__(
        self,
        translation_provider=None,
        prompt_builder=None,
        source_lang: str = "ja",
        target_lang: str = "en",
    ):
        """
        ReviewPipelineを初期化
        
        Args:
            translation_provider: 翻訳プロバイダー（AI呼び出し用）
            prompt_builder: プロンプトビルダー
            source_lang: 原文言語コード
            target_lang: 翻訳先言語コード
        """
        self.translation_provider = translation_provider
        self.prompt_builder = prompt_builder
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # 統計情報
        self._stats = {
            "2nd_pass_reviewed": 0,
            "2nd_pass_modified": 0,
            "3rd_pass_rules_checked": 0,
            "3rd_pass_rules_issues": 0,
            "3rd_pass_ai_checked": 0,
            "3rd_pass_ai_modified": 0,
            "4th_pass_checked": 0,
            "4th_pass_discrepancies": 0,
        }
    
    async def run_2nd_pass(
        self,
        source_texts: List[Dict[str, Any]],
        pass1_results: List[Dict[str, Any]],
        character_profiles: Optional[List] = None,
    ) -> List[PassResult]:
        """
        2nd pass: セルフレビュー（修正が必要な行のみ返す）
        
        Args:
            source_texts: 原文テキストのリスト
                          [{"text_id": str, "character": str|None, "source_text": str}, ...]
            pass1_results: 1st pass翻訳結果のリスト
                           [{"text_id": str, "translated_text": str}, ...]
            character_profiles: キャラクタープロファイルリスト
            
        Returns:
            修正が必要な行のPassResultリスト
        """
        if not self.translation_provider or not self.prompt_builder:
            raise ReviewPipelineError(
                "2nd passにはtranslation_providerとprompt_builderが必要です"
            )
        
        self._stats["2nd_pass_reviewed"] += len(pass1_results)
        stats_dict = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0, "response_time_ms": 0}
        
        # レビュープロンプトを構築
        user_prompt = self.prompt_builder.build_review_prompt(
            source_texts=source_texts,
            translations=pass1_results,
            character_profiles=character_profiles,
        )
        
        system_prompt = self.prompt_builder.build_system_prompt(pass_type="review")
        
        # AI呼び出し
        response = await self.translation_provider.translate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        
        stats_dict["input_tokens"] = response.input_tokens
        stats_dict["output_tokens"] = response.output_tokens
        stats_dict["cache_read_tokens"] = response.cache_read_tokens
        stats_dict["cache_write_tokens"] = response.cache_write_tokens
        stats_dict["response_time_ms"] = response.response_time_ms
        
        # 結果を解析
        results = []
        for review in response.translations:
            text_id = review.get("text_id", "")
            revised = review.get("revised_translation", "")
            reason = review.get("reason", "")
            
            if text_id and revised:
                results.append(PassResult(
                    text_id=text_id,
                    result=revised,
                    reason=reason,
                    changed=(revised != next((t["translated_text"] for t in pass1_results if t["text_id"] == text_id), "")),
                    remarks=review.get("remarks"),
                ))
                if revised != next((t["translated_text"] for t in pass1_results if t["text_id"] == text_id), ""):
                    self._stats["2nd_pass_modified"] += 1
        
        logger.info(
            f"2nd pass完了: {len(results)}/{len(pass1_results)} 件修正"
        )
        
        return results, stats_dict
    
    def run_3rd_pass_preprocess(
        self,
        all_translations: List[Dict[str, Any]],
    ) -> List[PassResult]:
        """
        3rd pass AI実行前のルールベース前処理（機械的修正）
        
        処理内容:
        - 台詞行（character_idあり）のクォーテーション付与
        - 全角括弧の正規化
        - 連続スペース・改行の正規化
        
        Args:
            all_translations: 全翻訳結果のリスト
            
        Returns:
            修正された行のPassResultリスト
        """
        results = []
        import re
        
        for trans in all_translations:
            text_id = trans.get("text_id", "")
            original = trans.get("translated_text") or ""
            character = trans.get("character")
            
            if not original or not text_id:
                continue
                
            modified = original
            
            # 全角括弧の正規化
            modified = modified.replace("「", '"').replace("」", '"')
            modified = modified.replace("『", '"').replace("』", '"')
            
            # 連続スペースの正規化 (タグなどを壊さない範囲で)
            modified = re.sub(r' {2,}', ' ', modified)
            
            # 台詞行のクォーテーション付与
            if character and str(character).strip():
                # 先頭末尾の空白除去
                modified = modified.strip()
                # 既にクォートで囲まれていない場合
                if not (modified.startswith('"') and modified.endswith('"')) and not (modified.startswith("'") and modified.endswith("'")):
                    # ただし、地の文＋台詞のように途中にクォートがある複雑な文は除く
                    if modified.count('"') == 0:
                        modified = f'"{modified}"'
            
            if modified != original:
                trans["translated_text"] = modified
                results.append(PassResult(
                    text_id=text_id,
                    result=modified,
                    reason="（前処理）フォーマット・記号の正規化",
                    changed=True,
                ))
                
        return results

    def run_3rd_pass_rules(
        self,
        all_translations: List[Dict[str, Any]],
    ) -> List[RuleCheckResult]:
        """
        3rd pass: ルールベース一貫性チェック（正規表現）
        
        チェック項目:
        - オノマトペの方針統一
        - 括弧・記号の統一
        - 数字表記の統一
        - 句読点の統一
        
        Args:
            all_translations: 全翻訳結果のリスト
                              [{"text_id": str, "source_text": str, "translated_text": str, "character": str|None}, ...]
            
        Returns:
            RuleCheckResultのリスト
        """
        self._stats["3rd_pass_rules_checked"] += len(all_translations)
        
        issues = []
        
        # 統計を収集
        bracket_stats = self._collect_bracket_stats(all_translations)
        number_stats = self._collect_number_stats(all_translations)
        punctuation_stats = self._collect_punctuation_stats(all_translations)
        
        # 各翻訳をチェック
        for trans in all_translations:
            text_id = trans.get("text_id", "")
            translated = trans.get("translated_text", "")
            
            # 括弧の不統一チェック
            bracket_issues = self._check_bracket_consistency(
                text_id, translated, bracket_stats
            )
            issues.extend(bracket_issues)
            
            # 数字表記の不統一チェック
            number_issues = self._check_number_consistency(
                text_id, translated, number_stats
            )
            issues.extend(number_issues)
            
            # 句読点の不統一チェック
            punctuation_issues = self._check_punctuation_consistency(
                text_id, translated, punctuation_stats
            )
            issues.extend(punctuation_issues)
        
        self._stats["3rd_pass_rules_issues"] += len(issues)
        
        logger.info(
            f"3rd pass (ルールベース) 完了: {len(issues)} 件の問題を検出"
        )
        
        return issues
    
    async def run_3rd_pass_ai(
        self,
        all_translations: List[Dict[str, Any]],
        rule_issues: Optional[List[RuleCheckResult]] = None,
    ) -> List[PassResult]:
        """
        3rd pass: AI一貫性チェック（文体・ニュアンス系）
        
        Args:
            all_translations: 全翻訳結果のリスト
            rule_issues: ルールベースチェックで検出された問題（参考情報）
            
        Returns:
            修正が必要な行のPassResultリスト
        """
        if not self.translation_provider or not self.prompt_builder:
            raise ReviewPipelineError(
                "3rd pass AIにはtranslation_providerとprompt_builderが必要です"
            )
        
        self._stats["3rd_pass_ai_checked"] += len(all_translations)
        stats_dict = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0, "response_time_ms": 0}
        
        # 一貫性チェックプロンプトを構築
        user_prompt = self.prompt_builder.build_consistency_prompt(all_translations)
        system_prompt = self.prompt_builder.build_system_prompt(pass_type="consistency")
        
        # AI呼び出し
        response = await self.translation_provider.translate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        
        stats_dict["input_tokens"] = response.input_tokens
        stats_dict["output_tokens"] = response.output_tokens
        stats_dict["cache_read_tokens"] = response.cache_read_tokens
        stats_dict["cache_write_tokens"] = response.cache_write_tokens
        stats_dict["response_time_ms"] = response.response_time_ms
        
        # 結果を解析
        results = []
        for fix in response.translations:
            text_id = fix.get("text_id", "")
            revised = fix.get("revised_translation", "")
            reason = fix.get("reason", "")
            
            if text_id and revised:
                results.append(PassResult(
                    text_id=text_id,
                    result=revised,
                    reason=reason,
                    changed=True,
                    remarks=fix.get("remarks"),
                ))
                self._stats["3rd_pass_ai_modified"] += 1
        
        logger.info(
            f"3rd pass (AI) 完了: {len(results)}/{len(all_translations)} 件修正"
        )
        
        return results, stats_dict
    
    async def run_4th_pass(
        self,
        translations: List[Dict[str, Any]],
        target_text_ids: Optional[List[str]] = None,
    ) -> Tuple[List[BacktranslationResult], Dict[str, Any]]:
        """
        4th pass: バックトランスレーション検証を実行（バイアス排除版）
        
        Args:
            translations: 検証対象の翻訳結果リスト
                          [{"text_id": str, "source_text": str, "translated_text": str}, ...]
            target_text_ids: 特定のIDのみ実行する場合のリスト
                          
        Returns:
            (結果リスト, 統計情報)
        """
        if not self.translation_provider or not self.prompt_builder:
            raise ReviewPipelineError(
                "4th passにはtranslation_providerとprompt_builderが必要です"
            )
        
        # 対象行をフィルタリング
        if target_text_ids:
            target_set = set(target_text_ids)
            filtered = [t for t in translations if t.get("text_id") in target_set]
        else:
            filtered = translations
        
        if not filtered:
            logger.info("4th pass: 検証対象なし")
            return [], {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0, "response_time_ms": 0}
        
        self._stats["4th_pass_checked"] += len(filtered)
        total_stats = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0, "response_time_ms": 0}
        
        # --- Step 1: Blind Backtranslation ---
        # 原文を隠して再翻訳のみを行う
        step1_prompt = self.prompt_builder.build_backtrans_step1_prompt(
            translations=filtered,
            source_lang=self.source_lang,
        )
        system_prompt_bt = self.prompt_builder.build_system_prompt(pass_type="backtranslation") # backtranslations形式で出力
        
        res1 = await self.translation_provider.translate(
            system_prompt=system_prompt_bt,
            user_prompt=step1_prompt,
        )
        
        for k in total_stats:
            if hasattr(res1, k):
                total_stats[k] += getattr(res1, k)
        
        # 再翻訳結果をマッピング（backtranslation または translated_text キーをサポート）
        bt_map = {}
        for item in res1.translations:
            text_id = item.get("text_id")
            # backtranslation キーを優先、なければ translated_text をフォールバック
            backtrans = item.get("backtranslation") or item.get("translated_text")
            if text_id and backtrans:
                bt_map[text_id] = backtrans

        logger.debug(f"4th pass Step 1完了: {len(res1.translations)}件のレスポンス, {len(bt_map)}件のマッピング成功")
        if len(res1.translations) > 0 and len(bt_map) == 0:
            logger.warning(f"4th pass Step 1: レスポンスはあるがマッピング失敗。レスポンス例: {res1.translations[0] if res1.translations else 'N/A'}")

        # --- Step 2: Semantic Comparison ---
        # 原文と再翻訳結果を比較する
        comparisons = []
        for t in filtered:
            text_id = t.get("text_id")
            if text_id in bt_map:
                comparisons.append({
                    "text_id": text_id,
                    "original_source": t.get("source_text", ""),
                    "backtranslation": bt_map[text_id]
                })
        
        if not comparisons:
            return [], total_stats
            
        step2_prompt = self.prompt_builder.build_backtrans_step2_prompt(
            comparisons=comparisons,
            source_lang=self.source_lang,
        )
        system_prompt_qa = self.prompt_builder.build_system_prompt(pass_type="backtranslation")
        
        res2 = await self.translation_provider.translate(
            system_prompt=system_prompt_qa,
            user_prompt=step2_prompt,
        )
        
        for k in total_stats:
            if hasattr(res2, k):
                total_stats[k] += getattr(res2, k)
        
        # 結果を統合
        results = []
        # res2.translations を使用 (パーサーが整形済み)
        comparison_results = res2.translations
        
        for cr in comparison_results:
            text_id = cr.get("text_id", "")
            original = next((t for t in filtered if t.get("text_id") == text_id), {})
            
            result = BacktranslationResult(
                text_id=text_id,
                original_source=original.get("source_text", ""),
                translation=original.get("translated_text", ""),
                backtranslation=bt_map.get(text_id, ""),
                similarity_score=cr.get("similarity_score", 0),
                has_discrepancy=cr.get("has_discrepancy", False),
                discrepancy_description=cr.get("discrepancy_description"),
                suggested_revision=cr.get("suggested_revision"),
            )
            results.append(result)
            
            if result.has_discrepancy:
                self._stats["4th_pass_discrepancies"] += 1
        
        logger.info(
            f"4th pass検証完了: {len(results)}件 (乖離検出: {self._stats['4th_pass_discrepancies']}件)"
        )
        
        return results, total_stats
    
    def _collect_bracket_stats(
        self,
        translations: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """括弧の使用統計を収集"""
        stats = {
            "jp_brackets": 0,
            "en_brackets": 0,
        }
        
        for trans in translations:
            text = trans.get("translated_text", "")
            
            if CONSISTENCY_PATTERNS["brackets_jp"].search(text):
                stats["jp_brackets"] += 1
            if CONSISTENCY_PATTERNS["brackets_en"].search(text):
                stats["en_brackets"] += 1
        
        return stats
    
    def _collect_number_stats(
        self,
        translations: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """数字表記の使用統計を収集"""
        stats = {
            "half_width": 0,
            "full_width": 0,
            "kanji": 0,
        }
        
        for trans in translations:
            text = trans.get("translated_text", "")
            
            if CONSISTENCY_PATTERNS["numbers_half"].search(text):
                stats["half_width"] += 1
            if CONSISTENCY_PATTERNS["numbers_full"].search(text):
                stats["full_width"] += 1
            if CONSISTENCY_PATTERNS["numbers_kanji"].search(text):
                stats["kanji"] += 1
        
        return stats
    
    def _collect_punctuation_stats(
        self,
        translations: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """句読点の使用統計を収集"""
        stats = {
            "jp_punctuation": 0,
            "en_punctuation": 0,
        }
        
        for trans in translations:
            text = trans.get("translated_text", "")
            
            if CONSISTENCY_PATTERNS["punctuation_jp"].search(text):
                stats["jp_punctuation"] += 1
            if CONSISTENCY_PATTERNS["punctuation_en"].search(text):
                stats["en_punctuation"] += 1
        
        return stats
    
    def _check_bracket_consistency(
        self,
        text_id: str,
        text: str,
        stats: Dict[str, int],
    ) -> List[RuleCheckResult]:
        """括弧の一貫性をチェック"""
        issues = []
        
        # 主要な括弧スタイルを判定
        dominant_style = "jp" if stats["jp_brackets"] > stats["en_brackets"] else "en"
        
        has_jp = bool(CONSISTENCY_PATTERNS["brackets_jp"].search(text))
        has_en = bool(CONSISTENCY_PATTERNS["brackets_en"].search(text))
        
        # 混在している場合は問題
        if has_jp and has_en:
            issues.append(RuleCheckResult(
                text_id=text_id,
                issue_type="bracket_mixed",
                original=text,
                suggested=text,  # 自動修正は行わない
                description=f"括弧スタイルが混在しています（主要スタイル: {dominant_style}）",
            ))
        
        return issues
    
    def _check_number_consistency(
        self,
        text_id: str,
        text: str,
        stats: Dict[str, int],
    ) -> List[RuleCheckResult]:
        """数字表記の一貫性をチェック"""
        issues = []
        
        has_half = bool(CONSISTENCY_PATTERNS["numbers_half"].search(text))
        has_full = bool(CONSISTENCY_PATTERNS["numbers_full"].search(text))
        
        # 半角と全角が混在している場合は問題
        if has_half and has_full:
            issues.append(RuleCheckResult(
                text_id=text_id,
                issue_type="number_mixed",
                original=text,
                suggested=text,
                description="数字表記（半角/全角）が混在しています",
            ))
        
        return issues
    
    def _check_punctuation_consistency(
        self,
        text_id: str,
        text: str,
        stats: Dict[str, int],
    ) -> List[RuleCheckResult]:
        """句読点の一貫性をチェック"""
        issues = []
        
        # 英語翻訳の場合は不要なチェックをスキップする
        if self.target_lang == "en":
            return issues
        
        # 主要な句読点スタイルを判定
        dominant_style = "jp" if stats["jp_punctuation"] > stats["en_punctuation"] else "en"
        
        has_jp = bool(CONSISTENCY_PATTERNS["punctuation_jp"].search(text))
        has_en = bool(CONSISTENCY_PATTERNS["punctuation_en"].search(text))
        
        # 混在している場合は問題（ただし英語翻訳では許容）
        if self.target_lang == "ja" and has_jp and has_en:
            issues.append(RuleCheckResult(
                text_id=text_id,
                issue_type="punctuation_mixed",
                original=text,
                suggested=text,
                description=f"句読点スタイルが混在しています（主要スタイル: {dominant_style}）",
            ))
        
        return issues
    
    def get_modified_text_ids(
        self,
        pass2_results: Optional[List[PassResult]] = None,
        pass3_results: Optional[List[PassResult]] = None,
    ) -> List[str]:
        """
        修正された行のtext_idリストを取得（4th pass対象決定用）
        
        Args:
            pass2_results: 2nd pass結果
            pass3_results: 3rd pass結果
            
        Returns:
            修正されたtext_idのリスト
        """
        modified_ids = set()
        
        if pass2_results:
            for result in pass2_results:
                if result.changed:
                    modified_ids.add(result.text_id)
        
        if pass3_results:
            for result in pass3_results:
                if result.changed:
                    modified_ids.add(result.text_id)
        
        return list(modified_ids)
    
    def apply_pass_results(
        self,
        translations: List[Dict[str, Any]],
        pass_results: List[PassResult],
        pass_name: str,
    ) -> List[Dict[str, Any]]:
        """
        パス結果を翻訳に適用
        
        Args:
            translations: 翻訳結果のリスト
            pass_results: パス結果のリスト
            pass_name: パス名（"pass_2", "pass_3"等）
            
        Returns:
            更新された翻訳結果のリスト
        """
        # text_idでマッピング
        result_map = {r.text_id: r for r in pass_results}
        
        updated = []
        for trans in translations:
            text_id = trans.get("text_id", "")
            new_trans = trans.copy()
            
            if text_id in result_map:
                result = result_map[text_id]
                new_trans[pass_name] = result.result
                new_trans[f"{pass_name}_reason"] = result.reason
                new_trans["translated_text"] = result.result  # 最新の翻訳を更新
            
            updated.append(new_trans)
        
        return updated
    
    def get_statistics(self) -> Dict[str, int]:
        """統計情報を取得"""
        return self._stats.copy()
    
    def reset_statistics(self) -> None:
        """統計情報をリセット"""
        self._stats = {
            "2nd_pass_reviewed": 0,
            "2nd_pass_modified": 0,
            "3rd_pass_rules_checked": 0,
            "3rd_pass_rules_issues": 0,
            "3rd_pass_ai_checked": 0,
            "3rd_pass_ai_modified": 0,
            "4th_pass_checked": 0,
            "4th_pass_discrepancies": 0,
        }
