# -*- coding: utf-8 -*-

from dataclasses import dataclass

from sw1 import INDUSTRY_BY_NAME
from industry_rules import INDUSTRY_RULES


@dataclass
class IndustryMatch:
    name: str
    framework_code: str
    index_code: str

    title_terms: tuple
    body_terms: tuple
    context_terms: tuple

    title_hits: int
    body_hits: int

    score: float
    weight: float = 0.0

    reason: str = ""


class IndustryClassifier:

    def __init__(
        self,
        max_industries=4,
        long_title_threshold=200,
    ):
        self.max_industries = max_industries
        self.long_title_threshold = long_title_threshold


    @staticmethod
    def mask_phrases(
        text,
        phrases,
    ):
        """
        屏蔽容易造成误判的特定语境。
        """

        result = text or ""

        for phrase in phrases:
            result = result.replace(
                phrase,
                " "
            )

        return result


    @staticmethod
    def matched_terms(
        text,
        terms,
    ):
        """
        返回实际出现过的不同关键词。

        注意：
        不按出现次数无限加分。
        """

        if not text:
            return []

        return [
            term
            for term in terms
            if term in text
        ]


    @staticmethod
    def total_occurrences(
        text,
        terms,
    ):
        """
        用于判断某个行业词是否在正文中反复出现。
        """

        if not text:
            return 0

        return sum(
            text.count(term)
            for term in terms
        )


    def normalize_article(
        self,
        title,
        text,
    ):
        """
        处理极少数原始数据异常：

        title 极长而 text 为空，
        很可能正文被放进了 title。

        此时把它作为正文处理，
        避免获得标题高权重。
        """

        title = title or ""
        text = text or ""

        if (
            len(title)
            > self.long_title_threshold
            and not text.strip()
        ):
            return "", title

        return title, text


    def evaluate_rule(
        self,
        title,
        text,
        rule,
    ):
        """
        判断一篇文章是否真正与该行业相关。
        """

        title_clean = self.mask_phrases(
            title,
            rule["exclude"],
        )

        body_clean = self.mask_phrases(
            text,
            rule["exclude"],
        )

        title_terms = self.matched_terms(
            title_clean,
            rule["strong"],
        )

        body_terms = self.matched_terms(
            body_clean,
            rule["strong"],
        )

        context_terms = self.matched_terms(
            body_clean,
            rule["context"],
        )

        body_occurrences = (
            self.total_occurrences(
                body_clean,
                rule["strong"],
            )
        )

        # ====================================================
        # 高精度准入条件
        # ====================================================

        reason = None

        # 1. 标题直接出现行业核心词
        if title_terms:
            reason = "title_strong"

        # 2. 正文出现至少两个不同核心词
        elif len(body_terms) >= 2:
            reason = "body_multi_strong"

        # 3. 一个核心词在正文反复出现 >= 3 次
        elif (
            len(body_terms) == 1
            and body_occurrences >= 3
        ):
            reason = "body_repeated_strong"

        # 4. 一个核心词 + 至少一个行业语境词
        elif (
            len(body_terms) == 1
            and len(context_terms) >= 1
        ):
            reason = "strong_plus_context"

        if reason is None:
            return None

        # ====================================================
        # 分数只用于多行业之间分配文章权重
        # 不再作为是否入选的唯一标准
        # ====================================================

        score = (
            4.0 * len(title_terms)
            + 2.0 * len(body_terms)
            + 1.0 * len(context_terms)
        )

        return {
            "title_terms": title_terms,
            "body_terms": body_terms,
            "context_terms": context_terms,
            "score": score,
            "reason": reason,
            "body_occurrences": body_occurrences,
        }


    def classify(
        self,
        title,
        text,
    ):

        title, text = self.normalize_article(
            title,
            text,
        )

        matches = []

        for industry_name, rule in INDUSTRY_RULES.items():

            if not rule["strong"]:
                continue

            result = self.evaluate_rule(
                title,
                text,
                rule,
            )

            if result is None:
                continue

            industry = INDUSTRY_BY_NAME[
                industry_name
            ]

            matches.append(
                IndustryMatch(
                    name=industry.name,
                    framework_code=industry.framework_code,
                    index_code=industry.index_code,

                    title_terms=tuple(
                        result["title_terms"]
                    ),

                    body_terms=tuple(
                        result["body_terms"]
                    ),

                    context_terms=tuple(
                        result["context_terms"]
                    ),

                    title_hits=len(
                        result["title_terms"]
                    ),

                    body_hits=len(
                        result["body_terms"]
                    ),

                    score=result["score"],
                    reason=result["reason"],
                )
            )

        matches.sort(
            key=lambda x: x.score,
            reverse=True
        )

        matches = matches[
            :self.max_industries
        ]

        if not matches:
            return []

        total_score = sum(
            x.score
            for x in matches
        )

        for match in matches:

            match.weight = (
                match.score
                / total_score
            )

        return matches