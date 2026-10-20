# -*- coding: utf-8 -*-

import os
import sys
import datetime
import numpy as np


CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if CURRENT_DIR not in sys.path:
    sys.path.insert(
        0,
        CURRENT_DIR
    )


from alphasim import AlphaBase
from alphasim import DataRegistry as dr


class PeopleDailyAIPolicyAlpha(AlphaBase):
    """
    People's Daily AI Policy Direction Factor

    Logic
    -----
    1. AI 判断每篇人民日报文章对指定申万一级行业的影响方向：
           +1 = 正向
            0 = 中性
           -1 = 负向

    2. AI 权重：
           article_weight
           * confidence
           * relevance
           * policy_strength

    3. 聚合为自然日 × 一级行业的：
           positive_attention
           negative_attention
           net_attention

    4. 构造 20 自然日累计及 365 日历史 surprise。

    5. 对行业做每日横截面 rank。

    6. 交易日 t 使用 t-1 自然日人民日报信号：
           周一 -> 周日
           长假后首个交易日 -> 前一自然日

    7. AlphaSim 当前历史申万二级行业
       映射到统一申万一级行业。

    8. 一级行业因子映射到股票。

    factorType
    ----------
    1 = positive_surprise_20_rank

        正向政策影响的异常程度。
        数值越大：
        该行业近期正向政策影响相对于自身历史越异常。

    2 = negative_surprise_20_rank

        负向政策影响的异常程度。
        注意：
        negative_attention 本身使用的是正数 magnitude。
        因此数值越大表示：
        负向政策冲击越异常。

    3 = net_surprise_20_rank

        净政策影响：
            positive - negative

        再计算20日累计、历史 surprise、行业横截面 rank。

    4 = net_attention_20_rank

        净政策影响20自然日累计的行业横截面 rank。
        不做时间序列 surprise。

    sign
    ----
    最终股票信号统一乘 sign。

    例如：

        sign = 1
            原方向

        sign = -1
            反向

    第一轮建议重点测试：

        factorType = 3
        sign = +1 / -1

        factorType = 2
        sign = +1 / -1
    """

    FACTOR_MAP = {

        # ==================================================
        # 原有
        # ==================================================

        1: "positive_surprise_20_rank",

        2: "negative_surprise_20_rank",

        # 当前基准：
        # Sharpe约0.70
        3: "net_surprise_20_rank",

        4: "net_attention_20_rank",

        # 10日
        5: "net_surprise_10_rank",

        # 40日
        6: "net_surprise_40_rank",

        # EWM
        7: "net_ewm10_surprise_rank",

        # AI quality，不看direction
        8: "quality_surprise_20_rank",

        # AI net剔除传统attention
        9: "ai_residual_rank",

        # 高置信负向
        10: "high_conf_negative_20_rank",

        # regulation/restriction/risk
        11: "constraint_surprise_20_rank",

        # ==================================================
        # 新增
        # ==================================================

        # 控制quality后：
        # direction还有没有独立信息？
        12: "direction_residual_rank",

        # 控制传统attention后：
        # AI quality还有没有独立信息？
        13: "quality_residual_rank",

        # AI quality 30日
        14: "quality_surprise_30_rank",

        # AI quality 40日
        15: "quality_surprise_40_rank",

        # net 20日 + 40日
        16: "net_surprise_20_40_combo_rank",
        # 17：
        # 75% factor16 + 25% quality residual
        17: "combo16_quality_residual_rank",

        # 18：
        # 平滑后的quality residual
        18: "quality_residual_ewm5_rank",

        # 19：
        # factor16 + quality20
        19: "combo16_quality20_rank",
    }

    def __init__(self, cfg):

        super().__init__(cfg)

        # ==================================================
        # 参数
        # ==================================================

        self.factor_type = int(
            cfg.getAttributeDefault(
                "factorType",
                3
            )
        )

        if self.factor_type not in self.FACTOR_MAP:
            raise ValueError(
                "factorType must be 1-19"
            )

        self.factor_name = (
            self.FACTOR_MAP[
                self.factor_type
            ]
        )

        # ----------------------------------------------
        # sign
        # ----------------------------------------------

        self.sign = float(
            cfg.getAttributeDefault(
                "sign",
                1.0
            )
        )

        # ----------------------------------------------
        # debug
        # ----------------------------------------------

        self.debug = int(
            cfg.getAttributeDefault(
                "debug",
                0
            )
        )

        # ==================================================
        # AlphaSim 数据
        # ==================================================

        self.industry = dr.getData(
            "industry"
        )

        # ==================================================
        # PeopleDaily AI 因子
        # ==================================================

        self.factor_path = (
            "/home/dwyao/WorkPlace/"
            "PeopleDailyFactor/data/factor/"
            "people_daily_ai_policy_factor.npz"
        )

        factor_data = np.load(
            self.factor_path
        )

        self.factor_dates = (
            factor_data["dates"]
            .astype(np.int64)
        )

        self.factor_codes = (
            factor_data["codes"]
            .astype(np.int64)
        )

        self.factor_matrix = (
            factor_data[
                self.factor_name
            ]
            .astype(np.float64)
        )

        # ==================================================
        # 基本完整性检查
        # ==================================================

        if (
            self.factor_matrix.shape[0]
            != len(self.factor_dates)
        ):
            raise ValueError(
                "factor matrix date dimension mismatch"
            )

        if (
            self.factor_matrix.shape[1]
            != len(self.factor_codes)
        ):
            raise ValueError(
                "factor matrix industry dimension mismatch"
            )

        # ==================================================
        # AlphaSim 日期
        # ==================================================

        self.date_path = (
            "/data/simucache/"
            "__universe/Dates.npy"
        )

        self.sim_dates = np.memmap(
            self.date_path,
            dtype=np.int64,
            mode="r"
        )

        # ==================================================
        # 初始化信息
        # ==================================================

        print(
            "AlphaSim dates:",
            len(self.sim_dates),
            self.sim_dates[:5],
            "...",
            self.sim_dates[-5:]
        )

        print("")
        print("=" * 70)
        print("PEOPLE DAILY AI POLICY ALPHA")
        print("=" * 70)

        print(
            "factorType:",
            self.factor_type
        )

        print(
            "factorName:",
            self.factor_name
        )

        print(
            "sign:",
            self.sign
        )

        print(
            "debug:",
            self.debug
        )

        print(
            "factorPath:",
            self.factor_path
        )

        print(
            "PeopleDaily dates:",
            int(self.factor_dates[0]),
            "->",
            int(self.factor_dates[-1])
        )

        print(
            "industries:",
            len(self.factor_codes)
        )

        print(
            "industry codes:",
            self.factor_codes
        )

        print(
            "factor matrix:",
            self.factor_matrix.shape
        )

        finite_count = int(
            np.isfinite(
                self.factor_matrix
            ).sum()
        )

        print(
            "finite:",
            finite_count,
            "/",
            self.factor_matrix.size
        )

        print("=" * 70)
        print("")


    # ======================================================
    # 日期：前一自然日
    # ======================================================

    @staticmethod
    def previous_calendar_date(
        yyyymmdd
    ):

        dt = datetime.datetime.strptime(
            str(int(yyyymmdd)),
            "%Y%m%d"
        )

        dt = (
            dt
            - datetime.timedelta(
                days=1
            )
        )

        return int(
            dt.strftime(
                "%Y%m%d"
            )
        )


    # ======================================================
    # 申万旧二级 -> 统一新一级
    # ======================================================

    @staticmethod
    def map_industry_array(raw):

        raw = np.asarray(
            raw,
            dtype=np.int64
        )

        # ----------------------------------------------
        # 普通情况：
        #
        # 110100 -> 110000
        # 220300 -> 220000
        # 710100 -> 710000
        # ----------------------------------------------

        level1 = (
            raw // 10000
        ) * 10000

        # ----------------------------------------------
        # 无效行业
        # ----------------------------------------------

        level1[
            raw <= 0
        ] = -1

        level1[
            raw == -999999999
        ] = -1

        # ----------------------------------------------
        # 旧采掘体系
        # ----------------------------------------------

        # 210100
        # 石油开采
        # -> 新一级：石油石化
        level1[
            raw == 210100
        ] = 750000

        # 210200
        # 煤炭开采
        # -> 新一级：煤炭
        level1[
            raw == 210200
        ] = 740000

        # 210300
        # 其他采掘
        # 行业归属存在歧义，暂不映射
        level1[
            raw == 210300
        ] = -1

        # 210400
        # 采掘服务
        # 二级层面存在歧义，暂不映射
        level1[
            raw == 210400
        ] = -1

        # ----------------------------------------------
        # 旧化工体系
        # ----------------------------------------------

        # 220100
        # 石油化工
        # -> 新一级：石油石化
        level1[
            raw == 220100
        ] = 750000

        # ----------------------------------------------
        # 旧公用事业体系
        # ----------------------------------------------

        # 410400
        # 环保工程及服务
        # -> 新一级：环保
        level1[
            raw == 410400
        ] = 760000

        # ----------------------------------------------
        # 综合
        #
        # 当前 PeopleDaily 文本因子没有综合行业
        # ----------------------------------------------

        level1[
            raw == 510100
        ] = -1

        return level1


    # ======================================================
    # 找 PeopleDaily 自然日因子
    # ======================================================

    def get_factor_row(
        self,
        signal_date
    ):

        pos = np.searchsorted(
            self.factor_dates,
            signal_date
        )

        if pos >= len(
            self.factor_dates
        ):
            return None

        if (
            self.factor_dates[pos]
            != signal_date
        ):
            return None

        return self.factor_matrix[
            pos
        ]


    # ======================================================
    # Alpha
    # ======================================================

    def generate(self, di):

        # ----------------------------------------------
        # 每日先清零
        # ----------------------------------------------

        self.alpha[:] = 0.0

        # ----------------------------------------------
        # 当前交易日
        # ----------------------------------------------

        trade_date = int(
            self.sim_dates[di]
        )

        # ----------------------------------------------
        # 使用前一自然日人民日报
        #
        # 周一 -> 周日
        #
        # 长假后首个交易日
        # -> 开盘前一自然日
        #
        # 因此 XML 中 delay 必须保持 0
        # ----------------------------------------------

        signal_date = (
            self.previous_calendar_date(
                trade_date
            )
        )

        factor_row = (
            self.get_factor_row(
                signal_date
            )
        )

        if factor_row is None:

            if self.debug:

                print(
                    "[PeopleDailyAI]",
                    "di=",
                    di,
                    "tradeDate=",
                    trade_date,
                    "signalDate=",
                    signal_date,
                    "NO FACTOR DATA"
                )

            return

        # ----------------------------------------------
        # 当前股票的历史申万行业
        # ----------------------------------------------

        raw_industry = np.asarray(
            self.industry[di]
        ).ravel()

        sw1 = self.map_industry_array(
            raw_industry
        )

        # ----------------------------------------------
        # 行业因子 -> 股票
        # ----------------------------------------------

        assigned_count = 0

        for j in range(
            len(self.factor_codes)
        ):

            code = int(
                self.factor_codes[j]
            )

            value = factor_row[j]

            if not np.isfinite(
                value
            ):
                continue

            mask = (
                sw1 == code
            )

            if not np.any(
                mask
            ):
                continue

            # ------------------------------------------
            # 不再像旧代码那样硬编码：
            #
            # self.alpha[mask] = -value
            #
            # 而是由 XML 的 sign 控制
            # ------------------------------------------

            self.alpha[
                mask
            ] = (
                self.sign
                * value
            )

            assigned_count += int(
                mask.sum()
            )

        # ----------------------------------------------
        # Universe
        # ----------------------------------------------

        try:

            valid = np.asarray(
                self.valid[di]
            ).ravel().astype(bool)

            self.alpha[
                ~valid
            ] = 0.0

        except Exception:

            pass

        # ----------------------------------------------
        # Debug
        # ----------------------------------------------

        if self.debug:

            finite_factor = int(
                np.isfinite(
                    factor_row
                ).sum()
            )

            nonzero_alpha = int(
                np.count_nonzero(
                    self.alpha
                )
            )

            print(
                "[PeopleDailyAI]",
                "di=",
                di,
                "tradeDate=",
                trade_date,
                "signalDate=",
                signal_date,
                "finiteIndustryFactor=",
                finite_factor,
                "assignedStocks=",
                assigned_count,
                "nonzeroAlpha=",
                nonzero_alpha,
                "alphaMin=",
                float(
                    np.min(
                        self.alpha
                    )
                ),
                "alphaMax=",
                float(
                    np.max(
                        self.alpha
                    )
                ),
            )