# -*- coding: utf-8 -*-

import os
import datetime
import numpy as np

from alphasim import AlphaBase
from alphasim import DataRegistry as dr


class PeopleDailySurpriseExploreAlpha(AlphaBase):
    """
    人民日报行业政策 Surprise 参数探索版

    核心定义
    --------
    short_attention(t)
        = 最近 shortWindow 个自然日 attention 之和

    surprise(t)
        = [short_attention(t) - 历史均值]
          / 历史标准差

    历史基准严格使用 t 以前的数据，
    不把当天 short_attention 放入均值/std。

    XML 参数
    --------
    shortWindow
        最近政策关注窗口
        默认 20

    historyWindow
        surprise 历史标准化窗口
        默认 365

    minHistory
        至少多少天历史才产生信号
        默认 120

    signalMode
        1 = surprise
        2 = surprise 的 diffLag 日变化

    diffLag
        signalMode=2 时使用
        默认 5

    rankMode
        1 = 每日行业截面 rank 到 [-1, 1]
        0 = 直接使用 z-score

    sign
        1  = 原方向
        -1 = 反方向

    extremeCut
        0   = 全部行业
        0.3 = 仅保留 |rank| >= 0.3
        0.5 = 仅保留 |rank| >= 0.5
        0.7 = 仅保留最极端部分

        注意：
        这个参数主要用于 rankMode=1。

    debug
        0 = 不打印
        1 = 打印初始化信息
    """


    def __init__(self, cfg):

        super().__init__(cfg)

        # ==================================================
        # XML 参数
        # ==================================================

        self.short_window = int(
            cfg.getAttributeDefault(
                "shortWindow",
                20
            )
        )

        self.history_window = int(
            cfg.getAttributeDefault(
                "historyWindow",
                365
            )
        )

        self.min_history = int(
            cfg.getAttributeDefault(
                "minHistory",
                120
            )
        )

        self.signal_mode = int(
            cfg.getAttributeDefault(
                "signalMode",
                1
            )
        )

        self.diff_lag = int(
            cfg.getAttributeDefault(
                "diffLag",
                5
            )
        )

        self.rank_mode = int(
            cfg.getAttributeDefault(
                "rankMode",
                1
            )
        )

        self.sign = float(
            cfg.getAttributeDefault(
                "sign",
                -1.0
            )
        )

        self.extreme_cut = float(
            cfg.getAttributeDefault(
                "extremeCut",
                0.0
            )
        )

        self.debug = int(
            cfg.getAttributeDefault(
                "debug",
                1
            )
        )

        # ==================================================
        # AlphaSim industry
        # ==================================================

        self.industry = dr.getData(
            "industry"
        )

        # ==================================================
        # 加载 PeopleDaily 每日 attention
        # ==================================================

        self.data_path = (
            "/home/dwyao/WorkPlace/"
            "PeopleDailyFactor/data/factor/"
            "people_daily_attention_runtime.npz"
        )

        data = np.load(
            self.data_path
        )

        self.calendar_dates = (
            data["dates"]
            .astype(np.int64)
        )

        self.codes = (
            data["codes"]
            .astype(np.int64)
        )

        self.attention = (
            data["attention"]
            .astype(np.float64)
        )

        # ==================================================
        # AlphaSim 交易日期
        # ==================================================

        self.sim_dates = np.memmap(
            "/data/simucache/"
            "__universe/Dates.npy",
            dtype=np.int64,
            mode="r"
        )

        # ==================================================
        # 初始化时一次性计算全部历史 signal
        #
        # generate() 中就不用每天重新 rolling
        # ==================================================

        self.signal_matrix = (
            self.build_signal_matrix()
        )

        if self.debug:

            print("")
            print("=" * 72)
            print("PEOPLE DAILY SURPRISE EXPLORE")
            print("=" * 72)

            print(
                "shortWindow   =",
                self.short_window
            )

            print(
                "historyWindow =",
                self.history_window
            )

            print(
                "minHistory    =",
                self.min_history
            )

            print(
                "signalMode    =",
                self.signal_mode
            )

            print(
                "diffLag       =",
                self.diff_lag
            )

            print(
                "rankMode      =",
                self.rank_mode
            )

            print(
                "sign          =",
                self.sign
            )

            print(
                "extremeCut    =",
                self.extreme_cut
            )

            print(
                "calendar:",
                int(self.calendar_dates[0]),
                "->",
                int(self.calendar_dates[-1])
            )

            print(
                "signal shape:",
                self.signal_matrix.shape
            )

            print("=" * 72)
            print("")


    # ======================================================
    # rolling sum
    # ======================================================

    @staticmethod
    def rolling_sum(
        x,
        window
    ):

        n, m = x.shape

        out = np.zeros(
            (n, m),
            dtype=np.float64
        )

        cs = np.vstack(
            (
                np.zeros(
                    (1, m),
                    dtype=np.float64
                ),
                np.cumsum(
                    x,
                    axis=0
                )
            )
        )

        for i in range(n):

            start = max(
                0,
                i - window + 1
            )

            out[i] = (
                cs[i + 1]
                - cs[start]
            )

        return out


    # ======================================================
    # surprise
    # ======================================================

    def build_surprise(
        self,
        short_attention
    ):

        n, m = (
            short_attention.shape
        )

        out = np.full(
            (n, m),
            np.nan,
            dtype=np.float64
        )

        for i in range(n):

            # 只能使用 i 之前的历史
            end = i

            start = max(
                0,
                end - self.history_window
            )

            history_len = (
                end - start
            )

            if (
                history_len
                < self.min_history
            ):
                continue

            history = (
                short_attention[
                    start:end
                ]
            )

            mean = np.mean(
                history,
                axis=0
            )

            # pandas rolling.std 默认 ddof=1
            std = np.std(
                history,
                axis=0,
                ddof=1
            )

            valid = (
                np.isfinite(std)
                & (std > 1e-12)
            )

            row = np.full(
                m,
                np.nan,
                dtype=np.float64
            )

            row[valid] = (
                (
                    short_attention[
                        i,
                        valid
                    ]
                    - mean[valid]
                )
                / std[valid]
            )

            out[i] = row

        return out


    # ======================================================
    # 截面 rank [-1, 1]
    # ======================================================

    @staticmethod
    def rank_row(
        x
    ):

        out = np.full(
            len(x),
            np.nan,
            dtype=np.float64
        )

        valid_idx = np.where(
            np.isfinite(x)
        )[0]

        n = len(valid_idx)

        if n == 0:
            return out

        if n == 1:
            out[
                valid_idx[0]
            ] = 0.0

            return out

        values = x[
            valid_idx
        ]

        order = np.argsort(
            values,
            kind="mergesort"
        )

        sorted_values = values[
            order
        ]

        ranks = np.empty(
            n,
            dtype=np.float64
        )

        # average rank for ties
        i = 0

        while i < n:

            j = i + 1

            while (
                j < n
                and
                sorted_values[j]
                == sorted_values[i]
            ):
                j += 1

            avg_rank = (
                i + j - 1
            ) / 2.0

            ranks[
                order[i:j]
            ] = avg_rank

            i = j

        ranks = (
            2.0
            * ranks
            / (n - 1.0)
            - 1.0
        )

        out[
            valid_idx
        ] = ranks

        return out


    def cross_section_rank(
        self,
        matrix
    ):

        out = np.full_like(
            matrix,
            np.nan,
            dtype=np.float64
        )

        for i in range(
            matrix.shape[0]
        ):

            out[i] = (
                self.rank_row(
                    matrix[i]
                )
            )

        return out


    # ======================================================
    # 构造 signal
    # ======================================================

    def build_signal_matrix(
        self
    ):

        # 最近 N 个自然日关注度
        short_attention = (
            self.rolling_sum(
                self.attention,
                self.short_window
            )
        )

        # surprise
        surprise = (
            self.build_surprise(
                short_attention
            )
        )

        if self.signal_mode == 1:

            signal = surprise

        elif self.signal_mode == 2:

            signal = np.full_like(
                surprise,
                np.nan
            )

            lag = self.diff_lag

            signal[
                lag:
            ] = (
                surprise[
                    lag:
                ]
                - surprise[
                    :-lag
                ]
            )

        else:

            raise ValueError(
                "signalMode must be 1 or 2"
            )

        # 截面 rank
        if self.rank_mode == 1:

            signal = (
                self.cross_section_rank(
                    signal
                )
            )

        # 极端行业筛选
        if (
            self.extreme_cut
            > 0.0
        ):

            signal[
                np.abs(signal)
                < self.extreme_cut
            ] = 0.0

        # 方向
        signal = (
            signal
            * self.sign
        )

        return signal


    # ======================================================
    # 旧申万二级 -> 统一一级
    # ======================================================

    @staticmethod
    def map_industry_array(
        raw
    ):

        raw = np.asarray(
            raw,
            dtype=np.int64
        )

        level1 = (
            raw // 10000
        ) * 10000

        level1[
            raw <= 0
        ] = -1

        level1[
            raw == -999999999
        ] = -1

        # 旧采掘
        level1[
            raw == 210100
        ] = 750000

        level1[
            raw == 210200
        ] = 740000

        # 暂不强行归类
        level1[
            raw == 210300
        ] = -1

        level1[
            raw == 210400
        ] = -1

        # 石油化工
        level1[
            raw == 220100
        ] = 750000

        # 环保工程及服务
        level1[
            raw == 410400
        ] = 760000

        # 综合无文本信号
        level1[
            raw == 510100
        ] = -1

        return level1


    # ======================================================
    # 前一天自然日
    # ======================================================

    @staticmethod
    def previous_calendar_date(
        yyyymmdd
    ):

        dt = datetime.datetime.strptime(
            str(int(yyyymmdd)),
            "%Y%m%d"
        )

        dt -= datetime.timedelta(
            days=1
        )

        return int(
            dt.strftime(
                "%Y%m%d"
            )
        )


    # ======================================================
    # 找 calendar date
    # ======================================================

    def get_signal_row(
        self,
        signal_date
    ):

        pos = np.searchsorted(
            self.calendar_dates,
            signal_date
        )

        if (
            pos
            >= len(self.calendar_dates)
        ):
            return None

        if (
            int(
                self.calendar_dates[
                    pos
                ]
            )
            != signal_date
        ):
            return None

        return self.signal_matrix[
            pos
        ]


    # ======================================================
    # 每日 Alpha
    # ======================================================

    def generate(
        self,
        di
    ):

        self.alpha[:] = 0.0

        trade_date = int(
            self.sim_dates[
                di
            ]
        )

        # 严格使用前一自然日人民日报
        signal_date = (
            self.previous_calendar_date(
                trade_date
            )
        )

        signal_row = (
            self.get_signal_row(
                signal_date
            )
        )

        if signal_row is None:
            return

        raw_industry = np.asarray(
            self.industry[di]
        ).ravel()

        sw1 = (
            self.map_industry_array(
                raw_industry
            )
        )

        for j in range(
            len(self.codes)
        ):

            value = (
                signal_row[j]
            )

            if not np.isfinite(
                value
            ):
                continue

            code = int(
                self.codes[j]
            )

            self.alpha[
                sw1 == code
            ] = value