# -*- coding: utf-8 -*-

import os
import sys
import datetime
import numpy as np


CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


from alphasim import AlphaBase
from alphasim import DataRegistry as dr


class PeopleDailyIndustryAlpha(AlphaBase):
    """
    People's Daily Industry Policy Factor

    Logic
    -----
    1. 人民日报自然日行业政策信号
    2. 交易日 t 使用 t-1 自然日信号
    3. AlphaSim 申万二级行业 -> 统一一级行业
    4. 一级行业信号映射到股票

    factorType:
        1 = attention_20_rank
        2 = attention_60_rank
        3 = delta_attention_rank
        4 = surprise_20_rank
    """

    FACTOR_MAP = {
        1: "attention_20_rank",
        2: "attention_60_rank",
        3: "delta_attention_rank",
        4: "surprise_20_rank",
    }


    def __init__(self, cfg):

        super().__init__(cfg)

        # ==================================================
        # 参数
        # ==================================================

        self.factor_type = int(
            cfg.getAttributeDefault(
                "factorType",
                1
            )
        )

        if self.factor_type not in self.FACTOR_MAP:
            raise ValueError(
                "factorType must be 1,2,3,4"
            )

        self.factor_name = (
            self.FACTOR_MAP[
                self.factor_type
            ]
        )

        # ==================================================
        # AlphaSim 数据
        # ==================================================

        self.industry = dr.getData(
            "industry"
        )

        # ==================================================
        # PeopleDaily 因子
        # ==================================================

        self.factor_path = (
            "/home/dwyao/WorkPlace/"
            "PeopleDailyFactor/data/factor/"
            "people_daily_industry_factor.npz"
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
        
        print(
            "AlphaSim dates:",
            len(self.sim_dates),
            self.sim_dates[:5],
            "...",
            self.sim_dates[-5:]
        )

        print("")
        print("=" * 70)
        print("PEOPLE DAILY INDUSTRY ALPHA")
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
            "factor matrix:",
            self.factor_matrix.shape
        )

        print("=" * 70)
        print("")


    # ======================================================
    # 日期
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
            - datetime.timedelta(days=1)
        )

        return int(
            dt.strftime("%Y%m%d")
        )


    # ======================================================
    # 申万旧二级 -> 新一级
    # ======================================================

    @staticmethod
    def map_industry_array(raw):

        raw = np.asarray(
            raw,
            dtype=np.int64
        )

        # 默认按前两位得到一级
        level1 = (
            raw // 10000
        ) * 10000

        # 无效
        level1[
            raw <= 0
        ] = -1

        level1[
            raw == -999999999
        ] = -1

        # ----------------------------------------------
        # 旧采掘拆分
        # ----------------------------------------------

        # 石油开采 -> 石油石化
        level1[
            raw == 210100
        ] = 750000

        # 煤炭开采 -> 煤炭
        level1[
            raw == 210200
        ] = 740000

        # 其他采掘：暂不映射
        level1[
            raw == 210300
        ] = -1

        # 采掘服务：二级层面有歧义
        level1[
            raw == 210400
        ] = -1

        # ----------------------------------------------
        # 旧化工中的石油化工
        # ----------------------------------------------

        level1[
            raw == 220100
        ] = 750000

        # ----------------------------------------------
        # 旧公用事业中的环保
        # ----------------------------------------------

        level1[
            raw == 410400
        ] = 760000

        # ----------------------------------------------
        # 综合无 PeopleDaily 信号
        # ----------------------------------------------

        level1[
            raw == 510100
        ] = -1

        return level1


    # ======================================================
    # 找人民日报自然日
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

        # 每日先清零
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
        # 长假后 -> 开盘前一自然日
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
            return

        # ----------------------------------------------
        # 当前股票行业
        # ----------------------------------------------

        raw_industry = np.asarray(
            self.industry[di]
        ).ravel()

        sw1 = self.map_industry_array(
            raw_industry
        )

        # ----------------------------------------------
        # 行业 -> 股票
        # ----------------------------------------------

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

            self.alpha[
                mask
            ] = -value

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