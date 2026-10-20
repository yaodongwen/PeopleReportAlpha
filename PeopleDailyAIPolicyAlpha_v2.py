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
    People's Daily AI Policy Industry Factor

    trade date t
        -> use t-1 calendar day

    sign:
        1  = original direction
        -1 = reverse direction

    extremeCut:
        0.0 = all industries
        0.3 = abs(rank) >= 0.3
        0.5 = abs(rank) >= 0.5

    Current research baseline:
        factor17, sign=-1
    """

    FACTOR_MAP = {

        1:
            "positive_surprise_20_rank",

        2:
            "negative_surprise_20_rank",

        3:
            "net_surprise_20_rank",

        4:
            "net_attention_20_rank",

        5:
            "net_surprise_10_rank",

        6:
            "net_surprise_40_rank",

        7:
            "net_ewm10_surprise_rank",

        8:
            "quality_surprise_20_rank",

        9:
            "ai_residual_rank",

        10:
            "high_conf_negative_20_rank",

        11:
            "constraint_surprise_20_rank",

        12:
            "direction_residual_rank",

        13:
            "quality_residual_rank",

        14:
            "quality_surprise_30_rank",

        15:
            "quality_surprise_40_rank",

        16:
            "net_surprise_20_40_combo_rank",

        17:
            "combo16_quality_residual_rank",

        18:
            "quality_residual_ewm5_rank",

        19:
            "combo16_quality20_rank",

        20:
            "factor17_ewm3_rank",

        21:
            "factor17_ewm5_rank",

        22:
            "combo17_factor18_rank",
    }


    def __init__(self, cfg):

        super().__init__(cfg)

        # ==================================================
        # Parameters
        # ==================================================

        self.factor_type = int(
            cfg.getAttributeDefault(
                "factorType",
                17
            )
        )

        if (
            self.factor_type
            not in self.FACTOR_MAP
        ):

            raise ValueError(
                "factorType must be 1-22"
            )

        self.factor_name = (
            self.FACTOR_MAP[
                self.factor_type
            ]
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
                0
            )
        )

        # ==================================================
        # AlphaSim data
        # ==================================================

        self.industry = dr.getData(
            "industry"
        )

        # ==================================================
        # PeopleDaily factor
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
            factor_data[
                "dates"
            ]
            .astype(np.int64)
        )

        self.factor_codes = (
            factor_data[
                "codes"
            ]
            .astype(np.int64)
        )

        if (
            self.factor_name
            not in factor_data.files
        ):

            raise KeyError(
                "%s not found in NPZ. "
                "Available=%s"
                % (
                    self.factor_name,
                    factor_data.files,
                )
            )

        self.factor_matrix = (
            factor_data[
                self.factor_name
            ]
            .astype(np.float64)
        )

        # ==================================================
        # Dimension checks
        # ==================================================

        if (
            self.factor_matrix.shape[0]
            != len(
                self.factor_dates
            )
        ):

            raise ValueError(
                "date dimension mismatch"
            )

        if (
            self.factor_matrix.shape[1]
            != len(
                self.factor_codes
            )
        ):

            raise ValueError(
                "industry dimension mismatch"
            )

        # ==================================================
        # AlphaSim dates
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

        print("")
        print("=" * 72)
        print("PEOPLE DAILY AI POLICY ALPHA")
        print("=" * 72)

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
            "extremeCut:",
            self.extreme_cut
        )

        print(
            "factor dates:",
            int(
                self.factor_dates[0]
            ),
            "->",
            int(
                self.factor_dates[-1]
            )
        )

        print(
            "factor matrix:",
            self.factor_matrix.shape
        )

        print(
            "industry codes:",
            self.factor_codes
        )

        print("=" * 72)
        print("")


    # ======================================================
    # Previous calendar day
    # ======================================================

    @staticmethod
    def previous_calendar_date(
        yyyymmdd
    ):

        dt = datetime.datetime.strptime(
            str(
                int(
                    yyyymmdd
                )
            ),
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
    # Old SW2 -> unified SW1
    # ======================================================

    @staticmethod
    def map_industry_array(raw):

        raw = np.asarray(
            raw,
            dtype=np.int64
        )

        level1 = (
            raw // 10000
        ) * 10000

        # invalid
        level1[
            raw <= 0
        ] = -1

        level1[
            raw == -999999999
        ] = -1

        # old mining
        level1[
            raw == 210100
        ] = 750000

        level1[
            raw == 210200
        ] = 740000

        level1[
            raw == 210300
        ] = -1

        level1[
            raw == 210400
        ] = -1

        # old chemical -> petrochemical
        level1[
            raw == 220100
        ] = 750000

        # old utility -> environmental
        level1[
            raw == 410400
        ] = 760000

        # comprehensive
        level1[
            raw == 510100
        ] = -1

        return level1


    # ======================================================
    # Get factor row
    # ======================================================

    def get_factor_row(
        self,
        signal_date
    ):

        pos = np.searchsorted(
            self.factor_dates,
            signal_date
        )

        if (
            pos
            >= len(
                self.factor_dates
            )
        ):

            return None

        if (
            self.factor_dates[
                pos
            ]
            != signal_date
        ):

            return None

        return self.factor_matrix[
            pos
        ]


    # ======================================================
    # Generate Alpha
    # ======================================================

    def generate(self, di):

        self.alpha[:] = 0.0

        # ----------------------------------------------
        # Trade date
        # ----------------------------------------------

        trade_date = int(
            self.sim_dates[
                di
            ]
        )

        # ----------------------------------------------
        # t uses t-1 calendar day
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
        # Industry
        # ----------------------------------------------

        raw_industry = np.asarray(
            self.industry[
                di
            ]
        ).ravel()

        sw1 = (
            self.map_industry_array(
                raw_industry
            )
        )

        assigned_count = 0

        # ----------------------------------------------
        # Industry -> stock
        # ----------------------------------------------

        for j in range(
            len(
                self.factor_codes
            )
        ):

            code = int(
                self.factor_codes[
                    j
                ]
            )

            value = float(
                factor_row[
                    j
                ]
            )

            if not np.isfinite(
                value
            ):
                continue

            # ==========================================
            # Extreme signal filter
            # ==========================================

            if (
                abs(value)
                <
                self.extreme_cut
            ):
                continue

            mask = (
                sw1 == code
            )

            if not np.any(
                mask
            ):
                continue

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
                self.valid[
                    di
                ]
            ).ravel().astype(
                bool
            )

            self.alpha[
                ~valid
            ] = 0.0

        except Exception:

            pass

        # ----------------------------------------------
        # Debug
        # ----------------------------------------------

        if self.debug:

            print(
                "[PeopleDailyAI]",
                "trade=",
                trade_date,
                "signal=",
                signal_date,
                "type=",
                self.factor_type,
                "cut=",
                self.extreme_cut,
                "assigned=",
                assigned_count,
                "nonzero=",
                int(
                    np.count_nonzero(
                        self.alpha
                    )
                ),
            )