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


class PeopleDailyAIPolicyAlpha(AlphaBase):

    # =========================================================
    # Existing text factors
    # =========================================================

    FACTOR_MAP = {

        1: "positive_surprise_20_rank",
        2: "negative_surprise_20_rank",
        3: "net_surprise_20_rank",
        4: "net_attention_20_rank",

        5: "net_surprise_10_rank",
        6: "net_surprise_40_rank",
        7: "net_ewm10_surprise_rank",

        8: "quality_surprise_20_rank",
        9: "ai_residual_rank",
        10: "high_conf_negative_20_rank",
        11: "constraint_surprise_20_rank",

        12: "direction_residual_rank",
        13: "quality_residual_rank",

        14: "quality_surprise_30_rank",
        15: "quality_surprise_40_rank",

        16: "net_surprise_20_40_combo_rank",

        17: "combo16_quality_residual_rank",

        18: "quality_residual_ewm5_rank",

        19: "combo16_quality20_rank",

        20: "factor17_ewm3_rank",
        21: "factor17_ewm5_rank",

        # current text benchmark
        22: "combo17_factor18_rank",
    }


    def __init__(self, cfg):

        super().__init__(cfg)

        # =====================================================
        # Basic parameters
        # =====================================================

        self.factor_type = int(
            cfg.getAttributeDefault(
                "factorType",
                22
            )
        )

        if self.factor_type not in self.FACTOR_MAP:
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
                0.3
            )
        )

        self.debug = int(
            cfg.getAttributeDefault(
                "debug",
                0
            )
        )

        # =====================================================
        # Price enhancement
        #
        # 0 = no price enhancement
        #
        # 1 = residualize factor22 against
        #     industry momentum 20/60/120
        #
        # 2 = factor22 + price momentum
        #     then sign=-1
        #     => policy crowding + price crowding reversal
        #
        # 3 = state conditioning:
        #     policy and price same sign => strengthen
        #     opposite sign => weaken
        # =====================================================

        self.price_mode = int(
            cfg.getAttributeDefault(
                "priceMode",
                0
            )
        )

        self.momentum_window = int(
            cfg.getAttributeDefault(
                "momentumWindow",
                60
            )
        )

        self.price_weight = float(
            cfg.getAttributeDefault(
                "priceWeight",
                0.25
            )
        )

        self.confirm_boost = float(
            cfg.getAttributeDefault(
                "confirmBoost",
                0.50
            )
        )

        self.conflict_penalty = float(
            cfg.getAttributeDefault(
                "conflictPenalty",
                0.50
            )
        )

        # =====================================================
        # AlphaSim stock data
        # =====================================================

        self.industry = dr.getData(
            "industry"
        )

        # Available for later experiments
        self.sector = dr.getData(
            "sector"
        )

        self.adj_close = dr.getData(
            "adj_close"
        )

        self.adj_open = dr.getData(
            "adj_open"
        )

        self.returns = dr.getData(
            "returns"
        )

        # =====================================================
        # Load PeopleDaily factor
        # =====================================================

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

        if self.factor_name not in factor_data.files:

            raise KeyError(
                "%s not found in NPZ"
                % self.factor_name
            )

        self.factor_matrix = (
            factor_data[
                self.factor_name
            ]
            .astype(np.float64)
        )

        # =====================================================
        # AlphaSim dates
        # =====================================================

        self.sim_dates = np.memmap(
            "/data/simucache/"
            "__universe/Dates.npy",
            dtype=np.int64,
            mode="r"
        )

        print("")
        print("=" * 75)
        print("PEOPLE DAILY AI + PRICE ALPHA")
        print("=" * 75)

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
            "priceMode:",
            self.price_mode
        )

        print(
            "momentumWindow:",
            self.momentum_window
        )

        print(
            "priceWeight:",
            self.price_weight
        )

        print(
            "confirmBoost:",
            self.confirm_boost
        )

        print(
            "conflictPenalty:",
            self.conflict_penalty
        )

        print("=" * 75)
        print("")


    # =========================================================
    # Calendar
    # =========================================================

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


    # =========================================================
    # Old industry code -> SW1
    # =========================================================

    @staticmethod
    def map_industry_array(raw):

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

        # old chemical
        level1[
            raw == 220100
        ] = 750000

        # environmental
        level1[
            raw == 410400
        ] = 760000

        # comprehensive
        level1[
            raw == 510100
        ] = -1

        return level1


    # =========================================================
    # Cross-sectional rank
    #
    # finite values -> approximately [-1, +1]
    # =========================================================

    @staticmethod
    def cs_rank(values):

        values = np.asarray(
            values,
            dtype=np.float64
        )

        result = np.full(
            len(values),
            np.nan,
            dtype=np.float64
        )

        valid = np.isfinite(
            values
        )

        idx = np.where(
            valid
        )[0]

        n = len(idx)

        if n <= 1:
            return result

        order = np.argsort(
            values[
                idx
            ]
        )

        ranks = np.empty(
            n,
            dtype=np.float64
        )

        ranks[
            order
        ] = np.arange(
            n,
            dtype=np.float64
        )

        # exactly [-1, +1]
        ranks = (
            2.0
            * ranks
            / float(
                n - 1
            )
            - 1.0
        )

        result[
            idx
        ] = ranks

        return result


    # =========================================================
    # Robust industry mean
    #
    # Winsorize stock lookback returns within industry.
    # =========================================================

    @staticmethod
    def robust_mean(x):

        x = np.asarray(
            x,
            dtype=np.float64
        )

        x = x[
            np.isfinite(x)
        ]

        if len(x) == 0:
            return np.nan

        # remove impossible/extreme bad data
        x = x[
            (x > -0.95)
            &
            (x < 5.0)
        ]

        if len(x) == 0:
            return np.nan

        if len(x) >= 10:

            lo = np.percentile(
                x,
                5
            )

            hi = np.percentile(
                x,
                95
            )

            x = np.clip(
                x,
                lo,
                hi
            )

        return float(
            np.mean(
                x
            )
        )


    # =========================================================
    # Get PeopleDaily factor
    # =========================================================

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

        return (
            self.factor_matrix[
                pos
            ]
            .copy()
        )


    # =========================================================
    # Industry momentum
    #
    # trade date = di
    #
    # only use prices through di-1.
    #
    # stock return:
    #
    # close[di-1] / close[di-1-window] - 1
    #
    # Then equal-weight aggregate by SW1.
    # =========================================================

    def calc_industry_momentum(
        self,
        di,
        window
    ):

        result = np.full(
            len(
                self.factor_codes
            ),
            np.nan,
            dtype=np.float64
        )

        end_di = (
            di - 1
        )

        start_di = (
            end_di - window
        )

        if start_di < 0:
            return result

        close_now = np.asarray(
            self.adj_close[
                end_di
            ],
            dtype=np.float64
        ).ravel()

        close_old = np.asarray(
            self.adj_close[
                start_di
            ],
            dtype=np.float64
        ).ravel()

        # Use yesterday's industry classification.
        raw_industry = np.asarray(
            self.industry[
                end_di
            ]
        ).ravel()

        sw1 = (
            self.map_industry_array(
                raw_industry
            )
        )

        valid_price = (
            np.isfinite(
                close_now
            )
            &
            np.isfinite(
                close_old
            )
            &
            (
                close_now > 0
            )
            &
            (
                close_old > 0
            )
        )

        stock_ret = np.full(
            len(
                close_now
            ),
            np.nan,
            dtype=np.float64
        )

        stock_ret[
            valid_price
        ] = (
            close_now[
                valid_price
            ]
            /
            close_old[
                valid_price
            ]
            - 1.0
        )

        # Optional universe filter
        try:

            valid_universe = np.asarray(
                self.valid[
                    end_di
                ]
            ).ravel().astype(
                bool
            )

        except Exception:

            valid_universe = np.ones(
                len(stock_ret),
                dtype=bool
            )

        for j, code in enumerate(
            self.factor_codes
        ):

            mask = (
                (sw1 == int(code))
                &
                valid_price
                &
                valid_universe
            )

            if not np.any(
                mask
            ):
                continue

            result[
                j
            ] = (
                self.robust_mean(
                    stock_ret[
                        mask
                    ]
                )
            )

        return result


    # =========================================================
    # Residualize text factor against price momentum
    #
    # y = text factor
    #
    # X =
    #   industry momentum rank 20
    #   industry momentum rank 60
    #   industry momentum rank 120
    #
    # output = residual rank
    # =========================================================

    def price_residual_factor(
        self,
        di,
        factor_row
    ):

        mom20 = self.cs_rank(
            self.calc_industry_momentum(
                di,
                20
            )
        )

        mom60 = self.cs_rank(
            self.calc_industry_momentum(
                di,
                60
            )
        )

        mom120 = self.cs_rank(
            self.calc_industry_momentum(
                di,
                120
            )
        )

        y = np.asarray(
            factor_row,
            dtype=np.float64
        )

        valid = (
            np.isfinite(y)
            &
            np.isfinite(mom20)
            &
            np.isfinite(mom60)
            &
            np.isfinite(mom120)
        )

        output = np.full(
            len(y),
            np.nan,
            dtype=np.float64
        )

        n = int(
            np.sum(
                valid
            )
        )

        if n < 10:
            return output

        X = np.column_stack(
            [
                np.ones(n),
                mom20[
                    valid
                ],
                mom60[
                    valid
                ],
                mom120[
                    valid
                ],
            ]
        )

        beta = np.linalg.lstsq(
            X,
            y[
                valid
            ],
            rcond=None
        )[0]

        residual = (
            y[
                valid
            ]
            -
            np.dot(
                X,
                beta
            )
        )

        output[
            valid
        ] = residual

        return self.cs_rank(
            output
        )


    # =========================================================
    # PriceMode 2
    #
    # Price + policy crowding reversal
    #
    # Example:
    #
    # factor22 high
    # + previous 60d return high
    # = strongest crowding
    # = final sign=-1 -> strongest short
    #
    # factor22 low
    # + previous 60d return low
    # = strongest cold / laggard industry
    # = sign=-1 -> strongest long
    # =========================================================

    def price_blend_factor(
        self,
        di,
        factor_row
    ):

        momentum = (
            self.calc_industry_momentum(
                di,
                self.momentum_window
            )
        )

        momentum_rank = (
            self.cs_rank(
                momentum
            )
        )

        output = np.full(
            len(
                factor_row
            ),
            np.nan,
            dtype=np.float64
        )

        valid = (
            np.isfinite(
                factor_row
            )
            &
            np.isfinite(
                momentum_rank
            )
        )

        output[
            valid
        ] = (
            (
                1.0
                -
                self.price_weight
            )
            *
            factor_row[
                valid
            ]
            +
            self.price_weight
            *
            momentum_rank[
                valid
            ]
        )

        return self.cs_rank(
            output
        )


    # =========================================================
    # PriceMode 3
    #
    # Conditional interaction
    #
    # If policy crowding and price momentum have same sign:
    # strengthen signal.
    #
    # If conflicting:
    # weaken signal.
    # =========================================================

    def price_condition_factor(
        self,
        di,
        factor_row
    ):

        momentum = (
            self.calc_industry_momentum(
                di,
                self.momentum_window
            )
        )

        momentum_rank = (
            self.cs_rank(
                momentum
            )
        )

        output = np.full(
            len(
                factor_row
            ),
            np.nan,
            dtype=np.float64
        )

        valid = (
            np.isfinite(
                factor_row
            )
            &
            np.isfinite(
                momentum_rank
            )
        )

        for j in np.where(
            valid
        )[0]:

            f = float(
                factor_row[
                    j
                ]
            )

            m = float(
                momentum_rank[
                    j
                ]
            )

            # Same direction:
            #
            # positive policy crowding + prior rally
            # or
            # negative policy crowding + prior weakness
            #
            # strengthen reversal signal
            if (
                f * m
                >
                0
            ):

                multiplier = (
                    1.0
                    +
                    self.confirm_boost
                    *
                    abs(m)
                )

            # conflicting policy / price state
            else:

                multiplier = (
                    1.0
                    -
                    self.conflict_penalty
                    *
                    abs(m)
                )

                multiplier = max(
                    multiplier,
                    0.0
                )

            output[
                j
            ] = (
                f
                *
                multiplier
            )

        return self.cs_rank(
            output
        )


    # =========================================================
    # Apply price enhancement
    # =========================================================

    def apply_price_model(
        self,
        di,
        factor_row
    ):

        if self.price_mode == 0:

            return factor_row

        elif self.price_mode == 1:

            return (
                self.price_residual_factor(
                    di,
                    factor_row
                )
            )

        elif self.price_mode == 2:

            return (
                self.price_blend_factor(
                    di,
                    factor_row
                )
            )

        elif self.price_mode == 3:

            return (
                self.price_condition_factor(
                    di,
                    factor_row
                )
            )

        else:

            raise ValueError(
                "priceMode must be 0,1,2,3"
            )


    # =========================================================
    # Generate
    # =========================================================

    def generate(
        self,
        di
    ):

        self.alpha[:] = 0.0

        # =====================================================
        # Trade date
        # =====================================================

        trade_date = int(
            self.sim_dates[
                di
            ]
        )

        # People's Daily always uses t-1 calendar day
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

        # =====================================================
        # Add price information
        #
        # Price only uses through previous trading day.
        # =====================================================

        factor_row = (
            self.apply_price_model(
                di,
                factor_row
            )
        )

        # =====================================================
        # Current industry mapping
        #
        # Use previous trading day's classification to avoid
        # accidentally using information from trade date.
        # =====================================================

        industry_di = max(
            di - 1,
            0
        )

        raw_industry = np.asarray(
            self.industry[
                industry_di
            ]
        ).ravel()

        sw1 = (
            self.map_industry_array(
                raw_industry
            )
        )

        assigned_count = 0

        # =====================================================
        # Industry -> stock
        # =====================================================

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

            # weak industry signal filter
            if (
                abs(value)
                <
                self.extreme_cut
            ):
                continue

            mask = (
                sw1
                ==
                code
            )

            if not np.any(
                mask
            ):
                continue

            self.alpha[
                mask
            ] = (
                self.sign
                *
                value
            )

            assigned_count += int(
                np.sum(
                    mask
                )
            )

        # =====================================================
        # Universe
        # =====================================================

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

        # =====================================================
        # Debug
        # =====================================================

        if self.debug:

            print(
                "[PeopleDailyPrice]",
                "trade=",
                trade_date,
                "signal=",
                signal_date,
                "factor=",
                self.factor_type,
                "priceMode=",
                self.price_mode,
                "momWindow=",
                self.momentum_window,
                "priceWeight=",
                self.price_weight,
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