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

    # Offline factors stored in NPZ

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



        # Current text benchmark

        22: "combo17_factor18_rank",

    }



    # =========================================================

    # Runtime price-enhanced factors

    #

    # These all use Factor22 as text base.

    # =========================================================



    RUNTIME_FACTOR_MODE = {

        # Factor22 conditioned on 5-day industry return
        23: 4,

        # Factor22 conditioned on 5-day acceleration
        24: 5,

        # Factor22 conditioned on 5-day overnight return
        25: 6,

        # Factor22 conditioned on 5-day intraday return
        26: 7,

        # Factor22 long-side veto by weak 5-day return
        27: 8,

        # Factor22 long-side veto by weak 5-day acceleration
        28: 9,

        # Factor22 long-side veto by weak 5-day intraday return
        29: 10,

        # Factor22 long-side veto by weak 5-day overnight return
        30: 11,

        # Factor22 long-side veto when >=2 price states are weak
        31: 12,

    }





    def __init__(self, cfg):



        super().__init__(cfg)



        # =====================================================

        # Parameters

        # =====================================================



        self.factor_type = int(

            cfg.getAttributeDefault(

                "factorType",

                22

            )

        )



        if (

            self.factor_type not in self.FACTOR_MAP

            and

            self.factor_type not in self.RUNTIME_FACTOR_MODE

        ):

            raise ValueError(

                "factorType must be 1-31"

            )



        # -----------------------------------------------------

        # factor23-31 all start from factor22

        # -----------------------------------------------------



        if self.factor_type in self.RUNTIME_FACTOR_MODE:



            self.factor_name = (

                self.FACTOR_MAP[22]

            )



            self.price_mode = (

                self.RUNTIME_FACTOR_MODE[

                    self.factor_type

                ]

            )



        else:



            self.factor_name = (

                self.FACTOR_MAP[

                    self.factor_type

                ]

            )



            self.price_mode = int(

                cfg.getAttributeDefault(

                    "priceMode",

                    0

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

                0.3

            )

        )



        # -----------------------------------------------------

        # Price reaction parameters

        # -----------------------------------------------------



        self.reaction_window = int(

            cfg.getAttributeDefault(

                "reactionWindow",

                5

            )

        )



        # Rank is [-1, 1].

        #

        # 0.6 approximately means:

        # only top/bottom ~20% price-state industries

        self.price_extreme = float(

            cfg.getAttributeDefault(

                "priceExtreme",

                0.6

            )

        )



        # Same direction:

        # strengthen Factor22

        self.confirm_boost = float(

            cfg.getAttributeDefault(

                "confirmBoost",

                0.5

            )

        )



        # Opposite direction:

        # weaken Factor22

        self.conflict_penalty = float(

            cfg.getAttributeDefault(

                "conflictPenalty",

                0.5

            )

        )


        # Long-side veto parameters for Factor27-31

        self.veto_threshold = float(

            cfg.getAttributeDefault(

                "vetoThreshold",

                -0.6

            )

        )



        self.veto_min_count = int(

            cfg.getAttributeDefault(

                "vetoMinCount",

                2

            )

        )



        self.debug = int(

            cfg.getAttributeDefault(

                "debug",

                0

            )

        )



        # =====================================================

        # AlphaSim data

        # =====================================================



        self.industry = dr.getData(

            "industry"

        )



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

        # Load PeopleDaily NPZ

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

        print("=" * 78)

        print("PEOPLE DAILY AI + MARKET REACTION ALPHA")

        print("=" * 78)



        print(

            "factorType:",

            self.factor_type

        )



        print(

            "baseFactor:",

            self.factor_name

        )



        print(

            "priceMode:",

            self.price_mode

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

            "reactionWindow:",

            self.reaction_window

        )



        print(

            "priceExtreme:",

            self.price_extreme

        )



        print(

            "confirmBoost:",

            self.confirm_boost

        )



        print(

            "conflictPenalty:",

            self.conflict_penalty

        )


        print(

            "vetoThreshold:",

            self.veto_threshold

        )



        print(

            "vetoMinCount:",

            self.veto_min_count

        )



        print("=" * 78)

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

    # Old industry -> SW1

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



        # chemical -> petrochemical

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

    # Cross-sectional rank [-1, 1]

    # =========================================================



    @staticmethod

    def cs_rank(values):



        x = np.asarray(

            values,

            dtype=np.float64

        )



        result = np.full(

            len(x),

            np.nan,

            dtype=np.float64

        )



        valid = np.isfinite(

            x

        )



        idx = np.where(

            valid

        )[0]



        n = len(idx)



        if n <= 1:

            return result



        xv = x[

            idx

        ]



        # -----------------------------------------------------

        # Average ranks for ties

        # -----------------------------------------------------



        order = np.argsort(

            xv,

            kind="mergesort"

        )



        sorted_x = xv[

            order

        ]



        ranks_sorted = np.empty(

            n,

            dtype=np.float64

        )



        start = 0



        while start < n:



            end = start + 1



            while (

                end < n

                and

                sorted_x[end] == sorted_x[start]

            ):

                end += 1



            avg_rank = (

                start

                +

                end

                -

                1

            ) / 2.0



            ranks_sorted[

                start:end

            ] = avg_rank



            start = end



        ranks = np.empty(

            n,

            dtype=np.float64

        )



        ranks[

            order

        ] = ranks_sorted



        ranks = (

            2.0

            *

            ranks

            /

            float(n - 1)

            -

            1.0

        )



        result[

            idx

        ] = ranks



        return result





    # =========================================================

    # Robust mean

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



        # discard obviously bad return observations

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

            np.mean(x)

        )





    # =========================================================

    # Factor row

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

            >= len(self.factor_dates)

        ):

            return None



        if (

            self.factor_dates[pos]

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

    # Aggregate stock observations -> SW1 industries

    # =========================================================



    def aggregate_to_industry(

        self,

        stock_values,

        industry_di,

        universe_di=None

    ):



        result = np.full(

            len(self.factor_codes),

            np.nan,

            dtype=np.float64

        )



        stock_values = np.asarray(

            stock_values,

            dtype=np.float64

        ).ravel()



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



        finite = np.isfinite(

            stock_values

        )



        if universe_di is not None:



            try:



                valid_universe = np.asarray(

                    self.valid[

                        universe_di

                    ]

                ).ravel().astype(bool)



            except Exception:



                valid_universe = np.ones(

                    len(stock_values),

                    dtype=bool

                )



        else:



            valid_universe = np.ones(

                len(stock_values),

                dtype=bool

            )



        for j, code in enumerate(

            self.factor_codes

        ):



            mask = (

                (sw1 == int(code))

                &

                finite

                &

                valid_universe

            )



            if not np.any(mask):

                continue



            result[j] = (

                self.robust_mean(

                    stock_values[

                        mask

                    ]

                )

            )



        return result





    # =========================================================

    # N trading-day stock close return

    #

    # Uses only through di-1.

    # =========================================================



    def calc_industry_return(

        self,

        di,

        window

    ):



        end_di = di - 1



        start_di = (

            end_di

            -

            window

        )



        result = np.full(

            len(self.factor_codes),

            np.nan,

            dtype=np.float64

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



        stock_ret = np.full(

            len(close_now),

            np.nan,

            dtype=np.float64

        )



        valid = (

            np.isfinite(close_now)

            &

            np.isfinite(close_old)

            &

            (close_now > 0)

            &

            (close_old > 0)

        )



        stock_ret[

            valid

        ] = (

            close_now[

                valid

            ]

            /

            close_old[

                valid

            ]

            -

            1.0

        )



        return (

            self.aggregate_to_industry(

                stock_ret,

                industry_di=end_di,

                universe_di=end_di,

            )

        )





    # =========================================================

    # Factor24:

    #

    # Short-term acceleration

    #

    # ret5 - ret20 / 4

    #

    # Positive:

    # recent 5 days are much stronger than

    # the average pace implied by 20-day return.

    # =========================================================



    def calc_industry_acceleration(

        self,

        di

    ):



        ret5 = (

            self.calc_industry_return(

                di,

                5

            )

        )



        ret20 = (

            self.calc_industry_return(

                di,

                20

            )

        )



        result = np.full(

            len(ret5),

            np.nan,

            dtype=np.float64

        )



        valid = (

            np.isfinite(ret5)

            &

            np.isfinite(ret20)

        )



        result[

            valid

        ] = (

            ret5[

                valid

            ]

            -

            ret20[

                valid

            ]

            / 4.0

        )



        return result





    # =========================================================

    # Daily overnight return

    #

    # open[d] / close[d-1] - 1

    #

    # d <= di-1

    # so no lookahead.

    # =========================================================



    def calc_daily_industry_overnight(

        self,

        d

    ):



        result = np.full(

            len(self.factor_codes),

            np.nan,

            dtype=np.float64

        )



        if d <= 0:

            return result



        open_now = np.asarray(

            self.adj_open[

                d

            ],

            dtype=np.float64

        ).ravel()



        close_prev = np.asarray(

            self.adj_close[

                d - 1

            ],

            dtype=np.float64

        ).ravel()



        stock_ret = np.full(

            len(open_now),

            np.nan,

            dtype=np.float64

        )



        valid = (

            np.isfinite(open_now)

            &

            np.isfinite(close_prev)

            &

            (open_now > 0)

            &

            (close_prev > 0)

        )



        stock_ret[

            valid

        ] = (

            open_now[

                valid

            ]

            /

            close_prev[

                valid

            ]

            -

            1.0

        )



        return (

            self.aggregate_to_industry(

                stock_ret,

                industry_di=d - 1,

                universe_di=d - 1,

            )

        )





    # =========================================================

    # Daily intraday return

    #

    # close[d] / open[d] - 1

    #

    # d <= di-1

    # =========================================================



    def calc_daily_industry_intraday(

        self,

        d

    ):



        close_now = np.asarray(

            self.adj_close[

                d

            ],

            dtype=np.float64

        ).ravel()



        open_now = np.asarray(

            self.adj_open[

                d

            ],

            dtype=np.float64

        ).ravel()



        stock_ret = np.full(

            len(close_now),

            np.nan,

            dtype=np.float64

        )



        valid = (

            np.isfinite(close_now)

            &

            np.isfinite(open_now)

            &

            (close_now > 0)

            &

            (open_now > 0)

        )



        stock_ret[

            valid

        ] = (

            close_now[

                valid

            ]

            /

            open_now[

                valid

            ]

            -

            1.0

        )



        return (

            self.aggregate_to_industry(

                stock_ret,

                industry_di=max(

                    d - 1,

                    0

                ),

                universe_di=d,

            )

        )





    # =========================================================

    # Average industry market reaction over recent N days

    # =========================================================



    def calc_recent_reaction(

        self,

        di,

        reaction_type

    ):



        rows = []



        start_di = max(

            1,

            di - self.reaction_window

        )



        end_di = di - 1



        for d in range(

            start_di,

            end_di + 1

        ):



            if reaction_type == "overnight":



                row = (

                    self.calc_daily_industry_overnight(

                        d

                    )

                )



            elif reaction_type == "intraday":



                row = (

                    self.calc_daily_industry_intraday(

                        d

                    )

                )



            else:



                raise ValueError(

                    "invalid reaction_type"

                )



            rows.append(

                row

            )



        if not rows:



            return np.full(

                len(self.factor_codes),

                np.nan,

                dtype=np.float64

            )



        matrix = np.vstack(

            rows

        )



        result = np.full(

            matrix.shape[1],

            np.nan,

            dtype=np.float64

        )



        for j in range(

            matrix.shape[1]

        ):



            x = matrix[:, j]



            x = x[

                np.isfinite(x)

            ]



            if len(x) == 0:

                continue



            result[j] = (

                np.mean(x)

            )



        return result





    # =========================================================

    # Generic state conditioning

    #

    # Only modify Factor22 when the price state itself

    # is cross-sectionally extreme.

    #

    # factor > 0 + state > 0:

    #   policy crowding + strong price reaction

    #   strengthen eventual reversal

    #

    # factor > 0 + state < 0:

    #   policy and market disagree

    #   weaken reversal

    #

    # Same for negative side symmetrically.

    # =========================================================



    def condition_factor(

        self,

        factor_row,

        state_values

    ):



        state_rank = (

            self.cs_rank(

                state_values

            )

        )



        output = (

            np.asarray(

                factor_row,

                dtype=np.float64

            )

            .copy()

        )



        valid = (

            np.isfinite(output)

            &

            np.isfinite(state_rank)

        )



        for j in np.where(

            valid

        )[0]:



            f = float(

                output[j]

            )



            s = float(

                state_rank[j]

            )



            # ---------------------------------------------

            # Non-extreme market state:

            # do not touch Factor22

            # ---------------------------------------------



            if abs(s) < self.price_extreme:

                continue



            # ---------------------------------------------

            # Price confirms policy state

            # ---------------------------------------------



            if f * s > 0:



                multiplier = (

                    1.0

                    +

                    self.confirm_boost

                    *

                    abs(s)

                )



            # ---------------------------------------------

            # Price conflicts with policy state

            # ---------------------------------------------



            elif f * s < 0:



                multiplier = (

                    1.0

                    -

                    self.conflict_penalty

                    *

                    abs(s)

                )



                multiplier = max(

                    multiplier,

                    0.0

                )



            else:



                multiplier = 1.0



            output[j] = (

                f

                *

                multiplier

            )



        # Final daily industry rank

        return (

            self.cs_rank(

                output

            )

        )





    # =========================================================

    # Asymmetric long-side veto for Factor27-30

    # =========================================================



    def long_side_veto(

        self,

        factor_row,

        state_values

    ):



        factor_row = np.asarray(

            factor_row,

            dtype=np.float64

        ).copy()



        state_rank = self.cs_rank(

            state_values

        )



        output = factor_row.copy()



        valid = (

            np.isfinite(output)

            &

            np.isfinite(state_rank)

        )



        for j in np.where(valid)[0]:



            f = float(

                factor_row[j]

            )



            s = float(

                state_rank[j]

            )



            # sign=-1 means factor22 < 0 becomes the long side.
            # Only cancel longs when the corresponding price state
            # is cross-sectionally extremely weak.  Short side is
            # intentionally left unchanged.

            if (

                f < 0.0

                and

                s <= self.veto_threshold

            ):

                output[j] = 0.0



        # Deliberately DO NOT rerank here.  We want a clean veto
        # experiment: unaffected Factor22 industries keep exactly
        # their original signal values.

        return output





    # =========================================================

    # Multi-state long-side veto for Factor31

    # =========================================================



    def multi_state_long_veto(

        self,

        di,

        factor_row

    ):



        ret5 = self.calc_industry_return(

            di,

            5

        )



        accel5 = self.calc_industry_acceleration(

            di

        )



        overnight5 = self.calc_recent_reaction(

            di,

            "overnight"

        )



        intraday5 = self.calc_recent_reaction(

            di,

            "intraday"

        )



        ret5_rank = self.cs_rank(ret5)

        accel5_rank = self.cs_rank(accel5)

        overnight5_rank = self.cs_rank(overnight5)

        intraday5_rank = self.cs_rank(intraday5)



        factor_row = np.asarray(

            factor_row,

            dtype=np.float64

        ).copy()



        output = factor_row.copy()



        for j in range(len(factor_row)):



            f = factor_row[j]



            if not np.isfinite(f):

                continue



            # Only inspect Factor22 long candidates.

            if f >= 0.0:

                continue



            weak_count = 0



            states = [

                ret5_rank[j],

                accel5_rank[j],

                overnight5_rank[j],

                intraday5_rank[j],

            ]



            for state in states:

                if (

                    np.isfinite(state)

                    and

                    state <= self.veto_threshold

                ):

                    weak_count += 1



            if weak_count >= self.veto_min_count:

                output[j] = 0.0



        # Same as Factor27-30: no reranking after veto.

        return output





    # =========================================================

    # Runtime factors 23-31

    # =========================================================



    def apply_runtime_factor(

        self,

        di,

        factor_row

    ):



        # -----------------------------------------------------

        # Ordinary offline factor

        # -----------------------------------------------------



        if self.factor_type <= 22:



            return factor_row



        # -----------------------------------------------------

        # Factor23:

        # Factor22 × recent 5d return state

        # -----------------------------------------------------



        if self.factor_type == 23:



            state = (

                self.calc_industry_return(

                    di,

                    self.reaction_window

                )

            )



            return (

                self.condition_factor(

                    factor_row,

                    state

                )

            )



        # -----------------------------------------------------

        # Factor24:

        # Factor22 × 5d acceleration

        # -----------------------------------------------------



        elif self.factor_type == 24:



            state = (

                self.calc_industry_acceleration(

                    di

                )

            )



            return (

                self.condition_factor(

                    factor_row,

                    state

                )

            )



        # -----------------------------------------------------

        # Factor25:

        # Factor22 × recent overnight reaction

        # -----------------------------------------------------



        elif self.factor_type == 25:



            state = (

                self.calc_recent_reaction(

                    di,

                    "overnight"

                )

            )



            return (

                self.condition_factor(

                    factor_row,

                    state

                )

            )



        # -----------------------------------------------------

        # Factor26:

        # Factor22 × recent intraday reaction

        # -----------------------------------------------------



        elif self.factor_type == 26:



            state = (

                self.calc_recent_reaction(

                    di,

                    "intraday"

                )

            )



            return (

                self.condition_factor(

                    factor_row,

                    state

                )

            )



        # -----------------------------------------------------

        # Factor27: veto long side when 5-day return is weak

        # -----------------------------------------------------



        elif self.factor_type == 27:



            state = self.calc_industry_return(

                di,

                5

            )



            return self.long_side_veto(

                factor_row,

                state

            )



        # -----------------------------------------------------

        # Factor28: veto long side when acceleration is weak

        # -----------------------------------------------------



        elif self.factor_type == 28:



            state = self.calc_industry_acceleration(

                di

            )



            return self.long_side_veto(

                factor_row,

                state

            )



        # -----------------------------------------------------

        # Factor29: veto long side when intraday state is weak

        # -----------------------------------------------------



        elif self.factor_type == 29:



            state = self.calc_recent_reaction(

                di,

                "intraday"

            )



            return self.long_side_veto(

                factor_row,

                state

            )



        # -----------------------------------------------------

        # Factor30: veto long side when overnight state is weak

        # -----------------------------------------------------



        elif self.factor_type == 30:



            state = self.calc_recent_reaction(

                di,

                "overnight"

            )



            return self.long_side_veto(

                factor_row,

                state

            )



        # -----------------------------------------------------

        # Factor31: veto long side when >=2 price states are weak

        # -----------------------------------------------------



        elif self.factor_type == 31:



            return self.multi_state_long_veto(

                di,

                factor_row

            )



        else:



            raise ValueError(

                "invalid factorType"

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



        # PeopleDaily:

        # always t-1 natural calendar day

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

        # Factor23-31:

        # incorporate market reaction / asymmetric long-side veto

        # =====================================================



        factor_row = (

            self.apply_runtime_factor(

                di,

                factor_row

            )

        )



        # =====================================================

        # Industry mapping

        #

        # Use previous trading day's known classification

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

        # Industry -> stocks

        # =====================================================



        for j in range(

            len(self.factor_codes)

        ):



            code = int(

                self.factor_codes[j]

            )



            value = float(

                factor_row[j]

            )



            if not np.isfinite(

                value

            ):

                continue



            # -------------------------------------------------

            # Existing Factor22 weak-signal filter

            # -------------------------------------------------



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

                np.sum(mask)

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

                "[PeopleDailyReaction]",

                "trade=",

                trade_date,

                "signal=",

                signal_date,

                "factorType=",

                self.factor_type,

                "reactionWindow=",

                self.reaction_window,

                "priceExtreme=",

                self.price_extreme,

                "boost=",

                self.confirm_boost,

                "penalty=",

                self.conflict_penalty,

                "nonzero=",

                int(

                    np.count_nonzero(

                        self.alpha

                    )

                ),

            )