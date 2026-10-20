# -*- coding: utf-8 -*-

import os
import csv
import datetime
import numpy as np

from alphasim import AlphaBase
from alphasim import DataRegistry as dr


class PeopleDailyBucketStudyAlpha(AlphaBase):

    """
    Research-only diagnostic alpha.

    Purpose:
        Export daily SW1 industry states:

        - Factor22
        - 5d industry return
        - 5d acceleration
        - 5d overnight return
        - 5d intraday return
        - future 1d industry return
        - future 5d industry return

    IMPORTANT:
        Future returns are used ONLY for research diagnostics.
        They are NEVER used to generate alpha.

    Factor timing:
        trade date t
        uses PeopleDaily signal from previous natural day t-1.

    Forward return:
        fwd1 = close[t] / close[t-1] - 1
        fwd5 = close[t+4] / close[t-1] - 1

    Industry membership:
        fixed at t-1 when calculating future return.
    """

    def __init__(self, cfg):

        super().__init__(cfg)

        # =====================================================
        # Parameters
        # =====================================================

        self.start_date = int(
            cfg.getAttributeDefault(
                "studyStart",
                20160104
            )
        )

        self.end_date = int(
            cfg.getAttributeDefault(
                "studyEnd",
                20240628
            )
        )

        self.output_file = (
            "/home/dwyao/WorkPlace/"
            "PeopleDailyFactor/outputs/"
            "factor22_bucket_raw.csv"
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

        self.adj_close = dr.getData(
            "adj_close"
        )

        self.adj_open = dr.getData(
            "adj_open"
        )

        # =====================================================
        # PeopleDaily Factor22
        # =====================================================

        factor_path = (
            "/home/dwyao/WorkPlace/"
            "PeopleDailyFactor/data/factor/"
            "people_daily_ai_policy_factor.npz"
        )

        x = np.load(
            factor_path
        )

        self.factor_dates = (
            x["dates"]
            .astype(np.int64)
        )

        self.factor_codes = (
            x["codes"]
            .astype(np.int64)
        )

        self.factor_matrix = (
            x[
                "combo17_factor18_rank"
            ]
            .astype(np.float64)
        )

        # =====================================================
        # Dates
        # =====================================================

        self.sim_dates = np.memmap(
            "/data/simucache/"
            "__universe/Dates.npy",
            dtype=np.int64,
            mode="r"
        )

        # =====================================================
        # Prepare output
        # =====================================================

        output_dir = os.path.dirname(
            self.output_file
        )

        if output_dir:
            if not os.path.exists(
                output_dir
            ):
                os.makedirs(
                    output_dir
                )

        # remove old file once when object created
        if os.path.exists(
            self.output_file
        ):
            os.remove(
                self.output_file
            )

        self.header_written = False

        print("")
        print("=" * 80)
        print("PEOPLE DAILY FACTOR22 BUCKET STUDY")
        print("=" * 80)
        print("output:", self.output_file)
        print(
            "study:",
            self.start_date,
            "->",
            self.end_date
        )
        print("=" * 80)
        print("")


    # =========================================================
    # Calendar
    # =========================================================

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

        dt -= datetime.timedelta(
            days=1
        )

        return int(
            dt.strftime(
                "%Y%m%d"
            )
        )


    # =========================================================
    # Industry mapping
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

        level1[
            raw == 220100
        ] = 750000

        level1[
            raw == 410400
        ] = 760000

        level1[
            raw == 510100
        ] = -1

        return level1


    # =========================================================
    # Cross-sectional rank [-1,1]
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
                sorted_x[end]
                ==
                sorted_x[start]
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
            float(
                n - 1
            )
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

        # remove obviously broken observations
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
    # Factor22 row
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
    # Aggregate stock values by industry
    # =========================================================

    def aggregate_by_industry(
        self,
        stock_values,
        industry_di
    ):

        stock_values = np.asarray(
            stock_values,
            dtype=np.float64
        ).ravel()

        raw_industry = np.asarray(
            self.industry[
                industry_di
            ]
        ).ravel()

        sw1 = self.map_industry_array(
            raw_industry
        )

        result = np.full(
            len(
                self.factor_codes
            ),
            np.nan,
            dtype=np.float64
        )

        finite = np.isfinite(
            stock_values
        )

        for j, code in enumerate(
            self.factor_codes
        ):

            mask = (
                (sw1 == int(code))
                &
                finite
            )

            if not np.any(
                mask
            ):
                continue

            result[
                j
            ] = self.robust_mean(
                stock_values[
                    mask
                ]
            )

        return result


    # =========================================================
    # Historical industry close return
    #
    # uses only through t-1
    # =========================================================

    def industry_close_return(
        self,
        di,
        window
    ):

        end_di = (
            di - 1
        )

        start_di = (
            end_di
            -
            window
        )

        result = np.full(
            len(
                self.factor_codes
            ),
            np.nan,
            dtype=np.float64
        )

        if start_di < 0:
            return result

        close1 = np.asarray(
            self.adj_close[
                end_di
            ],
            dtype=np.float64
        ).ravel()

        close0 = np.asarray(
            self.adj_close[
                start_di
            ],
            dtype=np.float64
        ).ravel()

        stock_ret = np.full(
            len(
                close1
            ),
            np.nan,
            dtype=np.float64
        )

        valid = (
            np.isfinite(close1)
            &
            np.isfinite(close0)
            &
            (close1 > 0)
            &
            (close0 > 0)
        )

        stock_ret[
            valid
        ] = (
            close1[
                valid
            ]
            /
            close0[
                valid
            ]
            -
            1.0
        )

        return self.aggregate_by_industry(
            stock_ret,
            end_di
        )


    # =========================================================
    # Acceleration:
    #
    # ret5 - ret20/4
    # =========================================================

    def industry_acceleration(
        self,
        di
    ):

        ret5 = self.industry_close_return(
            di,
            5
        )

        ret20 = self.industry_close_return(
            di,
            20
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
            /
            4.0
        )

        return result


    # =========================================================
    # Daily overnight:
    #
    # open[d] / close[d-1] - 1
    # =========================================================

    def daily_industry_overnight(
        self,
        d
    ):

        if d <= 0:

            return np.full(
                len(
                    self.factor_codes
                ),
                np.nan,
                dtype=np.float64
            )

        op = np.asarray(
            self.adj_open[
                d
            ],
            dtype=np.float64
        ).ravel()

        prev_close = np.asarray(
            self.adj_close[
                d - 1
            ],
            dtype=np.float64
        ).ravel()

        stock_ret = np.full(
            len(op),
            np.nan,
            dtype=np.float64
        )

        valid = (
            np.isfinite(op)
            &
            np.isfinite(prev_close)
            &
            (op > 0)
            &
            (prev_close > 0)
        )

        stock_ret[
            valid
        ] = (
            op[
                valid
            ]
            /
            prev_close[
                valid
            ]
            -
            1.0
        )

        return self.aggregate_by_industry(
            stock_ret,
            d - 1
        )


    # =========================================================
    # Daily intraday:
    #
    # close[d] / open[d] - 1
    # =========================================================

    def daily_industry_intraday(
        self,
        d
    ):

        close = np.asarray(
            self.adj_close[
                d
            ],
            dtype=np.float64
        ).ravel()

        op = np.asarray(
            self.adj_open[
                d
            ],
            dtype=np.float64
        ).ravel()

        stock_ret = np.full(
            len(close),
            np.nan,
            dtype=np.float64
        )

        valid = (
            np.isfinite(close)
            &
            np.isfinite(op)
            &
            (close > 0)
            &
            (op > 0)
        )

        stock_ret[
            valid
        ] = (
            close[
                valid
            ]
            /
            op[
                valid
            ]
            -
            1.0
        )

        return self.aggregate_by_industry(
            stock_ret,
            max(
                d - 1,
                0
            )
        )


    # =========================================================
    # Recent mean reaction
    # =========================================================

    def recent_reaction(
        self,
        di,
        kind,
        window=5
    ):

        rows = []

        for d in range(
            di - window,
            di
        ):

            if d <= 0:
                continue

            if kind == "overnight":

                row = (
                    self.daily_industry_overnight(
                        d
                    )
                )

            elif kind == "intraday":

                row = (
                    self.daily_industry_intraday(
                        d
                    )
                )

            else:

                raise ValueError(
                    "bad reaction kind"
                )

            rows.append(
                row
            )

        if not rows:

            return np.full(
                len(
                    self.factor_codes
                ),
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

            x = matrix[
                :,
                j
            ]

            x = x[
                np.isfinite(x)
            ]

            if len(x):

                result[
                    j
                ] = np.mean(
                    x
                )

        return result


    # =========================================================
    # Future industry returns
    #
    # Important:
    # These are diagnostic labels ONLY.
    #
    # t trade date:
    #
    # fwd1:
    # close[t] / close[t-1] - 1
    #
    # fwd5:
    # close[t+4] / close[t-1] - 1
    #
    # membership fixed at t-1.
    # =========================================================

    def future_industry_return(
        self,
        di,
        horizon
    ):

        start_di = (
            di - 1
        )

        if horizon == 1:
            end_di = di

        else:
            end_di = (
                di
                +
                horizon
                -
                1
            )

        result = np.full(
            len(
                self.factor_codes
            ),
            np.nan,
            dtype=np.float64
        )

        if (
            start_di < 0
            or
            end_di
            >= len(
                self.sim_dates
            )
        ):
            return result

        close0 = np.asarray(
            self.adj_close[
                start_di
            ],
            dtype=np.float64
        ).ravel()

        close1 = np.asarray(
            self.adj_close[
                end_di
            ],
            dtype=np.float64
        ).ravel()

        stock_ret = np.full(
            len(close0),
            np.nan,
            dtype=np.float64
        )

        valid = (
            np.isfinite(close0)
            &
            np.isfinite(close1)
            &
            (close0 > 0)
            &
            (close1 > 0)
        )

        stock_ret[
            valid
        ] = (
            close1[
                valid
            ]
            /
            close0[
                valid
            ]
            -
            1.0
        )

        # fixed membership at signal time
        return self.aggregate_by_industry(
            stock_ret,
            start_di
        )


    # =========================================================
    # Write rows
    # =========================================================

    def write_rows(
        self,
        rows
    ):

        fieldnames = [
            "trade_date",
            "signal_date",
            "industry",

            "factor22",

            "ret5",
            "ret5_rank",

            "accel5",
            "accel5_rank",

            "overnight5",
            "overnight5_rank",

            "intraday5",
            "intraday5_rank",

            "fwd1",
            "fwd5",
        ]

        mode = (
            "a"
            if self.header_written
            else
            "w"
        )

        with open(
            self.output_file,
            mode,
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            if not self.header_written:

                writer.writeheader()

                self.header_written = True

            writer.writerows(
                rows
            )


    # =========================================================
    # Generate
    # =========================================================

    def generate(
        self,
        di
    ):

        # This is research alpha:
        # do not take positions.
        self.alpha[:] = 0.0

        trade_date = int(
            self.sim_dates[
                di
            ]
        )

        if (
            trade_date
            <
            self.start_date
            or
            trade_date
            >
            self.end_date
        ):
            return

        signal_date = (
            self.previous_calendar_date(
                trade_date
            )
        )

        factor22 = (
            self.get_factor_row(
                signal_date
            )
        )

        if factor22 is None:
            return

        # =====================================================
        # Historical price states
        # =====================================================

        ret5 = (
            self.industry_close_return(
                di,
                5
            )
        )

        accel5 = (
            self.industry_acceleration(
                di
            )
        )

        overnight5 = (
            self.recent_reaction(
                di,
                "overnight",
                5
            )
        )

        intraday5 = (
            self.recent_reaction(
                di,
                "intraday",
                5
            )
        )

        # =====================================================
        # Rank price states
        # =====================================================

        ret5_rank = self.cs_rank(
            ret5
        )

        accel5_rank = self.cs_rank(
            accel5
        )

        overnight5_rank = self.cs_rank(
            overnight5
        )

        intraday5_rank = self.cs_rank(
            intraday5
        )

        # =====================================================
        # Future labels
        # =====================================================

        fwd1 = (
            self.future_industry_return(
                di,
                1
            )
        )

        fwd5 = (
            self.future_industry_return(
                di,
                5
            )
        )

        # =====================================================
        # Output
        # =====================================================

        rows = []

        for j, code in enumerate(
            self.factor_codes
        ):

            rows.append(
                {
                    "trade_date":
                        trade_date,

                    "signal_date":
                        signal_date,

                    "industry":
                        int(code),

                    "factor22":
                        factor22[j],

                    "ret5":
                        ret5[j],

                    "ret5_rank":
                        ret5_rank[j],

                    "accel5":
                        accel5[j],

                    "accel5_rank":
                        accel5_rank[j],

                    "overnight5":
                        overnight5[j],

                    "overnight5_rank":
                        overnight5_rank[j],

                    "intraday5":
                        intraday5[j],

                    "intraday5_rank":
                        intraday5_rank[j],

                    "fwd1":
                        fwd1[j],

                    "fwd5":
                        fwd5[j],
                }
            )

        self.write_rows(
            rows
        )

        if (
            self.debug
            and
            di % 100 == 0
        ):

            print(
                "[BucketStudy]",
                trade_date,
                "rows=",
                len(rows)
            )