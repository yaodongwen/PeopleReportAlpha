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
    # Policy-type text factors stored in a separate NPZ.
    #
    # Factor32-37 all use the same acceleration long-side veto
    # that made Factor28 work. Only the TEXT SOURCE changes.
    # =========================================================

    POLICY_TYPE_FACTOR_MAP = {
        32: "supportive_surprise_20_rank",
        33: "restrictive_surprise_20_rank",
        34: "investment_subsidy_surprise_20_rank",
        35: "demand_stimulus_surprise_20_rank",
        36: "supportive_residual_rank",
        37: "restrictive_residual_rank",
    }


    # =========================================================
    # Factor28 + Factor36 combinations.
    #
    # Combination is performed at the SW1 industry-signal level
    # AFTER applying the same acceleration long-side veto to
    # both source signals, and BEFORE stock mapping.
    #
    # Then the combined industry signal is re-ranked
    # cross-sectionally to restore a common [-1, 1] scale.
    # =========================================================

    COMBO_FACTOR_WEIGHTS = {
        38: (0.85, 0.15),
        39: (0.75, 0.25),
        40: (0.65, 0.35),
    }


    # =========================================================
    # Factor45-50: single-factor synthesis methods for F28+F36.
    #
    # Goal:
    #   keep ONE final industry factor, while reducing the
    #   destructive interaction seen in simple linear blending.
    #
    # 45-47: orthogonal incremental synthesis
    #   F36_perp = residual(F36 | 1 + F28), cross-sectionally
    #   combo = F28 + lambda * F36_perp
    #   combo -> CS rank -> ONE acceleration long veto
    #
    # 48: protect strong F28 tails
    #   |F28| >= 0.6 : keep F28 exactly
    #   otherwise     : F28 + 0.25*F36
    #   no rerank before veto, to preserve strong F28 values
    #
    # 49: same-direction confirmation
    #   if F28 and F36 have same sign:
    #       strengthen |F28| by up to 25% according to |F36|
    #   otherwise:
    #       keep F28 unchanged
    #   clip to [-1,1], then ONE acceleration long veto
    #
    # 50: post-veto dual-rank fusion
    #   apply acceleration veto to F28 and F36 separately
    #   rank active (non-zero) values separately, preserving veto 0
    #   final = 0.75*rank(F28) + 0.25*rank(F36)
    #   no final rerank
    # =========================================================

    SYNTHESIS_FACTOR_TYPES = {
        45, 46, 47, 48, 49, 50
    }

    ORTHO_LAMBDAS = {
        45: 0.15,
        46: 0.25,
        47: 0.35,
    }

    FACTOR48_WEAK_WEIGHT = 0.25
    FACTOR48_STRONG_CUT = 0.60

    FACTOR49_CONFIRM_STRENGTH = 0.25

    FACTOR50_WEIGHT28 = 0.75
    FACTOR50_WEIGHT36 = 0.25


    # =========================================================
    # Factor51-54: NEW orthogonal time-structure information.
    #
    # 51: policy persistence surprise rank + acceleration veto
    # 52: policy burst surprise rank + acceleration veto
    #
    # 53: one-factor rank ensemble
    #     0.75 * RankPreserveZero(F50)
    #   + 0.25 * RankPreserveZero(F51 after veto)
    #
    # 54: one-factor rank ensemble
    #     0.75 * RankPreserveZero(F50)
    #   + 0.25 * RankPreserveZero(F52 after veto)
    #
    # No final rerank for 53/54, matching the useful structure
    # discovered in Factor50.
    # =========================================================

    TIME_STRUCTURE_FACTOR_MAP = {
        51: "policy_persistence_surprise_20_rank",
        52: "policy_burst_surprise_5_20_rank",
    }

    TIME_ENSEMBLE_FACTOR_MAP = {
        53: "policy_persistence_surprise_20_rank",
        54: "policy_burst_surprise_5_20_rank",
    }

    TIME_ENSEMBLE_WEIGHT50 = 0.75
    TIME_ENSEMBLE_WEIGHT_NEW = 0.25


    # =========================================================
    # Factor55: orthogonalized burst increment.
    #
    # burst_perp = residual(F52 | 1 + F50)
    # final      = 0.75*RankPreserveZero(F50)
    #            + 0.25*RankPreserveZero(burst_perp_after_veto)
    #
    # No final rerank.
    # =========================================================

    ORTHO_BURST_FACTOR_TYPES = {55}
    FACTOR55_WEIGHT50 = 0.75
    FACTOR55_WEIGHT_BURST = 0.25


    # =========================================================
    # Factor56-60: concentration + first-appearance novelty
    #
    # 56 = concentration standalone + acceleration veto
    # 57 = first-appearance novelty standalone + acceleration veto
    #
    # 58 = 0.75*Rank(F54) + 0.25*Rank(F56)
    # 59 = 0.75*Rank(F54) + 0.25*Rank(F57)
    #
    # 60 = three-view ensemble:
    #      0.70*Rank(F54)
    #    + 0.15*Rank(F56)
    #    + 0.15*Rank(F57)
    #
    # No final rerank for 58-60.
    # =========================================================

    CN_FACTOR_MAP = {
        56: "policy_concentration_surprise_20_rank",
        57: "policy_first_appearance_surprise_20_rank",
    }

    CN_ENSEMBLE_FACTOR_MAP = {
        58: "policy_concentration_surprise_20_rank",
        59: "policy_first_appearance_surprise_20_rank",
    }

    CN_THREE_VIEW_FACTOR_TYPES = {60}

    CN_WEIGHT54 = 0.75
    CN_WEIGHT_NEW = 0.25

    CN3_WEIGHT54 = 0.70
    CN3_WEIGHT_CONC = 0.15
    CN3_WEIGHT_NOVELTY = 0.15


    REVERSE_CONCENTRATION_FACTOR_TYPES = {61}
    REVERSE_CONCENTRATION_ENSEMBLE_TYPES = {62}

    SEMANTIC_FACTOR_MAP = {
        63: "semantic_first_appearance_surprise_20_rank",
    }

    SEMANTIC_ENSEMBLE_FACTOR_MAP = {
        64: "semantic_first_appearance_surprise_20_rank",
    }

    NEXT_WEIGHT54 = 0.75
    NEXT_WEIGHT_NEW = 0.25


    # =========================================================
    # Factor65-70: continuous semantic novelty + alternative
    # combination architectures.
    #
    # 65 = continuous semantic novelty standalone
    # 66 = reversed continuous semantic novelty standalone
    #
    # 67 = F54 with F61 as LONG-SIDE disagreement veto
    # 68 = F54 with F66 as LONG-SIDE disagreement veto
    #
    # 69 = ONE Alpha module, but combine F54 and F61 as two
    #      separately stock-mapped / separately L1-normalized
    #      sleeves, then 75/25 sum.
    #
    # 70 = same stock-sleeve architecture for F54 + F66.
    #
    # This tests whether previous failures came from industry-
    # level rank mixing rather than lack of incremental info.
    # =========================================================

    SEMANTIC_CONT_FACTOR_MAP = {
        65: "semantic_novelty_surprise_20_rank",
        66: "semantic_novelty_surprise_20_rank",
    }

    GATE_FACTOR_TYPES = {67, 68}
    STOCK_SLEEVE_FACTOR_TYPES = {69, 70}

    GATE_ADVERSE_CUT = 0.60
    SLEEVE_WEIGHT54 = 0.75
    SLEEVE_WEIGHT_AUX = 0.25


    # =========================================================
    # Factor71 robustness test:
    #
    # SAME stock-sleeve architecture as Factor69, but 50/50.
    #
    # Purpose:
    #   test whether Factor69's improvement is robust to a large
    #   change in sleeve weights, rather than optimizing weights.
    #
    # 71 = 0.50 * F54 stock sleeve
    #    + 0.50 * reversed-concentration stock sleeve
    # =========================================================

    STOCK_SLEEVE_ROBUST_FACTOR_TYPES = {71}

    SLEEVE71_WEIGHT54 = 0.50
    SLEEVE71_WEIGHT_AUX = 0.50


    # =========================================================
    # Factor72: policy direction consistency standalone
    #
    # consistency_20
    #   = abs(sum20(signed_attention))
    #     / sum20(abs(signed_attention))
    #
    # NPZ already contains:
    #   shifted365 historical surprise -> daily CS rank
    #
    # Runtime:
    #   same acceleration long-side veto as Factor28/F54 family
    # =========================================================

    DIRECTION_CONSISTENCY_FACTOR_MAP = {
        72: "policy_direction_consistency_surprise_20_rank",
    }


    # =========================================================
    # Factor73: signed direction consistency standalone
    #
    # signed_consistency_20
    #   = sum20(signed_attention)
    #     / sum20(abs(signed_attention))
    #
    # Positive and negative persistent policy regimes remain
    # distinguishable.
    # =========================================================

    SIGNED_DIRECTION_CONSISTENCY_FACTOR_MAP = {
        73: "policy_signed_direction_consistency_surprise_20_rank",
    }


    # =========================================================
    # Factors74-75:
    # combine frozen F54 with low-correlation Factor73 using the
    # stock-sleeve architecture (one Alpha module / one vector).
    #
    # 74 = 75% F54 + 25% F73
    # 75 = 50% F54 + 50% F73
    #
    # These are coarse constraint tests, not a weight sweep.
    # Goal: see whether F73 can dilute F54's pool correlation
    # below the hard |corr| < 0.20 requirement while retaining
    # useful Sharpe.
    # =========================================================

    F54_F73_STOCK_SLEEVE_TYPES = {74, 75}

    F74_WEIGHT54 = 0.75
    F74_WEIGHT73 = 0.25

    F75_WEIGHT54 = 0.50
    F75_WEIGHT73 = 0.50


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

            and

            self.factor_type not in self.POLICY_TYPE_FACTOR_MAP

            and

            self.factor_type not in self.COMBO_FACTOR_WEIGHTS

            and

            self.factor_type not in self.SYNTHESIS_FACTOR_TYPES

            and

            self.factor_type not in self.TIME_STRUCTURE_FACTOR_MAP

            and

            self.factor_type not in self.TIME_ENSEMBLE_FACTOR_MAP

            and

            self.factor_type not in self.ORTHO_BURST_FACTOR_TYPES

            and

            self.factor_type not in self.CN_FACTOR_MAP

            and

            self.factor_type not in self.CN_ENSEMBLE_FACTOR_MAP

            and

            self.factor_type not in self.CN_THREE_VIEW_FACTOR_TYPES

            and

            self.factor_type not in self.REVERSE_CONCENTRATION_FACTOR_TYPES

            and

            self.factor_type not in self.REVERSE_CONCENTRATION_ENSEMBLE_TYPES

            and

            self.factor_type not in self.SEMANTIC_FACTOR_MAP

            and

            self.factor_type not in self.SEMANTIC_ENSEMBLE_FACTOR_MAP

            and

            self.factor_type not in self.SEMANTIC_CONT_FACTOR_MAP

            and

            self.factor_type not in self.GATE_FACTOR_TYPES

            and

            self.factor_type not in self.STOCK_SLEEVE_FACTOR_TYPES

            and

            self.factor_type not in self.STOCK_SLEEVE_ROBUST_FACTOR_TYPES

            and

            self.factor_type not in self.DIRECTION_CONSISTENCY_FACTOR_MAP

            and

            self.factor_type not in self.SIGNED_DIRECTION_CONSISTENCY_FACTOR_MAP

            and

            self.factor_type not in self.F54_F73_STOCK_SLEEVE_TYPES

        ):

            raise ValueError(

                "factorType must be 1-75"

            )



        # -----------------------------------------------------

        # factor23-31 all start from factor22

        # -----------------------------------------------------



        if self.factor_type in self.F54_F73_STOCK_SLEEVE_TYPES:

            self.factor_name = "factor54_factor73_stock_sleeve"
            self.price_mode = 33

        elif self.factor_type in self.SIGNED_DIRECTION_CONSISTENCY_FACTOR_MAP:

            self.factor_name = self.SIGNED_DIRECTION_CONSISTENCY_FACTOR_MAP[
                self.factor_type
            ]
            self.price_mode = 32

        elif self.factor_type in self.DIRECTION_CONSISTENCY_FACTOR_MAP:

            self.factor_name = self.DIRECTION_CONSISTENCY_FACTOR_MAP[
                self.factor_type
            ]
            self.price_mode = 31

        elif self.factor_type in self.STOCK_SLEEVE_ROBUST_FACTOR_TYPES:

            self.factor_name = "factor54_reversed_concentration_stock_sleeve_50_50"
            self.price_mode = 30

        elif self.factor_type in self.STOCK_SLEEVE_FACTOR_TYPES:

            self.factor_name = "factor54_stock_sleeve_combo"
            self.price_mode = 29

        elif self.factor_type in self.GATE_FACTOR_TYPES:

            self.factor_name = "factor54_aux_long_gate"
            self.price_mode = 28

        elif self.factor_type in self.SEMANTIC_CONT_FACTOR_MAP:

            self.factor_name = self.SEMANTIC_CONT_FACTOR_MAP[
                self.factor_type
            ]
            self.semantic_name = self.factor_name
            self.price_mode = 27

        elif self.factor_type in self.SEMANTIC_ENSEMBLE_FACTOR_MAP:

            self.factor_name = "factor54_semantic_novelty_ensemble"
            self.semantic_name = self.SEMANTIC_ENSEMBLE_FACTOR_MAP[
                self.factor_type
            ]
            self.price_mode = 25

        elif self.factor_type in self.SEMANTIC_FACTOR_MAP:

            self.factor_name = self.SEMANTIC_FACTOR_MAP[
                self.factor_type
            ]
            self.semantic_name = self.factor_name
            self.price_mode = 24

        elif self.factor_type in self.REVERSE_CONCENTRATION_ENSEMBLE_TYPES:

            self.factor_name = "factor54_reversed_concentration_ensemble"
            self.price_mode = 23

        elif self.factor_type in self.REVERSE_CONCENTRATION_FACTOR_TYPES:

            self.factor_name = "policy_concentration_surprise_20_rank"
            self.price_mode = 22

        elif self.factor_type in self.CN_THREE_VIEW_FACTOR_TYPES:

            self.factor_name = "factor54_concentration_novelty_ensemble"
            self.cn_name = "three_view"
            self.price_mode = 21

        elif self.factor_type in self.CN_ENSEMBLE_FACTOR_MAP:

            self.factor_name = "factor54_cn_rank_ensemble"
            self.cn_name = self.CN_ENSEMBLE_FACTOR_MAP[
                self.factor_type
            ]
            self.price_mode = 20

        elif self.factor_type in self.CN_FACTOR_MAP:

            self.factor_name = self.CN_FACTOR_MAP[
                self.factor_type
            ]
            self.cn_name = self.factor_name
            self.price_mode = 19

        elif self.factor_type in self.ORTHO_BURST_FACTOR_TYPES:

            self.factor_name = "factor50_orthogonal_burst_ensemble"
            self.time_structure_name = "policy_burst_surprise_5_20_rank"
            self.price_mode = 18

        elif self.factor_type in self.TIME_ENSEMBLE_FACTOR_MAP:

            self.factor_name = "factor50_time_structure_ensemble"
            self.time_structure_name = (
                self.TIME_ENSEMBLE_FACTOR_MAP[
                    self.factor_type
                ]
            )
            self.price_mode = 17

        elif self.factor_type in self.TIME_STRUCTURE_FACTOR_MAP:

            self.factor_name = (
                self.TIME_STRUCTURE_FACTOR_MAP[
                    self.factor_type
                ]
            )
            self.time_structure_name = self.factor_name
            self.price_mode = 16

        elif self.factor_type in self.SYNTHESIS_FACTOR_TYPES:

            self.factor_name = "factor28_factor36_synthesis"
            self.price_mode = 15

            if self.factor_type in self.ORTHO_LAMBDAS:
                self.synthesis_lambda = (
                    self.ORTHO_LAMBDAS[
                        self.factor_type
                    ]
                )
            else:
                self.synthesis_lambda = None

        elif self.factor_type in self.COMBO_FACTOR_WEIGHTS:

            self.factor_name = "factor28_factor36_combo"

            self.combo_weight28, self.combo_weight36 = (
                self.COMBO_FACTOR_WEIGHTS[
                    self.factor_type
                ]
            )

            # Same price rule as Factor28 / Factor36:
            # acceleration long-side veto on each source first.
            self.price_mode = 14

        elif self.factor_type in self.POLICY_TYPE_FACTOR_MAP:

            self.factor_name = (
                self.POLICY_TYPE_FACTOR_MAP[
                    self.factor_type
                ]
            )

            # Factor32-37 use acceleration veto after loading
            # the policy-type text factor.
            self.price_mode = 13

        elif self.factor_type in self.RUNTIME_FACTOR_MODE:



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



        if not hasattr(self, "combo_weight28"):
            self.combo_weight28 = None
            self.combo_weight36 = None


        if not hasattr(self, "synthesis_lambda"):
            self.synthesis_lambda = None


        if not hasattr(self, "time_structure_name"):
            self.time_structure_name = None


        if not hasattr(self, "cn_name"):
            self.cn_name = None

        if not hasattr(self, "semantic_name"):
            self.semantic_name = None


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

        base_factor_path = (
            "/home/dwyao/WorkPlace/"
            "PeopleDailyFactor/data/factor/"
            "people_daily_ai_policy_factor.npz"
        )

        policy_type_factor_path = (
            "/home/dwyao/WorkPlace/"
            "PeopleDailyFactor/data/factor/"
            "people_daily_policy_type_factor.npz"
        )


        time_structure_factor_path = (
            "/home/dwyao/WorkPlace/"
            "PeopleDailyFactor/data/factor/"
            "people_daily_time_structure_factor.npz"
        )


        cn_factor_path = (
            "/home/dwyao/WorkPlace/"
            "PeopleDailyFactor/data/factor/"
            "people_daily_concentration_novelty_factor.npz"
        )


        semantic_factor_path = (
            "/home/dwyao/WorkPlace/"
            "PeopleDailyFactor/data/factor/"
            "people_daily_semantic_novelty_factor.npz"
        )


        direction_consistency_factor_path = (
            "/home/dwyao/WorkPlace/"
            "PeopleDailyFactor/data/factor/"
            "people_daily_direction_consistency_factor.npz"
        )


        signed_direction_consistency_factor_path = (
            "/home/dwyao/WorkPlace/"
            "PeopleDailyFactor/data/factor/"
            "people_daily_signed_direction_consistency_factor.npz"
        )

        # -----------------------------------------------------
        # Factor38-40 need BOTH Factor22(raw source of Factor28)
        # and Factor36(raw supportive residual source).
        # -----------------------------------------------------

        if (
            self.factor_type in self.GATE_FACTOR_TYPES
            or
            self.factor_type in self.STOCK_SLEEVE_FACTOR_TYPES
            or
            self.factor_type in self.STOCK_SLEEVE_ROBUST_FACTOR_TYPES
            or
            self.factor_type in self.F54_F73_STOCK_SLEEVE_TYPES
        ):

            base_data = np.load(base_factor_path)
            policy_data = np.load(policy_type_factor_path)
            time_data = np.load(time_structure_factor_path)
            cn_data = np.load(cn_factor_path)
            semantic_data = np.load(semantic_factor_path)

            base_dates = base_data["dates"].astype(np.int64)
            base_codes = base_data["codes"].astype(np.int64)

            for label, d in [
                ("policy", policy_data),
                ("time", time_data),
                ("cn", cn_data),
                ("semantic", semantic_data),
            ]:
                if not np.array_equal(
                    base_dates,
                    d["dates"].astype(np.int64)
                ):
                    raise ValueError(
                        "base/%s dates mismatch" % label
                    )

                if not np.array_equal(
                    base_codes,
                    d["codes"].astype(np.int64)
                ):
                    raise ValueError(
                        "base/%s codes mismatch" % label
                    )

            self.factor_dates = base_dates
            self.factor_codes = base_codes

            self.factor28_matrix = (
                base_data["combo17_factor18_rank"]
                .astype(np.float64)
            )

            self.factor36_matrix = (
                policy_data["supportive_residual_rank"]
                .astype(np.float64)
            )

            self.burst_matrix = (
                time_data["policy_burst_surprise_5_20_rank"]
                .astype(np.float64)
            )

            self.concentration_matrix = (
                cn_data["policy_concentration_surprise_20_rank"]
                .astype(np.float64)
            )

            self.semantic_cont_matrix = (
                semantic_data["semantic_novelty_surprise_20_rank"]
                .astype(np.float64)
            )

            if self.factor_type in self.F54_F73_STOCK_SLEEVE_TYPES:
                signed_consistency_data = np.load(
                    signed_direction_consistency_factor_path
                )

                if not np.array_equal(
                    base_dates,
                    signed_consistency_data["dates"].astype(np.int64)
                ):
                    raise ValueError(
                        "base/signed consistency dates mismatch"
                    )

                if not np.array_equal(
                    base_codes,
                    signed_consistency_data["codes"].astype(np.int64)
                ):
                    raise ValueError(
                        "base/signed consistency codes mismatch"
                    )

                self.factor73_matrix = (
                    signed_consistency_data[
                        "policy_signed_direction_consistency_surprise_20_rank"
                    ].astype(np.float64)
                )

            self.factor_matrix = self.factor28_matrix
            self.factor_path = base_factor_path

        elif (
            self.factor_type in self.REVERSE_CONCENTRATION_ENSEMBLE_TYPES
            or
            self.factor_type in self.SEMANTIC_ENSEMBLE_FACTOR_MAP
        ):

            base_data = np.load(base_factor_path)
            policy_data = np.load(policy_type_factor_path)
            time_data = np.load(time_structure_factor_path)
            cn_data = np.load(cn_factor_path)

            semantic_data = None
            if self.factor_type in self.SEMANTIC_ENSEMBLE_FACTOR_MAP:
                semantic_data = np.load(semantic_factor_path)

            base_dates = base_data["dates"].astype(np.int64)
            base_codes = base_data["codes"].astype(np.int64)

            datasets = [
                ("policy", policy_data),
                ("time", time_data),
                ("cn", cn_data),
            ]
            if semantic_data is not None:
                datasets.append(("semantic", semantic_data))

            for label, d in datasets:
                if not np.array_equal(base_dates, d["dates"].astype(np.int64)):
                    raise ValueError("base/%s dates mismatch" % label)
                if not np.array_equal(base_codes, d["codes"].astype(np.int64)):
                    raise ValueError("base/%s codes mismatch" % label)

            self.factor_dates = base_dates
            self.factor_codes = base_codes

            self.factor28_matrix = base_data[
                "combo17_factor18_rank"
            ].astype(np.float64)

            self.factor36_matrix = policy_data[
                "supportive_residual_rank"
            ].astype(np.float64)

            self.burst_matrix = time_data[
                "policy_burst_surprise_5_20_rank"
            ].astype(np.float64)

            self.concentration_matrix = cn_data[
                "policy_concentration_surprise_20_rank"
            ].astype(np.float64)

            if semantic_data is not None:
                self.semantic_matrix = semantic_data[
                    self.semantic_name
                ].astype(np.float64)

            self.factor_matrix = self.factor28_matrix
            self.factor_path = base_factor_path

        elif (
            self.factor_type in self.CN_ENSEMBLE_FACTOR_MAP
            or
            self.factor_type in self.CN_THREE_VIEW_FACTOR_TYPES
        ):

            base_data = np.load(base_factor_path)
            policy_data = np.load(policy_type_factor_path)
            time_data = np.load(time_structure_factor_path)
            cn_data = np.load(cn_factor_path)

            base_dates = base_data["dates"].astype(np.int64)
            base_codes = base_data["codes"].astype(np.int64)

            for label, d in [
                ("policy", policy_data),
                ("time", time_data),
                ("cn", cn_data),
            ]:
                if not np.array_equal(
                    base_dates,
                    d["dates"].astype(np.int64)
                ):
                    raise ValueError(
                        "base/%s NPZ dates do not match" % label
                    )

                if not np.array_equal(
                    base_codes,
                    d["codes"].astype(np.int64)
                ):
                    raise ValueError(
                        "base/%s NPZ codes do not match" % label
                    )

            self.factor_dates = base_dates
            self.factor_codes = base_codes

            self.factor28_matrix = (
                base_data["combo17_factor18_rank"]
                .astype(np.float64)
            )

            self.factor36_matrix = (
                policy_data["supportive_residual_rank"]
                .astype(np.float64)
            )

            self.burst_matrix = (
                time_data["policy_burst_surprise_5_20_rank"]
                .astype(np.float64)
            )

            self.concentration_matrix = (
                cn_data["policy_concentration_surprise_20_rank"]
                .astype(np.float64)
            )

            self.novelty_matrix = (
                cn_data["policy_first_appearance_surprise_20_rank"]
                .astype(np.float64)
            )

            if self.factor_type in self.CN_ENSEMBLE_FACTOR_MAP:
                self.cn_matrix = (
                    cn_data[self.cn_name]
                    .astype(np.float64)
                )

            self.factor_matrix = self.factor28_matrix

            self.factor_path = (
                base_factor_path
                + " + "
                + policy_type_factor_path
                + " + "
                + time_structure_factor_path
                + " + "
                + cn_factor_path
            )

        elif (
            self.factor_type in self.TIME_ENSEMBLE_FACTOR_MAP
            or
            self.factor_type in self.ORTHO_BURST_FACTOR_TYPES
        ):

            base_data = np.load(
                base_factor_path
            )

            policy_data = np.load(
                policy_type_factor_path
            )

            time_data = np.load(
                time_structure_factor_path
            )

            base_dates = base_data["dates"].astype(np.int64)
            base_codes = base_data["codes"].astype(np.int64)
            policy_dates = policy_data["dates"].astype(np.int64)
            policy_codes = policy_data["codes"].astype(np.int64)
            time_dates = time_data["dates"].astype(np.int64)
            time_codes = time_data["codes"].astype(np.int64)

            if not np.array_equal(base_dates, policy_dates):
                raise ValueError(
                    "base/policy NPZ dates do not match"
                )

            if not np.array_equal(base_codes, policy_codes):
                raise ValueError(
                    "base/policy NPZ codes do not match"
                )

            if not np.array_equal(base_dates, time_dates):
                raise ValueError(
                    "base/time NPZ dates do not match"
                )

            if not np.array_equal(base_codes, time_codes):
                raise ValueError(
                    "base/time NPZ codes do not match"
                )

            if "combo17_factor18_rank" not in base_data.files:
                raise KeyError(
                    "combo17_factor18_rank not found in base NPZ"
                )

            if "supportive_residual_rank" not in policy_data.files:
                raise KeyError(
                    "supportive_residual_rank not found in policy NPZ"
                )

            if self.time_structure_name not in time_data.files:
                raise KeyError(
                    "%s not found in time-structure NPZ"
                    % self.time_structure_name
                )

            self.factor_dates = base_dates
            self.factor_codes = base_codes

            self.factor28_matrix = (
                base_data["combo17_factor18_rank"]
                .astype(np.float64)
            )

            self.factor36_matrix = (
                policy_data["supportive_residual_rank"]
                .astype(np.float64)
            )

            self.time_structure_matrix = (
                time_data[self.time_structure_name]
                .astype(np.float64)
            )

            self.factor_matrix = self.factor28_matrix

            self.factor_path = (
                base_factor_path
                + " + "
                + policy_type_factor_path
                + " + "
                + time_structure_factor_path
            )

        elif (
            self.factor_type in self.COMBO_FACTOR_WEIGHTS
            or
            self.factor_type in self.SYNTHESIS_FACTOR_TYPES
        ):

            base_data = np.load(
                base_factor_path
            )

            policy_data = np.load(
                policy_type_factor_path
            )

            base_dates = (
                base_data["dates"]
                .astype(np.int64)
            )

            base_codes = (
                base_data["codes"]
                .astype(np.int64)
            )

            policy_dates = (
                policy_data["dates"]
                .astype(np.int64)
            )

            policy_codes = (
                policy_data["codes"]
                .astype(np.int64)
            )

            if not np.array_equal(
                base_dates,
                policy_dates
            ):
                raise ValueError(
                    "base/policy NPZ dates do not match"
                )

            if not np.array_equal(
                base_codes,
                policy_codes
            ):
                raise ValueError(
                    "base/policy NPZ codes do not match"
                )

            if (
                "combo17_factor18_rank"
                not in base_data.files
            ):
                raise KeyError(
                    "combo17_factor18_rank "
                    "not found in base NPZ"
                )

            if (
                "supportive_residual_rank"
                not in policy_data.files
            ):
                raise KeyError(
                    "supportive_residual_rank "
                    "not found in policy-type NPZ"
                )

            self.factor_dates = base_dates
            self.factor_codes = base_codes

            # Raw text signal underlying Factor28.
            self.factor28_matrix = (
                base_data[
                    "combo17_factor18_rank"
                ]
                .astype(np.float64)
            )

            # Raw text signal underlying Factor36.
            self.factor36_matrix = (
                policy_data[
                    "supportive_residual_rank"
                ]
                .astype(np.float64)
            )

            # Kept for compatibility with helper methods.
            self.factor_matrix = (
                self.factor28_matrix
            )

            self.factor_path = (
                base_factor_path
                + " + "
                + policy_type_factor_path
            )

        else:

            if (
                self.factor_type
                in self.POLICY_TYPE_FACTOR_MAP
            ):

                self.factor_path = (
                    policy_type_factor_path
                )

            elif (
                self.factor_type
                in self.TIME_STRUCTURE_FACTOR_MAP
            ):

                self.factor_path = (
                    time_structure_factor_path
                )

            elif (
                self.factor_type
                in self.CN_FACTOR_MAP
            ):

                self.factor_path = (
                    cn_factor_path
                )

            elif (
                self.factor_type
                in self.REVERSE_CONCENTRATION_FACTOR_TYPES
            ):

                self.factor_path = (
                    cn_factor_path
                )

            elif (
                self.factor_type
                in self.SEMANTIC_FACTOR_MAP
            ):

                self.factor_path = (
                    semantic_factor_path
                )

            elif (
                self.factor_type
                in self.SEMANTIC_CONT_FACTOR_MAP
            ):

                self.factor_path = (
                    semantic_factor_path
                )

            elif (
                self.factor_type
                in self.DIRECTION_CONSISTENCY_FACTOR_MAP
            ):

                self.factor_path = (
                    direction_consistency_factor_path
                )

            elif (
                self.factor_type
                in self.SIGNED_DIRECTION_CONSISTENCY_FACTOR_MAP
            ):

                self.factor_path = (
                    signed_direction_consistency_factor_path
                )

            else:

                self.factor_path = (
                    base_factor_path
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

            if (
                self.factor_name
                not in factor_data.files
            ):

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


            if self.factor_type in self.REVERSE_CONCENTRATION_FACTOR_TYPES:
                self.factor_matrix = -1.0 * self.factor_matrix


            if self.factor_type == 66:
                self.factor_matrix = -1.0 * self.factor_matrix


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


        if self.factor_type in self.COMBO_FACTOR_WEIGHTS:

            print(
                "comboWeight28:",
                self.combo_weight28
            )

            print(
                "comboWeight36:",
                self.combo_weight36
            )


        if self.factor_type in self.SYNTHESIS_FACTOR_TYPES:

            print(
                "synthesisType:",
                self.factor_type
            )

            if self.synthesis_lambda is not None:
                print(
                    "synthesisLambda:",
                    self.synthesis_lambda
                )


        if (
            self.factor_type in self.TIME_STRUCTURE_FACTOR_MAP
            or
            self.factor_type in self.TIME_ENSEMBLE_FACTOR_MAP
            or
            self.factor_type in self.ORTHO_BURST_FACTOR_TYPES
        ):

            print(
                "timeStructure:",
                self.time_structure_name
            )


        if (
            self.factor_type in self.CN_FACTOR_MAP
            or
            self.factor_type in self.CN_ENSEMBLE_FACTOR_MAP
            or
            self.factor_type in self.CN_THREE_VIEW_FACTOR_TYPES
        ):

            print(
                "concentrationNovelty:",
                self.cn_name
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


    def get_matrix_row(

        self,

        signal_date,

        matrix

    ):

        pos = np.searchsorted(
            self.factor_dates,
            signal_date
        )

        if pos >= len(self.factor_dates):
            return None

        if self.factor_dates[pos] != signal_date:
            return None

        return (
            matrix[pos]
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
    # Factor45-50 synthesis helpers
    # =========================================================

    @staticmethod
    def orthogonal_residual(
        y,
        x
    ):
        """
        Daily cross-sectional residual:
            y = a + b*x + residual

        Returns raw residuals on the original y scale.
        """
        y = np.asarray(
            y,
            dtype=np.float64
        )
        x = np.asarray(
            x,
            dtype=np.float64
        )

        out = np.full(
            len(y),
            np.nan,
            dtype=np.float64
        )

        valid = (
            np.isfinite(y)
            &
            np.isfinite(x)
        )

        n = int(
            valid.sum()
        )

        if n < 3:
            return out

        X = np.column_stack([
            np.ones(
                n,
                dtype=np.float64
            ),
            x[valid],
        ])

        beta, _, _, _ = np.linalg.lstsq(
            X,
            y[valid],
            rcond=None
        )

        out[valid] = (
            y[valid]
            -
            np.dot(
                X,
                beta
            )
        )

        return out


    def cs_rank_preserve_zero(
        self,
        values
    ):
        """
        Cross-sectional rank only ACTIVE non-zero values.

        Explicit zeros (for example from long-side veto)
        remain zero and are not allowed to re-enter through ranking.
        """
        x = np.asarray(
            values,
            dtype=np.float64
        )

        out = np.full(
            len(x),
            np.nan,
            dtype=np.float64
        )

        finite = np.isfinite(x)
        zero = finite & (x == 0.0)
        active = finite & (x != 0.0)

        out[zero] = 0.0

        if active.sum() > 0:
            out[active] = self.cs_rank(
                x[active]
            )

        return out


    def build_factor50_core(
        self,
        di,
        row28_raw,
        row36_raw
    ):
        """
        Exact reusable Factor50 core:
          source veto -> rank active values preserving veto zeros
          -> 75/25 rank fusion -> NO final rerank.
        """
        f28 = np.asarray(
            row28_raw,
            dtype=np.float64
        ).copy()

        f36 = np.asarray(
            row36_raw,
            dtype=np.float64
        ).copy()

        accel = self.calc_industry_acceleration(
            di
        )

        v28 = self.long_side_veto(
            f28,
            accel
        )

        v36 = self.long_side_veto(
            f36,
            accel
        )

        r28 = self.cs_rank_preserve_zero(
            v28
        )

        r36 = self.cs_rank_preserve_zero(
            v36
        )

        combo = np.full(
            len(f28),
            np.nan,
            dtype=np.float64
        )

        valid_combo = (
            np.isfinite(r28)
            &
            np.isfinite(r36)
        )

        combo[valid_combo] = (
            self.FACTOR50_WEIGHT28
            * r28[valid_combo]
            +
            self.FACTOR50_WEIGHT36
            * r36[valid_combo]
        )

        return combo


    def build_time_ensemble_factor(
        self,
        di,
        row28_raw,
        row36_raw,
        time_row_raw
    ):
        """
        Factor53/54:
          1) construct exact Factor50 core
          2) acceleration-veto the new time-structure factor
          3) rank both active signals while preserving explicit veto zeros
          4) 75/25 rank ensemble
          5) NO final rerank
        """
        factor50 = self.build_factor50_core(
            di,
            row28_raw,
            row36_raw
        )

        time_row = np.asarray(
            time_row_raw,
            dtype=np.float64
        ).copy()

        accel = self.calc_industry_acceleration(
            di
        )

        time_veto = self.long_side_veto(
            time_row,
            accel
        )

        r50 = self.cs_rank_preserve_zero(
            factor50
        )

        rnew = self.cs_rank_preserve_zero(
            time_veto
        )

        combo = np.full(
            len(factor50),
            np.nan,
            dtype=np.float64
        )

        valid = (
            np.isfinite(r50)
            &
            np.isfinite(rnew)
        )

        combo[valid] = (
            self.TIME_ENSEMBLE_WEIGHT50
            * r50[valid]
            +
            self.TIME_ENSEMBLE_WEIGHT_NEW
            * rnew[valid]
        )

        return combo


    def build_factor54_core(
        self,
        di,
        row28_raw,
        row36_raw,
        burst_row_raw
    ):
        """
        Exact reusable Factor54 core:
          F50 + raw burst rank ensemble, 75/25, no final rerank.
        """
        factor50 = self.build_factor50_core(
            di,
            row28_raw,
            row36_raw
        )

        burst = np.asarray(
            burst_row_raw,
            dtype=np.float64
        ).copy()

        accel = self.calc_industry_acceleration(di)

        burst_veto = self.long_side_veto(
            burst,
            accel
        )

        r50 = self.cs_rank_preserve_zero(
            factor50
        )

        rburst = self.cs_rank_preserve_zero(
            burst_veto
        )

        out = np.full(
            len(factor50),
            np.nan,
            dtype=np.float64
        )

        valid = (
            np.isfinite(r50)
            &
            np.isfinite(rburst)
        )

        out[valid] = (
            self.TIME_ENSEMBLE_WEIGHT50
            * r50[valid]
            +
            self.TIME_ENSEMBLE_WEIGHT_NEW
            * rburst[valid]
        )

        return out


    def build_cn_ensemble_factor(
        self,
        di,
        row28_raw,
        row36_raw,
        burst_row_raw,
        cn_row_raw
    ):
        """
        Factor58/59:
          F54 core
          + concentration OR first-appearance novelty
          -> rank each, preserve explicit veto zeros
          -> 75/25 ensemble
          -> no final rerank.
        """
        factor54 = self.build_factor54_core(
            di,
            row28_raw,
            row36_raw,
            burst_row_raw
        )

        cn = np.asarray(
            cn_row_raw,
            dtype=np.float64
        ).copy()

        accel = self.calc_industry_acceleration(di)

        cn_veto = self.long_side_veto(
            cn,
            accel
        )

        r54 = self.cs_rank_preserve_zero(
            factor54
        )

        rcn = self.cs_rank_preserve_zero(
            cn_veto
        )

        out = np.full(
            len(factor54),
            np.nan,
            dtype=np.float64
        )

        valid = (
            np.isfinite(r54)
            &
            np.isfinite(rcn)
        )

        out[valid] = (
            self.CN_WEIGHT54
            * r54[valid]
            +
            self.CN_WEIGHT_NEW
            * rcn[valid]
        )

        return out


    def build_aux_row(
        self,
        di,
        kind,
        concentration_row_raw=None,
        semantic_cont_row_raw=None
    ):
        """
        Build the auxiliary signal used by factors 67-70.

        kind='conc':
            reversed concentration (Factor61 logic)

        kind='semantic':
            reversed continuous semantic novelty (Factor66 logic)
        """
        accel = self.calc_industry_acceleration(di)

        if kind == "conc":
            aux = (
                -1.0
                * np.asarray(
                    concentration_row_raw,
                    dtype=np.float64
                )
            )

        elif kind == "semantic":
            aux = (
                -1.0
                * np.asarray(
                    semantic_cont_row_raw,
                    dtype=np.float64
                )
            )

        else:
            raise ValueError("unknown aux kind: %s" % kind)

        return self.long_side_veto(
            aux,
            accel
        )


    def build_long_disagreement_gate(
        self,
        factor54,
        aux
    ):
        """
        Do NOT blend rankings.

        Keep F54 unchanged except:
        - F54 < 0 is a long candidate under sign=-1.
        - If auxiliary signal is strongly adverse (aux >= +0.60),
          veto that F54 long candidate to zero.
        - Shorts are untouched.

        This tests auxiliary information as a filter rather than
        as an equal-ranking competitor.
        """
        base = np.asarray(
            factor54,
            dtype=np.float64
        ).copy()

        aux = np.asarray(
            aux,
            dtype=np.float64
        )

        valid = (
            np.isfinite(base)
            &
            np.isfinite(aux)
        )

        veto = (
            valid
            &
            (base < 0.0)
            &
            (aux >= self.GATE_ADVERSE_CUT)
        )

        base[veto] = 0.0

        return base


    def industry_row_to_stock_sleeve(
        self,
        sw1,
        industry_row,
        valid_mask=None
    ):
        """
        Map one industry signal to stock alpha exactly as a sleeve:
        - apply extremeCut BEFORE combination
        - apply sign
        - apply valid universe
        - normalize each sleeve separately by L1 gross

        This is designed to mimic independent-book combination
        while still returning ONE final alpha vector from ONE module.
        """
        out = np.zeros(
            len(sw1),
            dtype=np.float64
        )

        row = np.asarray(
            industry_row,
            dtype=np.float64
        )

        for j in range(
            len(self.factor_codes)
        ):
            value = float(row[j])

            if not np.isfinite(value):
                continue

            if abs(value) < self.extreme_cut:
                continue

            code = int(
                self.factor_codes[j]
            )

            mask = (
                sw1 == code
            )

            if np.any(mask):
                out[mask] = (
                    self.sign
                    * value
                )

        if valid_mask is not None:
            out[
                ~valid_mask
            ] = 0.0

        gross = np.sum(
            np.abs(out)
        )

        if gross > 0.0:
            out = (
                out / gross
            )

        return out


    def build_reverse_concentration_ensemble(
        self,
        di,
        row28_raw,
        row36_raw,
        burst_row_raw,
        concentration_row_raw
    ):
        factor54 = self.build_factor54_core(
            di, row28_raw, row36_raw, burst_row_raw
        )

        reversed_conc = -1.0 * np.asarray(
            concentration_row_raw,
            dtype=np.float64
        )

        accel = self.calc_industry_acceleration(di)

        conc_veto = self.long_side_veto(
            reversed_conc,
            accel
        )

        r54 = self.cs_rank_preserve_zero(factor54)
        rconc = self.cs_rank_preserve_zero(conc_veto)

        out = np.full(
            len(factor54),
            np.nan,
            dtype=np.float64
        )

        valid = np.isfinite(r54) & np.isfinite(rconc)

        out[valid] = (
            self.NEXT_WEIGHT54 * r54[valid]
            + self.NEXT_WEIGHT_NEW * rconc[valid]
        )

        return out


    def build_semantic_ensemble(
        self,
        di,
        row28_raw,
        row36_raw,
        burst_row_raw,
        semantic_row_raw
    ):
        factor54 = self.build_factor54_core(
            di, row28_raw, row36_raw, burst_row_raw
        )

        semantic = np.asarray(
            semantic_row_raw,
            dtype=np.float64
        )

        accel = self.calc_industry_acceleration(di)

        semantic_veto = self.long_side_veto(
            semantic,
            accel
        )

        r54 = self.cs_rank_preserve_zero(factor54)
        rsem = self.cs_rank_preserve_zero(semantic_veto)

        out = np.full(
            len(factor54),
            np.nan,
            dtype=np.float64
        )

        valid = np.isfinite(r54) & np.isfinite(rsem)

        out[valid] = (
            self.NEXT_WEIGHT54 * r54[valid]
            + self.NEXT_WEIGHT_NEW * rsem[valid]
        )

        return out


    def build_cn_three_view_factor(
        self,
        di,
        row28_raw,
        row36_raw,
        burst_row_raw,
        concentration_row_raw,
        novelty_row_raw
    ):
        """
        Factor60:
          70% Rank(F54)
          15% Rank(concentration)
          15% Rank(first-appearance novelty)
          preserving explicit veto zeros, no final rerank.
        """
        factor54 = self.build_factor54_core(
            di,
            row28_raw,
            row36_raw,
            burst_row_raw
        )

        accel = self.calc_industry_acceleration(di)

        conc_veto = self.long_side_veto(
            np.asarray(concentration_row_raw, dtype=np.float64),
            accel
        )

        novelty_veto = self.long_side_veto(
            np.asarray(novelty_row_raw, dtype=np.float64),
            accel
        )

        r54 = self.cs_rank_preserve_zero(factor54)
        rconc = self.cs_rank_preserve_zero(conc_veto)
        rnov = self.cs_rank_preserve_zero(novelty_veto)

        out = np.full(
            len(factor54),
            np.nan,
            dtype=np.float64
        )

        valid = (
            np.isfinite(r54)
            &
            np.isfinite(rconc)
            &
            np.isfinite(rnov)
        )

        out[valid] = (
            self.CN3_WEIGHT54 * r54[valid]
            +
            self.CN3_WEIGHT_CONC * rconc[valid]
            +
            self.CN3_WEIGHT_NOVELTY * rnov[valid]
        )

        return out


    def build_orthogonal_burst_factor(
        self,
        di,
        row28_raw,
        row36_raw,
        burst_row_raw
    ):
        """
        Factor55:
          1) build exact Factor50 core
          2) cross-sectionally orthogonalize burst to Factor50
          3) apply acceleration long-side veto to the burst residual
          4) rank Factor50 and residual separately, preserving explicit zeros
          5) 75/25 rank ensemble
          6) no final rerank
        """
        factor50 = self.build_factor50_core(
            di,
            row28_raw,
            row36_raw
        )

        burst = np.asarray(
            burst_row_raw,
            dtype=np.float64
        ).copy()

        burst_perp = self.orthogonal_residual(
            burst,
            factor50
        )

        accel = self.calc_industry_acceleration(
            di
        )

        burst_perp_veto = self.long_side_veto(
            burst_perp,
            accel
        )

        r50 = self.cs_rank_preserve_zero(
            factor50
        )

        rburst = self.cs_rank_preserve_zero(
            burst_perp_veto
        )

        combo = np.full(
            len(factor50),
            np.nan,
            dtype=np.float64
        )

        valid = (
            np.isfinite(r50)
            &
            np.isfinite(rburst)
        )

        combo[valid] = (
            self.FACTOR55_WEIGHT50
            * r50[valid]
            +
            self.FACTOR55_WEIGHT_BURST
            * rburst[valid]
        )

        return combo


    def build_synthesis_factor(
        self,
        di,
        row28_raw,
        row36_raw
    ):
        """
        Build one final industry factor for Factor45-50.
        """
        f28 = np.asarray(
            row28_raw,
            dtype=np.float64
        ).copy()

        f36 = np.asarray(
            row36_raw,
            dtype=np.float64
        ).copy()

        valid = (
            np.isfinite(f28)
            &
            np.isfinite(f36)
        )

        # -----------------------------------------------------
        # Factor45-47:
        # orthogonal F36 increment -> final rank -> one veto
        # -----------------------------------------------------
        if self.factor_type in self.ORTHO_LAMBDAS:

            f36_perp = self.orthogonal_residual(
                f36,
                f28
            )

            combo = np.full(
                len(f28),
                np.nan,
                dtype=np.float64
            )

            combo[valid] = (
                f28[valid]
                +
                self.synthesis_lambda
                * f36_perp[valid]
            )

            ranked = self.cs_rank(
                combo
            )

            accel = self.calc_industry_acceleration(
                di
            )

            return self.long_side_veto(
                ranked,
                accel
            )

        # -----------------------------------------------------
        # Factor48:
        # protect strong F28; only weak/mid F28 can absorb F36.
        # No rerank before veto so strong F28 stays unchanged.
        # -----------------------------------------------------
        elif self.factor_type == 48:

            combo = f28.copy()

            weak = (
                valid
                &
                (
                    np.abs(f28)
                    < self.FACTOR48_STRONG_CUT
                )
            )

            combo[weak] = (
                f28[weak]
                +
                self.FACTOR48_WEAK_WEIGHT
                * f36[weak]
            )

            # Keep scale bounded and comparable with rank factors.
            combo[valid] = np.clip(
                combo[valid],
                -1.0,
                1.0
            )

            accel = self.calc_industry_acceleration(
                di
            )

            return self.long_side_veto(
                combo,
                accel
            )

        # -----------------------------------------------------
        # Factor49:
        # F36 may CONFIRM F28, but may not reverse/cancel it.
        # -----------------------------------------------------
        elif self.factor_type == 49:

            combo = f28.copy()

            same = (
                valid
                &
                (f28 != 0.0)
                &
                (f36 != 0.0)
                &
                (
                    np.sign(f28)
                    ==
                    np.sign(f36)
                )
            )

            boost = (
                1.0
                +
                self.FACTOR49_CONFIRM_STRENGTH
                * np.abs(f36[same])
            )

            combo[same] = (
                f28[same]
                * boost
            )

            combo[valid] = np.clip(
                combo[valid],
                -1.0,
                1.0
            )

            accel = self.calc_industry_acceleration(
                di
            )

            return self.long_side_veto(
                combo,
                accel
            )

        # -----------------------------------------------------
        # Factor50:
        # veto first; re-rank active values in each source while
        # preserving zeros; then rank-fuse without final rerank.
        # -----------------------------------------------------
        elif self.factor_type == 50:

            return self.build_factor50_core(
                di,
                f28,
                f36
            )

        else:
            raise ValueError(
                "invalid synthesis factorType"
            )


    # =========================================================

    # Runtime factors 23-31

    # =========================================================



    def apply_runtime_factor(

        self,

        di,

        factor_row

    ):



        # -----------------------------------------------------
        # Factor38-40 are built from two source matrices in
        # generate(), because both raw rows are required.
        # -----------------------------------------------------

        if (
            self.factor_type in self.COMBO_FACTOR_WEIGHTS
            or
            self.factor_type in self.SYNTHESIS_FACTOR_TYPES
            or
            self.factor_type in self.TIME_ENSEMBLE_FACTOR_MAP
            or
            self.factor_type in self.ORTHO_BURST_FACTOR_TYPES
            or
            self.factor_type in self.CN_ENSEMBLE_FACTOR_MAP
            or
            self.factor_type in self.CN_THREE_VIEW_FACTOR_TYPES
            or
            self.factor_type in self.REVERSE_CONCENTRATION_ENSEMBLE_TYPES
            or
            self.factor_type in self.SEMANTIC_ENSEMBLE_FACTOR_MAP
        ):

            return factor_row


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


        # -----------------------------------------------------
        # Factor32-37:
        # policy-type text factors
        #
        # Keep the Factor28 price logic fixed:
        # only veto LONG signals when 5d acceleration is
        # cross-sectionally very weak.
        # -----------------------------------------------------

        elif self.factor_type in self.POLICY_TYPE_FACTOR_MAP:

            state = self.calc_industry_acceleration(
                di
            )

            return self.long_side_veto(
                factor_row,
                state
            )


        # -----------------------------------------------------
        # Factor51-52: time-structure text factors.
        # Use exactly the same acceleration long-side veto.
        # -----------------------------------------------------

        elif self.factor_type in self.TIME_STRUCTURE_FACTOR_MAP:

            state = self.calc_industry_acceleration(
                di
            )

            return self.long_side_veto(
                factor_row,
                state
            )


        elif self.factor_type in self.CN_FACTOR_MAP:

            state = self.calc_industry_acceleration(
                di
            )

            return self.long_side_veto(
                factor_row,
                state
            )


        elif self.factor_type in self.REVERSE_CONCENTRATION_FACTOR_TYPES:

            state = self.calc_industry_acceleration(di)

            return self.long_side_veto(
                factor_row,
                state
            )


        elif self.factor_type in self.SEMANTIC_FACTOR_MAP:

            state = self.calc_industry_acceleration(di)

            return self.long_side_veto(
                factor_row,
                state
            )


        elif self.factor_type in self.SEMANTIC_CONT_FACTOR_MAP:

            state = self.calc_industry_acceleration(di)

            return self.long_side_veto(
                factor_row,
                state
            )


        elif self.factor_type in self.DIRECTION_CONSISTENCY_FACTOR_MAP:

            state = self.calc_industry_acceleration(di)

            return self.long_side_veto(
                factor_row,
                state
            )


        elif self.factor_type in self.SIGNED_DIRECTION_CONSISTENCY_FACTOR_MAP:

            state = self.calc_industry_acceleration(di)

            return self.long_side_veto(
                factor_row,
                state
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

        stock_sleeve_rows = None



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



        # =====================================================
        # Factor38-40 legacy simple blends; Factor45-50 use dedicated synthesis:
        #
        # 1) load raw Factor22 and raw Factor36 text signals
        # 2) apply the SAME acceleration long-side veto to each
        # 3) combine at industry level
        # 4) cross-sectionally re-rank the combined signal
        #
        # This keeps the combination structurally clean:
        # no stock-level blending and no extra threshold sweep.
        # =====================================================

        if (
            self.factor_type in self.GATE_FACTOR_TYPES
            or
            self.factor_type in self.STOCK_SLEEVE_FACTOR_TYPES
            or
            self.factor_type in self.STOCK_SLEEVE_ROBUST_FACTOR_TYPES
            or
            self.factor_type in self.F54_F73_STOCK_SLEEVE_TYPES
        ):

            row28_raw = self.get_matrix_row(
                signal_date,
                self.factor28_matrix
            )

            row36_raw = self.get_matrix_row(
                signal_date,
                self.factor36_matrix
            )

            burst_row_raw = self.get_matrix_row(
                signal_date,
                self.burst_matrix
            )

            concentration_row_raw = self.get_matrix_row(
                signal_date,
                self.concentration_matrix
            )

            semantic_cont_row_raw = self.get_matrix_row(
                signal_date,
                self.semantic_cont_matrix
            )

            factor73_row_raw = None
            if self.factor_type in self.F54_F73_STOCK_SLEEVE_TYPES:
                factor73_row_raw = self.get_matrix_row(
                    signal_date,
                    self.factor73_matrix
                )

            if (
                row28_raw is None
                or row36_raw is None
                or burst_row_raw is None
                or concentration_row_raw is None
                or semantic_cont_row_raw is None
                or (
                    self.factor_type in self.F54_F73_STOCK_SLEEVE_TYPES
                    and factor73_row_raw is None
                )
            ):
                return

            factor54 = self.build_factor54_core(
                di,
                row28_raw,
                row36_raw,
                burst_row_raw
            )

            if self.factor_type in self.F54_F73_STOCK_SLEEVE_TYPES:
                state = self.calc_industry_acceleration(di)

                factor73 = self.long_side_veto(
                    np.asarray(
                        factor73_row_raw,
                        dtype=np.float64
                    ),
                    state
                )

                stock_sleeve_rows = (
                    factor54,
                    factor73
                )

                factor_row = factor54

            elif self.factor_type in {67, 69, 71}:
                aux = self.build_aux_row(
                    di,
                    "conc",
                    concentration_row_raw=concentration_row_raw
                )

            else:
                aux = self.build_aux_row(
                    di,
                    "semantic",
                    semantic_cont_row_raw=semantic_cont_row_raw
                )

            if self.factor_type in self.F54_F73_STOCK_SLEEVE_TYPES:
                # Factor74/75 were already fully constructed above
                # as (factor54, factor73). Do NOT fall through to the
                # generic auxiliary branch, because there is no `aux`
                # variable for these factor types.
                pass

            elif self.factor_type in self.GATE_FACTOR_TYPES:
                factor_row = self.build_long_disagreement_gate(
                    factor54,
                    aux
                )

            else:
                stock_sleeve_rows = (
                    factor54,
                    aux
                )

                # Placeholder only; standard industry->stock mapping
                # is bypassed below for stock-sleeve factors.
                factor_row = factor54

        elif self.factor_type in self.SEMANTIC_ENSEMBLE_FACTOR_MAP:

            row28_raw = self.get_matrix_row(
                signal_date,
                self.factor28_matrix
            )

            row36_raw = self.get_matrix_row(
                signal_date,
                self.factor36_matrix
            )

            burst_row_raw = self.get_matrix_row(
                signal_date,
                self.burst_matrix
            )

            semantic_row_raw = self.get_matrix_row(
                signal_date,
                self.semantic_matrix
            )

            if (
                row28_raw is None
                or row36_raw is None
                or burst_row_raw is None
                or semantic_row_raw is None
            ):
                return

            factor_row = self.build_semantic_ensemble(
                di,
                row28_raw,
                row36_raw,
                burst_row_raw,
                semantic_row_raw
            )

        elif self.factor_type in self.REVERSE_CONCENTRATION_ENSEMBLE_TYPES:

            row28_raw = self.get_matrix_row(
                signal_date,
                self.factor28_matrix
            )

            row36_raw = self.get_matrix_row(
                signal_date,
                self.factor36_matrix
            )

            burst_row_raw = self.get_matrix_row(
                signal_date,
                self.burst_matrix
            )

            concentration_row_raw = self.get_matrix_row(
                signal_date,
                self.concentration_matrix
            )

            if (
                row28_raw is None
                or row36_raw is None
                or burst_row_raw is None
                or concentration_row_raw is None
            ):
                return

            factor_row = self.build_reverse_concentration_ensemble(
                di,
                row28_raw,
                row36_raw,
                burst_row_raw,
                concentration_row_raw
            )

        elif self.factor_type in self.CN_THREE_VIEW_FACTOR_TYPES:

            row28_raw = self.get_matrix_row(
                signal_date,
                self.factor28_matrix
            )

            row36_raw = self.get_matrix_row(
                signal_date,
                self.factor36_matrix
            )

            burst_row_raw = self.get_matrix_row(
                signal_date,
                self.burst_matrix
            )

            concentration_row_raw = self.get_matrix_row(
                signal_date,
                self.concentration_matrix
            )

            novelty_row_raw = self.get_matrix_row(
                signal_date,
                self.novelty_matrix
            )

            if any(
                x is None
                for x in [
                    row28_raw,
                    row36_raw,
                    burst_row_raw,
                    concentration_row_raw,
                    novelty_row_raw,
                ]
            ):
                return

            factor_row = self.build_cn_three_view_factor(
                di,
                row28_raw,
                row36_raw,
                burst_row_raw,
                concentration_row_raw,
                novelty_row_raw
            )

        elif self.factor_type in self.CN_ENSEMBLE_FACTOR_MAP:

            row28_raw = self.get_matrix_row(
                signal_date,
                self.factor28_matrix
            )

            row36_raw = self.get_matrix_row(
                signal_date,
                self.factor36_matrix
            )

            burst_row_raw = self.get_matrix_row(
                signal_date,
                self.burst_matrix
            )

            cn_row_raw = self.get_matrix_row(
                signal_date,
                self.cn_matrix
            )

            if (
                row28_raw is None
                or row36_raw is None
                or burst_row_raw is None
                or cn_row_raw is None
            ):
                return

            factor_row = self.build_cn_ensemble_factor(
                di,
                row28_raw,
                row36_raw,
                burst_row_raw,
                cn_row_raw
            )

        elif self.factor_type in self.ORTHO_BURST_FACTOR_TYPES:

            row28_raw = self.get_matrix_row(
                signal_date,
                self.factor28_matrix
            )

            row36_raw = self.get_matrix_row(
                signal_date,
                self.factor36_matrix
            )

            burst_row_raw = self.get_matrix_row(
                signal_date,
                self.time_structure_matrix
            )

            if (
                row28_raw is None
                or
                row36_raw is None
                or
                burst_row_raw is None
            ):
                return

            factor_row = self.build_orthogonal_burst_factor(
                di,
                row28_raw,
                row36_raw,
                burst_row_raw
            )

        elif self.factor_type in self.TIME_ENSEMBLE_FACTOR_MAP:

            row28_raw = self.get_matrix_row(
                signal_date,
                self.factor28_matrix
            )

            row36_raw = self.get_matrix_row(
                signal_date,
                self.factor36_matrix
            )

            time_row_raw = self.get_matrix_row(
                signal_date,
                self.time_structure_matrix
            )

            if (
                row28_raw is None
                or
                row36_raw is None
                or
                time_row_raw is None
            ):
                return

            factor_row = self.build_time_ensemble_factor(
                di,
                row28_raw,
                row36_raw,
                time_row_raw
            )

        elif self.factor_type in self.SYNTHESIS_FACTOR_TYPES:

            row28_raw = self.get_matrix_row(
                signal_date,
                self.factor28_matrix
            )

            row36_raw = self.get_matrix_row(
                signal_date,
                self.factor36_matrix
            )

            if (
                row28_raw is None
                or
                row36_raw is None
            ):
                return

            factor_row = self.build_synthesis_factor(
                di,
                row28_raw,
                row36_raw
            )

        elif self.factor_type in self.COMBO_FACTOR_WEIGHTS:

            row28_raw = self.get_matrix_row(
                signal_date,
                self.factor28_matrix
            )

            row36_raw = self.get_matrix_row(
                signal_date,
                self.factor36_matrix
            )

            if (
                row28_raw is None
                or
                row36_raw is None
            ):
                return

            accel_state = (
                self.calc_industry_acceleration(
                    di
                )
            )

            row28 = self.long_side_veto(
                row28_raw,
                accel_state
            )

            row36 = self.long_side_veto(
                row36_raw,
                accel_state
            )

            factor_row = (
                self.combo_weight28
                * row28
                +
                self.combo_weight36
                * row36
            )

            # Preserve NaNs where neither source is valid.
            valid_combo = (
                np.isfinite(row28)
                &
                np.isfinite(row36)
            )

            ranked = self.cs_rank(
                factor_row[
                    valid_combo
                ]
            )

            combined = np.full(
                len(factor_row),
                np.nan,
                dtype=np.float64
            )

            combined[
                valid_combo
            ] = ranked

            factor_row = combined

        else:

            factor_row = (
                self.get_factor_row(
                    signal_date
                )
            )

            if factor_row is None:
                return

            # Factor23-37:
            # market reaction / long-side veto.
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

        if stock_sleeve_rows is not None:

            try:
                valid_now = np.asarray(
                    self.valid[di]
                ).ravel().astype(bool)

            except Exception:
                valid_now = None

            sleeve54 = self.industry_row_to_stock_sleeve(
                sw1,
                stock_sleeve_rows[0],
                valid_mask=valid_now
            )

            sleeve_aux = self.industry_row_to_stock_sleeve(
                sw1,
                stock_sleeve_rows[1],
                valid_mask=valid_now
            )

            if self.factor_type == 74:

                self.alpha[:] = (
                    self.F74_WEIGHT54
                    * sleeve54
                    +
                    self.F74_WEIGHT73
                    * sleeve_aux
                )

            elif self.factor_type == 75:

                self.alpha[:] = (
                    self.F75_WEIGHT54
                    * sleeve54
                    +
                    self.F75_WEIGHT73
                    * sleeve_aux
                )

            elif self.factor_type in self.STOCK_SLEEVE_ROBUST_FACTOR_TYPES:

                self.alpha[:] = (
                    self.SLEEVE71_WEIGHT54
                    * sleeve54
                    +
                    self.SLEEVE71_WEIGHT_AUX
                    * sleeve_aux
                )

            else:

                self.alpha[:] = (
                    self.SLEEVE_WEIGHT54
                    * sleeve54
                    +
                    self.SLEEVE_WEIGHT_AUX
                    * sleeve_aux
                )

            assigned_count = int(
                np.sum(
                    np.abs(self.alpha) > 0.0
                )
            )

        else:

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