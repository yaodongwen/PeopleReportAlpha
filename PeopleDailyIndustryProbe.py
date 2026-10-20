# -*- coding: utf-8 -*-

import os
import sys
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

from sw_industry_mapper import map_sw2_to_sw1


class PeopleDailyIndustryProbe(AlphaBase):

    def __init__(self, cfg):

        super().__init__(cfg)

        self.industry = dr.getData("industry")

        self.printed = False


    def generate(self, di):

        if self.printed:
            return

        raw = np.asarray(
            self.industry[di]
        ).ravel()

        valid_mask = (
            raw != -999999999
        )

        valid_raw = raw[
            valid_mask
        ]

        mapped = np.array(
            [
                map_sw2_to_sw1(x)
                for x in valid_raw
            ],
            dtype=object
        )

        mapped_mask = np.array(
            [
                x is not None
                for x in mapped
            ]
        )

        print("")
        print("=" * 80)
        print("SW INDUSTRY MAPPING PROBE")
        print("=" * 80)

        print(
            "di:",
            di
        )

        print(
            "all stocks:",
            len(raw)
        )

        print(
            "valid SW2 stocks:",
            len(valid_raw)
        )

        print(
            "mapped stocks:",
            int(mapped_mask.sum())
        )

        print(
            "mapping coverage:",
            "%.2f%%"
            % (
                100.0
                * mapped_mask.sum()
                / max(
                    len(valid_raw),
                    1
                )
            )
        )

        print("")
        print("-" * 80)
        print("MAPPED LEVEL1 DISTRIBUTION")
        print("-" * 80)

        mapped_values = np.array(
            [
                int(x)
                for x in mapped[
                    mapped_mask
                ]
            ],
            dtype=np.int64
        )

        values, counts = np.unique(
            mapped_values,
            return_counts=True
        )

        for value, count in zip(
            values,
            counts
        ):

            print(
                "%d %d"
                % (
                    value,
                    count
                )
            )

        print("")
        print("-" * 80)
        print("UNMAPPED RAW INDUSTRIES")
        print("-" * 80)

        unmapped_raw = valid_raw[
            ~mapped_mask
        ]

        values, counts = np.unique(
            unmapped_raw,
            return_counts=True
        )

        for value, count in zip(
            values,
            counts
        ):

            print(
                "%d %d"
                % (
                    value,
                    count
                )
            )

        print("")
        print("=" * 80)

        self.printed = True