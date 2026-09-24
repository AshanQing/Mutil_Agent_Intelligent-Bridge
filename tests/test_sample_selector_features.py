from __future__ import annotations

import numpy as np

from scripts.selector import SampleSelector


def _selector() -> SampleSelector:
    return SampleSelector.__new__(SampleSelector)


def test_nested_vertical_profile_produces_real_terrain_features() -> None:
    data = {
        "K": {
            "纵断面结构": {
                "设计线高程序列": [
                    {"桩号": 0.0, "高程": 10.0},
                    {"桩号": 100.0, "高程": 20.0},
                ],
                "地形线高程序列": [
                    {"桩号": 0.0, "高程": 0.0},
                    {"桩号": 100.0, "高程": 15.0},
                ],
            }
        }
    }

    vector, validity = _selector()._extract_engineering_feature_payload(data)

    assert vector[0] == 10.0
    assert vector[1] == 7.5
    assert validity[:2].tolist() == [True, True]


def test_k_line_terrain_feature_table_is_supported() -> None:
    data = {
        "K": {
            "地形特征表": [
                {"桩号": "K1+000", "高差（设计线-地面线）": 12.0},
                {"桩号": "K1+020", "高差（设计线-地面线）": 4.0},
            ]
        }
    }

    vector, validity = _selector()._extract_engineering_feature_payload(data)

    assert vector[:2].tolist() == [12.0, 8.0]
    assert validity[:2].tolist() == [True, True]


def test_missing_terrain_is_not_equivalent_to_valid_zero_relief() -> None:
    selector = _selector()
    missing_vec, missing_mask = selector._extract_engineering_feature_payload({})
    zero_vec, zero_mask = selector._extract_engineering_feature_payload(
        {"K": {"地形特征表": [{"高差": 0.0}]}}
    )

    distance = selector._calculate_weighted_distance(
        missing_vec,
        zero_vec,
        missing_mask,
        zero_mask,
    )

    assert np.allclose(missing_vec[:2], zero_vec[:2])
    assert distance >= 0.7
