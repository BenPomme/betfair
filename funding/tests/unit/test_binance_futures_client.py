from __future__ import annotations

from funding.data.binance_futures_client import _normalize_rows


def test_normalize_rows_accepts_json_string_payload():
    rows = _normalize_rows('[{"longShortRatio":"1.2","longAccount":"0.55","shortAccount":"0.45"}]')

    assert rows == [
        {
            "longShortRatio": "1.2",
            "longAccount": "0.55",
            "shortAccount": "0.45",
        }
    ]


def test_normalize_rows_accepts_list_of_json_strings():
    rows = _normalize_rows(
        [
            '{"longShortRatio":"1.1","longAccount":"0.52","shortAccount":"0.48"}',
            {"longShortRatio": "0.9", "longAccount": "0.47", "shortAccount": "0.53"},
        ]
    )

    assert rows == [
        {
            "longShortRatio": "1.1",
            "longAccount": "0.52",
            "shortAccount": "0.48",
        },
        {
            "longShortRatio": "0.9",
            "longAccount": "0.47",
            "shortAccount": "0.53",
        },
    ]
