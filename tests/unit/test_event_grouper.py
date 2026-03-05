from data.event_grouper import get_cross_market_pairs


def test_get_cross_market_pairs_returns_pair_types() -> None:
    meta = {
        "mo": {"market_type": "MATCH_ODDS"},
        "dnb": {"market_type": "DRAW_NO_BET"},
        "ou25": {"market_type": "OVER_UNDER_25"},
        "btts": {"market_type": "BOTH_TEAMS_TO_SCORE"},
        "cs": {"market_type": "CORRECT_SCORE"},
    }
    pairs = get_cross_market_pairs(["mo", "dnb", "ou25", "btts", "cs"], meta)
    assert ("mo", "dnb", "mo_dnb") in pairs
    assert ("mo", "ou25", "mo_ou25") not in pairs
    assert ("mo", "btts", "mo_btts") not in pairs
    assert ("cs", "mo", "cs_mo") not in pairs

    experimental_pairs = get_cross_market_pairs(
        ["mo", "dnb", "ou25", "btts", "cs"],
        meta,
        include_experimental=True,
    )
    assert ("mo", "ou25", "mo_ou25") in experimental_pairs
    assert ("mo", "btts", "mo_btts") in experimental_pairs
    assert ("cs", "mo", "cs_mo") in experimental_pairs
