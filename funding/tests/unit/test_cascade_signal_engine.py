from funding.portfolios.cascade_alpha.signal_engine import CascadeSignalEngine


def test_cascade_signal_engine_detects_continuation():
    engine = CascadeSignalEngine()
    signal = engine.classify(
        "ETHUSDT",
        {
            "spread_bps": 4.0,
            "depth_usd": 250000.0,
            "price_return_1m_pct": 0.9,
            "price_return_5m_pct": 1.8,
            "open_interest_change_5m_pct": 1.2,
            "taker_imbalance": 0.7,
            "liquidation_burst_usd": 300000.0,
            "mark_price": 2500.0,
        },
    )
    assert signal is not None
    assert signal["setup"] == "CONTINUATION"
    assert signal["side"] == "LONG"


def test_cascade_signal_engine_detects_snapback():
    engine = CascadeSignalEngine()
    signal = engine.classify(
        "SOLUSDT",
        {
            "spread_bps": 5.0,
            "depth_usd": 200000.0,
            "price_return_1m_pct": -0.4,
            "price_return_5m_pct": -2.1,
            "open_interest_change_5m_pct": -1.6,
            "taker_imbalance": -0.55,
            "liquidation_burst_usd": 220000.0,
            "mark_price": 150.0,
        },
    )
    assert signal is not None
    assert signal["setup"] == "SNAPBACK"
    assert signal["side"] == "LONG"


def test_cascade_signal_engine_rejects_wide_spread():
    engine = CascadeSignalEngine()
    signal = engine.classify(
        "BTCUSDT",
        {
            "spread_bps": 40.0,
            "depth_usd": 500000.0,
            "price_return_1m_pct": 1.1,
            "price_return_5m_pct": 2.0,
            "open_interest_change_5m_pct": 1.0,
            "taker_imbalance": 0.8,
            "liquidation_burst_usd": 500000.0,
            "mark_price": 90000.0,
        },
    )
    assert signal is None
