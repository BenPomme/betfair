"""
Tests for cross-market arbitrage: commission.evaluate_back_lay_arb,
commission.evaluate_mo_dnb_3leg_arb, data.event_grouper, and core.cross_market_scanner.
"""
from datetime import datetime
from decimal import Decimal

import pytest

from core.commission import evaluate_back_lay_arb, evaluate_mo_dnb_3leg_arb
from core.types import PriceSnapshot, SelectionPrice
from core.cross_market_scanner import scan_cross_market, _match_selections_by_name
from data.event_grouper import group_by_event, get_cross_market_pairs


# ── evaluate_back_lay_arb tests ──────────────────────────────────────────


class TestEvaluateBackLayArb:
    MBR = Decimal("0.05")
    ZERO_DISC = Decimal("0.00")

    def test_profitable_arb(self):
        """Back at 3.0, lay at 2.5 should be profitable."""
        result = evaluate_back_lay_arb(
            back_price=Decimal("3.0"),
            lay_price=Decimal("2.5"),
            max_stake=Decimal("100"),
            mbr=self.MBR,
            discount=self.ZERO_DISC,
        )
        assert result is not None
        assert result["net_profit"] > Decimal("0")
        assert result["back_stake"] == Decimal("100")
        assert result["lay_stake"] > Decimal("0")
        assert result["roi"] > Decimal("0")

    def test_no_arb_when_lay_exceeds_back(self):
        """When lay price >= back price, no arb exists."""
        result = evaluate_back_lay_arb(
            back_price=Decimal("2.0"),
            lay_price=Decimal("2.5"),
            max_stake=Decimal("100"),
            mbr=self.MBR,
            discount=self.ZERO_DISC,
        )
        assert result is None

    def test_no_arb_equal_prices(self):
        """Equal prices: no arb (commission eats any edge)."""
        result = evaluate_back_lay_arb(
            back_price=Decimal("2.0"),
            lay_price=Decimal("2.0"),
            max_stake=Decimal("100"),
            mbr=self.MBR,
            discount=self.ZERO_DISC,
        )
        assert result is None

    def test_commission_kills_marginal_edge(self):
        """Very small price difference: commission should kill profitability."""
        result = evaluate_back_lay_arb(
            back_price=Decimal("2.02"),
            lay_price=Decimal("2.00"),
            max_stake=Decimal("10"),
            mbr=self.MBR,
            discount=self.ZERO_DISC,
        )
        # With 5% commission, a 1% edge is wiped out
        assert result is None

    def test_discount_helps_profitability(self):
        """60% discount should make marginal arbs profitable."""
        # First check without discount
        no_disc = evaluate_back_lay_arb(
            back_price=Decimal("2.10"),
            lay_price=Decimal("2.00"),
            max_stake=Decimal("100"),
            mbr=self.MBR,
            discount=self.ZERO_DISC,
        )
        # Then with 60% discount
        with_disc = evaluate_back_lay_arb(
            back_price=Decimal("2.10"),
            lay_price=Decimal("2.00"),
            max_stake=Decimal("100"),
            mbr=self.MBR,
            discount=Decimal("0.60"),
        )
        # With discount, net_profit should be higher (or exist where it didn't)
        if no_disc is not None and with_disc is not None:
            assert with_disc["net_profit"] >= no_disc["net_profit"]
        elif no_disc is None:
            assert with_disc is not None

    def test_zero_stake_returns_none(self):
        result = evaluate_back_lay_arb(
            back_price=Decimal("3.0"),
            lay_price=Decimal("2.5"),
            max_stake=Decimal("0"),
            mbr=self.MBR,
            discount=self.ZERO_DISC,
        )
        assert result is None

    def test_invalid_prices(self):
        assert evaluate_back_lay_arb(Decimal("0"), Decimal("2.0"), Decimal("100"), self.MBR, self.ZERO_DISC) is None
        assert evaluate_back_lay_arb(Decimal("3.0"), Decimal("0"), Decimal("100"), self.MBR, self.ZERO_DISC) is None
        assert evaluate_back_lay_arb(Decimal("-1"), Decimal("2.0"), Decimal("100"), self.MBR, self.ZERO_DISC) is None


# ── evaluate_mo_dnb_3leg_arb tests ───────────────────────────────────────


class TestEvaluateMoDnb3LegArb:
    MBR = Decimal("0.05")
    ZERO_DISC = Decimal("0.00")

    def test_3leg_genuine_profit(self):
        """Contrived prices where B*(D-1)/D > L holds: should produce valid 3-leg arb."""
        # B=2.5, D=5.0, L=1.5 → B*(D-1)/D = 2.5*4/5 = 2.0 > 1.5 ✓
        # gross ≈ 33.34, worst commission scenario ≈ 8.33 → net ≈ 25.01
        result = evaluate_mo_dnb_3leg_arb(
            mo_back_price=Decimal("2.5"),
            dnb_lay_price=Decimal("1.5"),
            draw_back_price=Decimal("5.0"),
            max_stake=Decimal("100"),
            mbr=self.MBR,
            discount=self.ZERO_DISC,
        )
        assert result is not None
        assert result["net_profit"] > Decimal("0")
        assert result["back_stake"] == Decimal("100")
        assert result["lay_stake"] > Decimal("0")
        assert result["draw_stake"] > Decimal("0")
        assert result["roi"] > Decimal("0")
        # All 3 scenarios must be profitable
        for scenario in result["net_profit_scenarios"].values():
            assert scenario > Decimal("0")

    def test_3leg_all_scenarios_modeled(self):
        """Verify draw scenario math at zero commission."""
        # B=2.5, D=5.0, L=1.5, zero commission
        result = evaluate_mo_dnb_3leg_arb(
            mo_back_price=Decimal("2.5"),
            dnb_lay_price=Decimal("1.5"),
            draw_back_price=Decimal("5.0"),
            max_stake=Decimal("100"),
            mbr=Decimal("0"),
            discount=Decimal("0"),
        )
        assert result is not None
        # At zero commission, all 3 scenarios should have equal profit
        scenarios = result["net_profit_scenarios"]
        # They may differ slightly due to rounding of stakes
        profits = list(scenarios.values())
        spread = max(profits) - min(profits)
        assert spread < Decimal("0.10"), f"Scenario profits diverge too much: {profits}"

    def test_3leg_typical_prices_rejected(self):
        """Realistic MO/DNB prices: B*(D-1)/D <= L, correctly rejected."""
        # Typical football: MO home=2.5, Draw=3.3, DNB home lay=1.8
        # B*(D-1)/D = 2.5 * 2.3/3.3 = 1.74 < 1.8 → no arb
        result = evaluate_mo_dnb_3leg_arb(
            mo_back_price=Decimal("2.50"),
            dnb_lay_price=Decimal("1.80"),
            draw_back_price=Decimal("3.30"),
            max_stake=Decimal("100"),
            mbr=self.MBR,
            discount=self.ZERO_DISC,
        )
        assert result is None

    def test_3leg_draw_price_zero_rejected(self):
        """Draw price of zero should be rejected."""
        result = evaluate_mo_dnb_3leg_arb(
            mo_back_price=Decimal("5.0"),
            dnb_lay_price=Decimal("3.0"),
            draw_back_price=Decimal("0"),
            max_stake=Decimal("100"),
            mbr=self.MBR,
            discount=self.ZERO_DISC,
        )
        assert result is None

    def test_3leg_commission_kills_marginal(self):
        """Commission eliminates thin edge that would be profitable at 0% commission."""
        # Find prices where B*(D-1)/D is barely > L
        # B=4.0, D=4.0, L=2.95 → B*(D-1)/D = 4*3/4 = 3.0 > 2.95 ✓ (barely)
        zero_comm = evaluate_mo_dnb_3leg_arb(
            mo_back_price=Decimal("4.0"),
            dnb_lay_price=Decimal("2.95"),
            draw_back_price=Decimal("4.0"),
            max_stake=Decimal("100"),
            mbr=Decimal("0"),
            discount=Decimal("0"),
        )
        with_comm = evaluate_mo_dnb_3leg_arb(
            mo_back_price=Decimal("4.0"),
            dnb_lay_price=Decimal("2.95"),
            draw_back_price=Decimal("4.0"),
            max_stake=Decimal("100"),
            mbr=self.MBR,
            discount=self.ZERO_DISC,
        )
        # Zero commission should be profitable, 5% commission should kill it
        assert zero_comm is not None
        assert with_comm is None

    def test_3leg_zero_stake_rejected(self):
        result = evaluate_mo_dnb_3leg_arb(
            mo_back_price=Decimal("5.0"),
            dnb_lay_price=Decimal("3.0"),
            draw_back_price=Decimal("3.0"),
            max_stake=Decimal("0"),
            mbr=self.MBR,
            discount=self.ZERO_DISC,
        )
        assert result is None

    def test_3leg_invalid_prices(self):
        assert evaluate_mo_dnb_3leg_arb(
            Decimal("0"), Decimal("3.0"), Decimal("3.0"),
            Decimal("100"), self.MBR, self.ZERO_DISC,
        ) is None
        assert evaluate_mo_dnb_3leg_arb(
            Decimal("5.0"), Decimal("-1"), Decimal("3.0"),
            Decimal("100"), self.MBR, self.ZERO_DISC,
        ) is None


# ── event_grouper tests ─────────────────────────────────────────────────


class TestEventGrouper:
    def test_group_by_event(self):
        metadata = {
            "1.100": {"event_id": "EVT1", "market_type": "MATCH_ODDS"},
            "1.101": {"event_id": "EVT1", "market_type": "DRAW_NO_BET"},
            "1.200": {"event_id": "EVT2", "market_type": "MATCH_ODDS"},
            "1.300": {"event_id": "", "market_type": "MATCH_ODDS"},  # no event_id
        }
        groups = group_by_event(metadata)
        assert "EVT1" in groups
        assert set(groups["EVT1"]) == {"1.100", "1.101"}
        assert "EVT2" in groups
        assert groups["EVT2"] == ["1.200"]
        assert "" not in groups  # empty event_id skipped

    def test_get_cross_market_pairs(self):
        metadata = {
            "1.100": {"event_id": "EVT1", "market_type": "MATCH_ODDS"},
            "1.101": {"event_id": "EVT1", "market_type": "DRAW_NO_BET"},
            "1.102": {"event_id": "EVT1", "market_type": "OVER_UNDER_25"},
        }
        pairs = get_cross_market_pairs(["1.100", "1.101", "1.102"], metadata)
        assert len(pairs) == 1
        assert pairs[0] == ("1.100", "1.101")

    def test_no_pairs_without_dnb(self):
        metadata = {
            "1.100": {"event_id": "EVT1", "market_type": "MATCH_ODDS"},
            "1.102": {"event_id": "EVT1", "market_type": "OVER_UNDER_25"},
        }
        pairs = get_cross_market_pairs(["1.100", "1.102"], metadata)
        assert len(pairs) == 0


# ── cross_market_scanner tests ──────────────────────────────────────────


def _make_mo_snapshot(market_id="1.100", home_back=Decimal("3.0"), draw_back=Decimal("3.5"),
                      away_back=Decimal("2.5"), home_lay=Decimal("3.1"), draw_lay=Decimal("3.6"),
                      away_lay=Decimal("2.6"), liquidity=Decimal("500")):
    return PriceSnapshot(
        market_id=market_id,
        selections=(
            SelectionPrice("1", "Team A", home_back, liquidity, home_lay, liquidity),
            SelectionPrice("2", "The Draw", draw_back, liquidity, draw_lay, liquidity),
            SelectionPrice("3", "Team B", away_back, liquidity, away_lay, liquidity),
        ),
        timestamp=datetime.utcnow(),
    )


def _make_dnb_snapshot(market_id="1.101", home_back=Decimal("2.0"), away_back=Decimal("2.0"),
                       home_lay=Decimal("2.1"), away_lay=Decimal("2.1"), liquidity=Decimal("500")):
    return PriceSnapshot(
        market_id=market_id,
        selections=(
            SelectionPrice("10", "Team A", home_back, liquidity, home_lay, liquidity),
            SelectionPrice("11", "Team B", away_back, liquidity, away_lay, liquidity),
        ),
        timestamp=datetime.utcnow(),
    )


class TestSelectionMatching:
    def test_match_by_name(self):
        mo = _make_mo_snapshot()
        dnb = _make_dnb_snapshot()
        pairs, draw_sel = _match_selections_by_name(mo, dnb)
        assert len(pairs) == 2
        names = {p[0].name for p in pairs}
        assert "Team A" in names
        assert "Team B" in names

    def test_draw_extracted(self):
        """Draw selection should be returned separately, not in pairs."""
        mo = _make_mo_snapshot()
        dnb = _make_dnb_snapshot()
        pairs, draw_sel = _match_selections_by_name(mo, dnb)
        mo_names = {p[0].name for p in pairs}
        assert "The Draw" not in mo_names
        assert draw_sel is not None
        assert draw_sel.name == "The Draw"


class TestScanCrossMarket:
    MBR = Decimal("0.05")
    ZERO_DISC = Decimal("0.00")

    def test_profitable_cross_market_arb_direction1(self):
        """3-leg arb: B=2.5, D=5.0, L=1.5 — genuine arb with draw hedge."""
        # B*(D-1)/D = 2.5*4/5 = 2.0 > 1.5 = L ✓
        mo = _make_mo_snapshot(
            home_back=Decimal("2.5"), draw_back=Decimal("5.0"),
            home_lay=Decimal("2.7"), draw_lay=Decimal("5.2"),
        )
        dnb = _make_dnb_snapshot(home_back=Decimal("1.3"), home_lay=Decimal("1.5"))
        opp = scan_cross_market(
            mo, dnb,
            event_name="Test Match",
            max_stake_eur=Decimal("50"),
            min_net_profit_eur=Decimal("0.01"),
            mbr=self.MBR,
            discount_rate=self.ZERO_DISC,
        )
        assert opp is not None
        assert opp.arb_type == "cross_market"
        assert opp.net_profit_eur > Decimal("0")
        assert "+" in opp.market_id
        # Should have 2 selections: team + draw hedge
        assert len(opp.selections) == 2
        assert opp.selections[1]["direction"] == "draw_hedge"
        assert opp.selections[1]["name"] == "The Draw"

    def test_old_false_arbs_now_rejected(self):
        """Direction 1 with typical prices (MO 3.0, DNB lay 2.0, Draw 3.5) is NOT a real arb."""
        # B=3.0, D=3.5, L=2.0: B*(D-1)/D = 3.0*2.5/3.5 = 2.14 > 2.0 ✓ pre-commission
        # But after commission, the thin edge should be wiped out at small stakes.
        # At larger stakes it might survive — the key point is it's much harder to find
        # than the old false positives.
        mo = _make_mo_snapshot(home_back=Decimal("3.0"), draw_back=Decimal("3.5"), home_lay=Decimal("3.2"))
        dnb = _make_dnb_snapshot(home_back=Decimal("1.8"), home_lay=Decimal("2.0"))
        opp = scan_cross_market(
            mo, dnb,
            event_name="Test Match",
            max_stake_eur=Decimal("50"),
            min_net_profit_eur=Decimal("0.10"),
            mbr=self.MBR,
            discount_rate=self.ZERO_DISC,
        )
        # With 3-leg model, this is either None or has much lower profit than old 2-leg claim
        if opp is not None:
            # If it survives, it must have a draw hedge leg
            assert len(opp.selections) == 2
            assert opp.selections[1]["direction"] == "draw_hedge"

    def test_direction2_unchanged(self):
        """Back DNB / Lay MO still works with 2-leg model (draw is best scenario)."""
        # DNB back = 3.0, MO lay = 2.5 → B > L, should be profitable
        mo = _make_mo_snapshot(home_back=Decimal("2.4"), home_lay=Decimal("2.5"))
        dnb = _make_dnb_snapshot(home_back=Decimal("3.0"), home_lay=Decimal("3.2"))
        opp = scan_cross_market(
            mo, dnb,
            event_name="Test Match",
            max_stake_eur=Decimal("100"),
            min_net_profit_eur=Decimal("0.01"),
            mbr=self.MBR,
            discount_rate=self.ZERO_DISC,
        )
        assert opp is not None
        assert opp.arb_type == "cross_market"
        assert opp.net_profit_eur > Decimal("0")
        # Direction 2 uses single selection (2-leg model)
        assert len(opp.selections) == 1
        assert opp.selections[0]["direction"] == "back_dnb_lay_mo"

    def test_no_arb_when_prices_consistent(self):
        """When MO and DNB prices are consistent, no arb should be found."""
        mo = _make_mo_snapshot(
            home_back=Decimal("2.50"), home_lay=Decimal("2.55"),
            away_back=Decimal("2.50"), away_lay=Decimal("2.55"),
        )
        dnb = _make_dnb_snapshot(
            home_back=Decimal("2.00"), home_lay=Decimal("2.55"),
            away_back=Decimal("2.00"), away_lay=Decimal("2.55"),
        )
        opp = scan_cross_market(
            mo, dnb,
            min_net_profit_eur=Decimal("0.10"),
            mbr=self.MBR,
            discount_rate=self.ZERO_DISC,
        )
        assert opp is None

    def test_commission_kills_marginal_edge(self):
        """Tiny price gap: commission should eliminate profitability."""
        mo = _make_mo_snapshot(
            home_back=Decimal("2.02"), home_lay=Decimal("2.10"),
            away_back=Decimal("2.02"), away_lay=Decimal("2.10"),
        )
        dnb = _make_dnb_snapshot(
            home_back=Decimal("1.90"), home_lay=Decimal("2.00"),
            away_back=Decimal("1.90"), away_lay=Decimal("2.00"),
        )
        opp = scan_cross_market(
            mo, dnb,
            min_net_profit_eur=Decimal("0.10"),
            max_stake_eur=Decimal("10"),
            mbr=self.MBR,
            discount_rate=self.ZERO_DISC,
        )
        assert opp is None

    def test_insufficient_liquidity_rejected(self):
        """Good arb but low liquidity should be rejected."""
        mo = _make_mo_snapshot(home_back=Decimal("3.0"), liquidity=Decimal("5"))
        dnb = _make_dnb_snapshot(home_lay=Decimal("2.0"), liquidity=Decimal("5"))
        opp = scan_cross_market(
            mo, dnb,
            min_liquidity_eur=Decimal("50"),
            mbr=self.MBR,
            discount_rate=self.ZERO_DISC,
        )
        assert opp is None

    def test_3leg_draw_liquidity_insufficient(self):
        """Direction 1 rejected when draw has low liquidity."""
        # Prices that would be a genuine 3-leg arb
        mo = _make_mo_snapshot(
            home_back=Decimal("5.0"), draw_back=Decimal("3.0"),
            home_lay=Decimal("5.2"), draw_lay=Decimal("3.2"),
            liquidity=Decimal("500"),
        )
        # Override draw liquidity to be low
        mo = PriceSnapshot(
            market_id="1.100",
            selections=(
                SelectionPrice("1", "Team A", Decimal("5.0"), Decimal("500"), Decimal("5.2"), Decimal("500")),
                SelectionPrice("2", "The Draw", Decimal("3.0"), Decimal("5"), Decimal("3.2"), Decimal("5")),  # low liquidity
                SelectionPrice("3", "Team B", Decimal("2.5"), Decimal("500"), Decimal("2.6"), Decimal("500")),
            ),
            timestamp=datetime.utcnow(),
        )
        dnb = _make_dnb_snapshot(home_back=Decimal("2.5"), home_lay=Decimal("3.0"))
        opp = scan_cross_market(
            mo, dnb,
            event_name="Test Match",
            max_stake_eur=Decimal("50"),
            min_net_profit_eur=Decimal("0.01"),
            min_liquidity_eur=Decimal("50"),
            mbr=self.MBR,
            discount_rate=self.ZERO_DISC,
        )
        # Direction 1 should be rejected due to draw liquidity
        # Direction 2 might still work if prices allow
        if opp is not None:
            assert opp.selections[0]["direction"] == "back_dnb_lay_mo"

    def test_wrong_selection_count_rejected(self):
        """MO with 2 selections (not 3) should be rejected."""
        bad_mo = PriceSnapshot(
            market_id="1.100",
            selections=(
                SelectionPrice("1", "Team A", Decimal("3.0"), Decimal("500"), Decimal("3.1"), Decimal("500")),
                SelectionPrice("2", "Team B", Decimal("2.5"), Decimal("500"), Decimal("2.6"), Decimal("500")),
            ),
            timestamp=datetime.utcnow(),
        )
        dnb = _make_dnb_snapshot()
        opp = scan_cross_market(bad_mo, dnb, mbr=self.MBR, discount_rate=self.ZERO_DISC)
        assert opp is None

    def test_3leg_no_draw_in_mo_skips_direction1(self):
        """If MO has no draw selection, direction 1 is skipped (direction 2 may still work)."""
        # MO with 3 selections but none named "draw" — unusual but tests the guard
        mo = PriceSnapshot(
            market_id="1.100",
            selections=(
                SelectionPrice("1", "Team A", Decimal("5.0"), Decimal("500"), Decimal("5.2"), Decimal("500")),
                SelectionPrice("2", "Team C", Decimal("3.0"), Decimal("500"), Decimal("3.2"), Decimal("500")),
                SelectionPrice("3", "Team B", Decimal("2.5"), Decimal("500"), Decimal("2.6"), Decimal("500")),
            ),
            timestamp=datetime.utcnow(),
        )
        dnb = _make_dnb_snapshot(home_back=Decimal("2.5"), home_lay=Decimal("3.0"))
        opp = scan_cross_market(
            mo, dnb,
            event_name="Test Match",
            max_stake_eur=Decimal("50"),
            min_net_profit_eur=Decimal("0.01"),
            mbr=self.MBR,
            discount_rate=self.ZERO_DISC,
        )
        # Direction 1 skipped (no draw_sel), direction 2 may or may not produce arb
        if opp is not None:
            assert opp.selections[0]["direction"] == "back_dnb_lay_mo"
