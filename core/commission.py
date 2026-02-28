"""
Decimal-based commission math for Betfair Spain.
Never use float for monetary values. All amounts rounded to 2 dp with ROUND_HALF_UP.
"""
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional

# Default Spain MBR; override from config at call site
MBR = Decimal("0.05")


def effective_rate(mbr: Decimal, discount: Decimal) -> Decimal:
    """Effective commission rate after discount: mbr * (1 - discount)."""
    return mbr * (Decimal("1") - discount)


def commission(
    net_winnings: Decimal, mbr: Decimal, discount: Decimal
) -> Decimal:
    """Commission on net winnings, rounded to 2 decimal places."""
    rate = effective_rate(mbr, discount)
    return (net_winnings * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def evaluate_back_back_arb(
    prices: List[Decimal],
    total_stake: Decimal,
    mbr: Decimal,
    discount: Decimal,
) -> Optional[dict]:
    """
    Evaluate a back-back arb: back all outcomes with equal-profit stakes.
    Returns dict with overround, stakes, net_profits, min_net_profit, roi if
    profitable after commission; None otherwise.
    Stakes are quantized to 2 dp.
    """
    if not prices or total_stake <= 0:
        return None
    overround = sum(Decimal("1") / p for p in prices)
    # Equal-profit stakes: stake_i = (1/p_i / overround) * total_stake
    stakes = [
        ((Decimal("1") / p / overround) * total_stake).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        for p in prices
    ]
    # Use actual sum of rounded stakes (may differ from total_stake by rounding)
    actual_total = sum(stakes)
    # Gross profit if outcome i wins: stake_i * price_i - actual_total
    profits = [s * p - actual_total for s, p in zip(stakes, prices)]
    net_profits = [g - commission(g, mbr, discount) for g in profits]

    if all(n > 0 for n in net_profits):
        return {
            "overround": overround,
            "stakes": stakes,
            "actual_total_stake": actual_total,
            "net_profits": net_profits,
            "min_net_profit": min(net_profits),
            "roi": min(net_profits) / actual_total,
        }
    return None


def evaluate_lay_lay_arb(
    lay_prices: List[Decimal],
    total_liability: Decimal,
    mbr: Decimal,
    discount: Decimal,
) -> Optional[dict]:
    """
    Evaluate a lay-lay arb: lay all outcomes with equal-profit stakes.
    Lay overround = sum(1/L_i) for all lay prices. Must be > 1.0 for arb.

    In lay markets, if we lay outcome i with stake s_i:
      - If outcome i LOSES: we win s_i
      - If outcome i WINS: we lose s_i * (L_i - 1)

    When outcome i occurs (wins), total profit is:
      profit_i = sum(s_j for j ≠ i) - s_i * (L_i - 1)
              = sum(all stakes) - s_i - s_i * (L_i - 1)
              = sum(all stakes) - s_i * L_i

    For equal-profit stakes across all outcomes:
      profit_1 = profit_2 = ... = profit_n
      => s_1 * L_1 = s_2 * L_2 = ... = s_n * L_n = k (constant)

    With total stake constraint: sum(s_i) = total_stake
      s_i = k / L_i
      sum(k / L_i) = total_stake
      k = total_stake / sum(1/L_i) = total_stake / lay_overround

    Profit for each outcome: P = total_stake - k = total_stake * (1 - 1/lay_overround)

    Returns dict with lay_overround, stakes, liabilities, total_collected,
    net_profits, min_net_profit, roi if profitable; None otherwise.
    All amounts quantized to 2 dp.
    """
    if not lay_prices or total_liability <= 0:
        return None

    # Lay overround: sum of reciprocals
    lay_overround = sum(Decimal("1") / l for l in lay_prices)

    # If lay_overround <= 1.0, no arb possible (no profit scenario exists)
    if lay_overround <= Decimal("1"):
        return None

    # For equal-profit lay stakes:
    # k = total_liability / lay_overround (constant profit per unit)
    # stake_i = k / L_i
    k = total_liability / lay_overround

    stakes = [
        (k / l).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        for l in lay_prices
    ]

    # liability_k = stake_k * (L_k - 1)
    liabilities = [
        (s * (l - Decimal("1"))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        for s, l in zip(stakes, lay_prices)
    ]

    # Total collected from all lay stakes
    total_collected = sum(stakes)

    # Gross profit if outcome i wins: total_collected - stake_i * L_i
    profits = [total_collected - s * l for s, l in zip(stakes, lay_prices)]

    # Apply commission only on positive winnings
    net_profits = [
        g - commission(g, mbr, discount) if g > 0 else g
        for g in profits
    ]

    # All net profits must be positive for a valid arb
    if all(n > 0 for n in net_profits):
        return {
            "lay_overround": lay_overround,
            "stakes": stakes,
            "liabilities": liabilities,
            "total_collected": total_collected,
            "net_profits": net_profits,
            "min_net_profit": min(net_profits),
            "roi": min(net_profits) / total_collected,
        }
    return None


def evaluate_mo_dnb_3leg_arb(
    mo_back_price: Decimal,
    dnb_lay_price: Decimal,
    draw_back_price: Decimal,
    max_stake: Decimal,
    mbr: Decimal,
    discount: Decimal,
) -> Optional[dict]:
    """
    Evaluate a 3-leg cross-market arb: back team X on MATCH_ODDS,
    lay team X on DRAW_NO_BET, back Draw on MATCH_ODDS.

    This hedges the draw outcome that the 2-leg model ignores. On a draw,
    DNB lay is voided (stake refunded) and the MO back loses — without the
    draw hedge this is an unhedged loss.

    Stake sizing (equal gross profit across all 3 scenarios):
      b = max_stake                (back team X on MO)
      l = b * B / L                (lay team X on DNB)
      d = b * B / (L * D)          (back Draw on MO)

    All 3 scenarios give the same gross profit: l - b - d.

    Commission applies per market on net positive winnings:
      MO market P&L:  scenario 1 = b*(B-1) - d,  scenario 2 = -(b+d),  scenario 3 = d*(D-1) - b
      DNB market P&L: scenario 1 = -l*(L-1),      scenario 2 = +l,       scenario 3 = 0 (void)

    Pre-commission profitability requires B*(D-1)/D > L (rarely true, correctly).

    Returns dict with stakes, per-scenario profits, net_profit, roi — or None.
    """
    B, L, D = mo_back_price, dnb_lay_price, draw_back_price

    # Validate inputs
    if any(v <= Decimal("0") for v in (B, L, D)):
        return None
    if max_stake <= Decimal("0"):
        return None

    # Pre-commission profitability check: B*(D-1)/D > L
    if B * (D - Decimal("1")) / D <= L:
        return None

    # Stake sizing
    b = max_stake
    l = (b * B / L).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    d = (b * B / (L * D)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    if l <= Decimal("0") or d <= Decimal("0"):
        return None

    # Gross profit is the same in all 3 scenarios (by construction)
    gross = l - b - d

    if gross <= Decimal("0"):
        return None

    # Per-market P&L for commission calculation
    # MO market (two back bets: team X at b, Draw at d)
    mo_pnl_1 = b * (B - Decimal("1")) - d       # X wins
    mo_pnl_2 = -(b + d)                          # X loses (other team wins)
    mo_pnl_3 = d * (D - Decimal("1")) - b        # Draw

    # DNB market (lay team X at l)
    dnb_pnl_1 = -(l * (L - Decimal("1")))        # X wins (lay loses)
    dnb_pnl_2 = l                                 # X loses (lay wins)
    # dnb_pnl_3 = 0                              # Draw (void)

    # Commission per scenario (only on positive market P&L)
    _zero = Decimal("0")
    comm_1 = (commission(mo_pnl_1, mbr, discount) if mo_pnl_1 > _zero else _zero)
    # DNB is negative in scenario 1, no commission
    comm_2 = (commission(dnb_pnl_2, mbr, discount) if dnb_pnl_2 > _zero else _zero)
    # MO is negative in scenario 2, no commission
    comm_3 = (commission(mo_pnl_3, mbr, discount) if mo_pnl_3 > _zero else _zero)
    # DNB is zero in scenario 3, no commission

    # Net profit per scenario
    net_1 = gross - comm_1
    net_2 = gross - comm_2
    net_3 = gross - comm_3

    min_net = min(net_1, net_2, net_3)

    if min_net <= Decimal("0"):
        return None

    lay_liability = (l * (L - Decimal("1"))).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    total_outlay = b + d + lay_liability
    roi = min_net / total_outlay if total_outlay > _zero else _zero

    return {
        "back_stake": b,
        "lay_stake": l,
        "draw_stake": d,
        "lay_liability": lay_liability,
        "total_outlay": total_outlay,
        "gross_profit": gross,
        "net_profit_scenarios": {
            "x_wins": net_1,
            "x_loses": net_2,
            "draw": net_3,
        },
        "commission_scenarios": {
            "x_wins": comm_1,
            "x_loses": comm_2,
            "draw": comm_3,
        },
        "net_profit": min_net,
        "roi": roi,
    }


def evaluate_back_lay_arb(
    back_price: Decimal,
    lay_price: Decimal,
    max_stake: Decimal,
    mbr: Decimal,
    discount: Decimal,
) -> Optional[dict]:
    """
    Evaluate a back-lay arb on the SAME selection across two markets.

    Strategy: back at price B in one market, lay at price L in another.
    For arb to exist, B > L (back price exceeds lay price).

    Stake sizing: equalize payouts so profit is guaranteed regardless of outcome.
      - back_stake is the stake placed on the back bet
      - lay_stake = (back_stake * B) / L  (equalizes total payout if selection wins)

    Profit scenarios:
      - Selection WINS:  back wins B*back_stake, lay loses lay_stake*(L-1)
        gross = back_stake*(B-1) - lay_stake*(L-1)
      - Selection LOSES: back loses back_stake, lay wins lay_stake
        gross = lay_stake - back_stake

    Commission applies separately to each market's net winnings (both are Betfair).
      - If selection wins: commission on back market winnings = back_stake*(B-1)
      - If selection loses: commission on lay market winnings = lay_stake

    max_stake is used as the back_stake. Returns dict with stakes, profits,
    commission, net_profit, roi — or None if unprofitable.
    """
    if back_price <= Decimal("0") or lay_price <= Decimal("0"):
        return None
    if max_stake <= Decimal("0"):
        return None

    # Arb requires back price > lay price
    if back_price <= lay_price:
        return None

    back_stake = max_stake
    # lay_stake such that total payout is equalized
    lay_stake = (back_stake * back_price / lay_price).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )

    if lay_stake <= Decimal("0"):
        return None

    # Scenario 1: selection WINS
    back_winnings = back_stake * (back_price - Decimal("1"))  # gross profit on back market
    lay_liability = lay_stake * (lay_price - Decimal("1"))     # loss on lay market
    gross_win = back_winnings - lay_liability
    # Commission on back market winnings (winner pays commission)
    comm_back_win = commission(back_winnings, mbr, discount) if back_winnings > 0 else Decimal("0")
    net_win = gross_win - comm_back_win

    # Scenario 2: selection LOSES
    gross_lose = lay_stake - back_stake
    # Commission on lay market winnings (lay stake collected)
    comm_lay_win = commission(lay_stake, mbr, discount) if lay_stake > 0 else Decimal("0")
    net_lose = gross_lose - comm_lay_win

    # Both scenarios must be profitable
    min_net = min(net_win, net_lose)
    if min_net <= Decimal("0"):
        return None

    total_outlay = back_stake + lay_stake * (lay_price - Decimal("1"))  # back stake + lay liability
    roi = min_net / total_outlay if total_outlay > 0 else Decimal("0")

    return {
        "back_stake": back_stake,
        "lay_stake": lay_stake,
        "lay_liability": lay_stake * (lay_price - Decimal("1")),
        "gross_win": gross_win,
        "gross_lose": gross_lose,
        "commission_win": comm_back_win,
        "commission_lose": comm_lay_win,
        "net_win": net_win,
        "net_lose": net_lose,
        "net_profit": min_net,
        "roi": roi,
    }
