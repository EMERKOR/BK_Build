"""
Betting Utilities - Professional Evaluation Framework

Implements industry-standard betting calculations:
- Vig removal (de-vigging)
- Expected Value (EV)
- Closing Line Value (CLV)
- Kelly Criterion bet sizing

Based on research from Unabated, Pinnacle, and professional betting literature.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


# ============================================================================
# AMERICAN ODDS CONVERSION
# ============================================================================

def american_to_implied_prob(odds: float) -> float:
    """
    Convert American odds to implied probability.

    Args:
        odds: American odds (negative for favorites, positive for underdogs)

    Returns:
        Implied probability as decimal (0 to 1)

    Examples:
        >>> american_to_implied_prob(-110)  # Standard line
        0.5238
        >>> american_to_implied_prob(+150)  # Underdog
        0.4
    """
    if odds < 0:
        # Favorite: IP = (-Odds) / ((-Odds) + 100)
        return (-odds) / ((-odds) + 100)
    else:
        # Underdog: IP = 100 / (Odds + 100)
        return 100 / (odds + 100)


def implied_prob_to_american(prob: float) -> float:
    """
    Convert implied probability to American odds.

    Args:
        prob: Implied probability (0 to 1)

    Returns:
        American odds
    """
    if prob >= 0.5:
        # Favorite (negative odds)
        return -100 * prob / (1 - prob)
    else:
        # Underdog (positive odds)
        return 100 * (1 - prob) / prob


# ============================================================================
# VIG REMOVAL (DE-VIGGING)
# ============================================================================

def devig_multiplicative(home_odds: float, away_odds: float) -> Tuple[float, float]:
    """
    Remove vig using multiplicative method (most common, conservative).

    The sportsbook's overround (vig) inflates both probabilities above 100%.
    This normalizes them to sum to 100%.

    Args:
        home_odds: American odds for home team
        away_odds: American odds for away team

    Returns:
        Tuple of (home_true_prob, away_true_prob) summing to 1.0

    Example:
        >>> devig_multiplicative(-110, -110)  # Standard line
        (0.5, 0.5)
        >>> devig_multiplicative(-150, +130)  # Favorite vs underdog
        (0.603, 0.397)
    """
    # Convert to implied probabilities
    home_ip = american_to_implied_prob(home_odds)
    away_ip = american_to_implied_prob(away_odds)

    # Calculate overround (total > 1.0)
    total_ip = home_ip + away_ip

    # Normalize to remove vig
    home_true = home_ip / total_ip
    away_true = away_ip / total_ip

    return home_true, away_true


def devig_power(home_odds: float, away_odds: float, k: float = 2.0) -> Tuple[float, float]:
    """
    Remove vig using power method (more sophisticated, accounts for favorite-longshot bias).

    This method is more accurate when there's a significant favorite, as it adjusts
    for the tendency of sportsbooks to overcharge on longshots.

    Args:
        home_odds: American odds for home team
        away_odds: American odds for away team
        k: Power parameter (2.0 is standard, higher k = more aggressive de-vigging)

    Returns:
        Tuple of (home_true_prob, away_true_prob)
    """
    home_ip = american_to_implied_prob(home_odds)
    away_ip = american_to_implied_prob(away_odds)

    # Apply power transformation
    home_power = home_ip ** k
    away_power = away_ip ** k

    # Normalize
    total_power = home_power + away_power
    home_true = home_power / total_power
    away_true = away_power / total_power

    return home_true, away_true


# ============================================================================
# EXPECTED VALUE (EV)
# ============================================================================

def calculate_ev(model_prob: float, market_prob: float, odds: float, stake: float = 100) -> float:
    """
    Calculate Expected Value for a bet.

    EV is the long-term average profit/loss per bet. Positive EV means profitable.

    Args:
        model_prob: Your model's true probability of winning (0 to 1)
        market_prob: Vig-free market probability (from de-vigging)
        odds: American odds being offered
        stake: Bet size (default $100)

    Returns:
        Expected value in dollars

    Example:
        >>> calculate_ev(0.55, 0.50, -110, stake=100)
        5.5  # Expect to profit $5.50 per $100 bet
    """
    # Calculate payout for a win
    if odds < 0:
        payout = stake * (100 / (-odds))
    else:
        payout = stake * (odds / 100)

    # EV = (p_win × payout) - (p_lose × stake)
    ev = (model_prob * payout) - ((1 - model_prob) * stake)

    return ev


def calculate_ev_percentage(model_prob: float, market_prob: float) -> float:
    """
    Calculate EV as a percentage edge over the market.

    This is a cleaner metric than dollar EV for comparing bets.

    Args:
        model_prob: Your model's probability
        market_prob: Vig-free market probability

    Returns:
        EV as percentage (e.g., 0.05 = 5% edge)
    """
    return (model_prob - market_prob) / market_prob


# ============================================================================
# CLOSING LINE VALUE (CLV)
# ============================================================================

def calculate_clv(bet_line: float, closing_line: float, is_home: bool = True) -> float:
    """
    Calculate Closing Line Value - the gold standard betting metric.

    CLV measures how much better your line was compared to the closing line.
    Positive CLV indicates a +EV bet, even if it loses.

    Args:
        bet_line: The line you bet at (e.g., -3.5)
        closing_line: The final line before kickoff (e.g., -4.5)
        is_home: Whether betting on home team (affects sign interpretation)

    Returns:
        CLV in points (positive = you got a better line)

    Example:
        >>> calculate_clv(-3.5, -4.5, is_home=True)
        1.0  # You got 1 point of CLV (bet at -3.5, closed at -4.5)
    """
    if is_home:
        # For home team: more negative is worse
        # CLV = closing_line - bet_line
        # If you bet at -3.5 and it closes at -4.5, you got +1 CLV
        return closing_line - bet_line
    else:
        # For away team: more positive is better
        return bet_line - closing_line


def calculate_clv_percentage(bet_odds: float, closing_odds: float) -> float:
    """
    Calculate CLV as a percentage change in implied probability.

    Args:
        bet_odds: American odds you bet at
        closing_odds: American odds at close

    Returns:
        CLV as percentage change
    """
    bet_prob = american_to_implied_prob(bet_odds)
    close_prob = american_to_implied_prob(closing_odds)

    return (close_prob - bet_prob) / bet_prob


# ============================================================================
# KELLY CRITERION BET SIZING
# ============================================================================

def kelly_criterion(model_prob: float, odds: float) -> float:
    """
    Calculate optimal bet size using Kelly Criterion.

    The Kelly Criterion maximizes long-term bankroll growth.

    Formula: f* = (bp - q) / b
    Where:
        - b = decimal odds - 1 (net odds received)
        - p = probability of winning
        - q = probability of losing (1 - p)

    Args:
        model_prob: Your model's probability of winning
        odds: American odds being offered

    Returns:
        Fraction of bankroll to bet (e.g., 0.05 = 5%)

    WARNING: Full Kelly is aggressive. Use fractional Kelly in practice.

    Example:
        >>> kelly_criterion(0.55, -110)
        0.055  # Bet 5.5% of bankroll
    """
    # Convert American odds to decimal
    if odds < 0:
        decimal_odds = 1 + (100 / (-odds))
    else:
        decimal_odds = 1 + (odds / 100)

    b = decimal_odds - 1  # Net odds
    p = model_prob
    q = 1 - p

    # Kelly formula
    kelly = (b * p - q) / b

    # Never bet negative Kelly
    return max(0, kelly)


def fractional_kelly(model_prob: float, odds: float, fraction: float = 0.25) -> float:
    """
    Calculate fractional Kelly - the professional standard.

    Fractional Kelly reduces variance and accounts for model uncertainty.

    Common fractions:
        - 1/4 Kelly (0.25): Very conservative, recommended for most
        - 1/2 Kelly (0.50): Moderate, still reduces variance significantly
        - Full Kelly (1.00): Aggressive, NOT recommended

    Args:
        model_prob: Your model's probability
        odds: American odds
        fraction: Fraction of Kelly to use (default 0.25 = Quarter Kelly)

    Returns:
        Fraction of bankroll to bet
    """
    full_kelly = kelly_criterion(model_prob, odds)
    return full_kelly * fraction


# ============================================================================
# PROBABILITY CALIBRATION
# ============================================================================

def spread_to_win_probability(spread_prediction: float, residual_std: float = 2.0) -> float:
    """
    Convert a spread prediction to a win probability using residual distribution.

    Assumes residuals are normally distributed around the prediction.

    Args:
        spread_prediction: Predicted spread (home team perspective, negative = home favored)
        residual_std: Standard deviation of model residuals (from validation)

    Returns:
        Probability home team wins (0 to 1)

    Example:
        >>> spread_to_win_probability(-7.0, residual_std=2.0)
        0.9998  # ~100% chance to win when favored by 7
        >>> spread_to_win_probability(3.0, residual_std=2.0)
        0.067  # ~7% chance to win when underdog by 3
    """
    from scipy.stats import norm

    # Spread < 0 means home team favored
    # P(home wins) = P(actual margin > 0) = P(actual > 0 | prediction, std)
    # Using normal distribution: P(X > 0) where X ~ N(spread_prediction, residual_std)

    # Standardize: z = (0 - spread_prediction) / residual_std
    z = -spread_prediction / residual_std

    # P(Z > z) for standard normal
    win_prob = norm.cdf(z)

    return win_prob


# ============================================================================
# AGAINST THE SPREAD (ATS) BETTING METRICS
# ============================================================================

def calculate_ats_outcome(actual_margin: float, spread_line: float) -> bool:
    """
    Calculate whether a bet against the spread won.

    ATS betting rules (home team perspective):
    - Home team covers if: actual_margin + spread_line > 0
    - If spread is -3.5, home team must win by 4+ to cover
    - If spread is +3.5, home team can lose by up to 3 and still cover

    Args:
        actual_margin: Actual game margin (home_score - away_score)
        spread_line: Spread line from home team perspective (negative = home favored)

    Returns:
        True if home team covered the spread, False otherwise

    Example:
        >>> calculate_ats_outcome(7, -3.5)  # Home won by 7, was -3.5 favorite
        True  # Covered (won by more than 3.5)
        >>> calculate_ats_outcome(3, -3.5)  # Home won by 3, was -3.5 favorite
        False  # Did not cover (won by less than 3.5)
        >>> calculate_ats_outcome(-2, +3.5)  # Home lost by 2, was +3.5 underdog
        True  # Covered (lost by less than 3.5)
    """
    return (actual_margin + spread_line) > 0


def calculate_units_won(
    ats_outcomes: pd.Series,
    stakes: float = 1.0,
    juice: int = -110
) -> float:
    """
    Calculate total units won/lost given ATS outcomes.

    Standard juice is -110 (risk $110 to win $100).

    Args:
        ats_outcomes: Boolean series of whether each bet covered
        stakes: Units risked per bet (default 1.0)
        juice: American odds for bets (default -110)

    Returns:
        Total units won (negative = loss)

    Example:
        >>> outcomes = pd.Series([True, True, False, True])  # 3-1 record
        >>> calculate_units_won(outcomes, stakes=1.0, juice=-110)
        1.73  # Won 3 * 0.909 = 2.73, lost 1 * 1.0 = 1.0, net = 1.73
    """
    wins = ats_outcomes.sum()
    losses = (~ats_outcomes).sum()

    # Calculate payout per unit won at given juice
    if juice < 0:
        payout_per_unit = 100 / (-juice)
    else:
        payout_per_unit = juice / 100

    units_won = (wins * payout_per_unit * stakes) - (losses * stakes)

    return units_won


def calculate_roi(units_won: float, units_risked: float) -> float:
    """
    Calculate Return on Investment as a percentage.

    ROI = (units_won / units_risked) * 100

    Args:
        units_won: Total units won/lost (from calculate_units_won)
        units_risked: Total units risked across all bets

    Returns:
        ROI as percentage (e.g., 5.0 = 5% ROI)

    Example:
        >>> calculate_roi(1.73, 4.0)  # Won 1.73 units on 4 bets
        43.25  # 43.25% ROI
    """
    if units_risked == 0:
        return 0.0

    return (units_won / units_risked) * 100


def calculate_ats_win_rate(ats_outcomes: pd.Series) -> float:
    """
    Calculate against-the-spread win rate.

    Args:
        ats_outcomes: Boolean series of whether each bet covered

    Returns:
        Win rate as decimal (e.g., 0.55 = 55%)

    Example:
        >>> outcomes = pd.Series([True, True, False, True])
        >>> calculate_ats_win_rate(outcomes)
        0.75  # 3 wins out of 4 bets
    """
    return ats_outcomes.mean()


def ats_summary_stats(
    actual_margins: pd.Series,
    spread_lines: pd.Series,
    stakes: float = 1.0,
    juice: int = -110
) -> dict:
    """
    Calculate comprehensive ATS betting statistics.

    Args:
        actual_margins: Series of actual game margins (home perspective)
        spread_lines: Series of spread lines (home perspective)
        stakes: Units risked per bet
        juice: American odds

    Returns:
        Dictionary with:
            - n_bets: Number of bets
            - wins: Number of covers
            - losses: Number of non-covers
            - ats_win_rate: Win rate as decimal
            - units_won: Total units won
            - units_risked: Total units risked
            - roi_pct: ROI as percentage
            - break_even_rate: Required win rate to break even

    Example:
        >>> margins = pd.Series([7, 3, -2, 10])
        >>> spreads = pd.Series([-3.5, -3.5, +3.5, -7.5])
        >>> stats = ats_summary_stats(margins, spreads)
        >>> stats['ats_win_rate']
        0.75
    """
    # Calculate outcomes
    ats_outcomes = pd.Series([
        calculate_ats_outcome(margin, spread)
        for margin, spread in zip(actual_margins, spread_lines)
    ])

    n_bets = len(ats_outcomes)
    wins = ats_outcomes.sum()
    losses = n_bets - wins

    ats_win_rate = calculate_ats_win_rate(ats_outcomes)
    units_won = calculate_units_won(ats_outcomes, stakes, juice)
    units_risked = n_bets * stakes
    roi_pct = calculate_roi(units_won, units_risked)
    break_even = break_even_win_rate(juice)

    return {
        'n_bets': n_bets,
        'wins': wins,
        'losses': losses,
        'ats_win_rate': ats_win_rate,
        'units_won': units_won,
        'units_risked': units_risked,
        'roi_pct': roi_pct,
        'break_even_rate': break_even,
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def break_even_win_rate(odds: float) -> float:
    """
    Calculate break-even win rate needed for profitability.

    Args:
        odds: American odds

    Returns:
        Win rate needed to break even (0 to 1)

    Example:
        >>> break_even_win_rate(-110)
        0.5238  # Need 52.38% to break even at -110
    """
    return american_to_implied_prob(odds)


def vig_percentage(home_odds: float, away_odds: float) -> float:
    """
    Calculate the vig (overround) as a percentage.

    Args:
        home_odds: American odds for home
        away_odds: American odds for away

    Returns:
        Vig as percentage (e.g., 0.0476 = 4.76% vig)
    """
    home_ip = american_to_implied_prob(home_odds)
    away_ip = american_to_implied_prob(away_odds)

    return (home_ip + away_ip) - 1.0
