"""
Betting markets: probabilities for every selection from one score distribution (so they are coherent — e.g.
P(over 1.5) ≥ P(over 2.5), double chance = sum of its 1X2 legs), and settlement of a selection from a final score.

Column names are "{market}:{selection}", e.g. "1x2:home", "ou_2.5:over", "btts:yes", "double_chance:home_draw".
Settlement here is for played matches; postponed / abandoned matches are voided by the settlement job.
"""
import numpy as np
import pandas as pd

LINES = (1.5, 2.5, 3.5)
MARKETS = {"1x2": ("home", "draw", "away"),
           "double_chance": ("home_draw", "home_away", "draw_away"),
           **{f"ou_{x}": ("over", "under") for x in LINES},
           "btts": ("yes", "no")}
EXCLUSIVE = [m for m in MARKETS if m != "double_chance"]     # exactly one selection wins


def columns(market: str) -> list[str]:
    return [f"{market}:{s}" for s in MARKETS[market]]


def from_grid(grid: np.ndarray) -> pd.DataFrame:
    """(n, G, G) exact-score probabilities (home goals on axis 1) → probability of every selection."""
    g = np.arange(grid.shape[1])
    h, a = g[:, None], g[None, :]

    def mass(mask):
        return grid[:, mask].sum(1)

    home, draw, away = mass(h > a), mass(h == a), mass(h < a)
    out = {"1x2:home": home, "1x2:draw": draw, "1x2:away": away,
           "double_chance:home_draw": home + draw, "double_chance:home_away": home + away,
           "double_chance:draw_away": draw + away}
    for x in LINES:
        over = mass(h + a > x)
        out[f"ou_{x}:over"], out[f"ou_{x}:under"] = over, 1.0 - over
    yes = mass((h > 0) & (a > 0))
    out["btts:yes"], out["btts:no"] = yes, 1.0 - yes
    return pd.DataFrame(out)


def from_1x2(p: np.ndarray) -> pd.DataFrame:
    """1X2 and double chance from (n, 3) home/draw/away probabilities (models without a score distribution)."""
    home, draw, away = p.T
    return pd.DataFrame({"1x2:home": home, "1x2:draw": draw, "1x2:away": away,
                         "double_chance:home_draw": home + draw, "double_chance:home_away": home + away,
                         "double_chance:draw_away": draw + away})


def won(market: str, selection: str, hg, ag) -> np.ndarray:
    """True where the selection wins given full-time goals."""
    hg, ag = np.asarray(hg), np.asarray(ag)
    if market == "1x2":
        return {"home": hg > ag, "draw": hg == ag, "away": hg < ag}[selection]
    if market == "double_chance":
        return {"home_draw": hg >= ag, "home_away": hg != ag, "draw_away": hg <= ag}[selection]
    if market.startswith("ou_"):
        over = hg + ag > float(market[3:])
        return over if selection == "over" else ~over
    if market == "btts":
        yes = (hg > 0) & (ag > 0)
        return yes if selection == "yes" else ~yes
    raise ValueError(f"unknown market {market}")


def outcome(market: str, hg, ag) -> np.ndarray:
    """Index of the winning selection for an exclusive market (order of MARKETS[market])."""
    wins = np.stack([won(market, s, hg, ag) for s in MARKETS[market]], axis=1)
    return wins.argmax(1)
