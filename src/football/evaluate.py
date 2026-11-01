"""
Scoring for match-outcome probabilities, and a flat-stake value-betting simulator.

Outcome columns are ordered (home, draw, away) for 1X2 and (over, under) for goals; y is the index of what happened.
"""
import numpy as np
import pandas as pd


def match_log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))


def rps(p: np.ndarray, y: np.ndarray) -> float:
    """Ranked probability score (ordered outcomes, lower is better); equals the Brier score for two outcomes."""
    k = p.shape[1]
    cum = np.cumsum(p, 1) - np.cumsum(np.eye(k)[y], 1)
    return float(np.mean((cum[:, :-1] ** 2).sum(1) / (k - 1)))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Multi-class Brier score (sum over outcomes); for two outcomes the usual (p − outcome)² of the first one."""
    sq = ((p - np.eye(p.shape[1])[y]) ** 2).sum(1)
    return float(np.mean(sq / 2 if p.shape[1] == 2 else sq))


def scores(p: np.ndarray, y: np.ndarray) -> dict:
    return {"matches": len(y), "log_loss": float(match_log_loss(p, y).mean()), "brier": brier(p, y),
            "rps": rps(p, y), "accuracy": float(np.mean(p.argmax(1) == y))}


def calibration(p: np.ndarray, won: np.ndarray, edges=np.linspace(0, 1, 11)) -> pd.DataFrame:
    """Predicted vs observed win rate per probability bin, pooling every selection given (flattened)."""
    p, won = np.ravel(p), np.ravel(won).astype(float)
    ok = np.isfinite(p)
    b = np.clip(np.digitize(p[ok], edges[1:-1]), 0, len(edges) - 2)
    t = pd.DataFrame({"bin": b, "p": p[ok], "won": won[ok]}).groupby("bin").agg(
        n=("p", "size"), predicted=("p", "mean"), observed=("won", "mean"))
    t.index = [f"{edges[i]:.0%}–{edges[i + 1]:.0%}" for i in t.index]
    return t


def value_bets(p: np.ndarray, o: np.ndarray, y: np.ndarray, edge: float = 0.0,
               commission: float = 0.0) -> pd.DataFrame:
    """One flat 1-unit bet per match on the outcome with the highest expected return, placed when that expected
    return (after commission on winnings) is above `edge`. Matches with a missing probability or price are skipped.
    Columns: row (position in the inputs), pick, odds, ev, won, profit."""
    payout = 1.0 + (o - 1.0) * (1.0 - commission)
    ev = p * payout - 1.0
    ok = np.isfinite(ev).all(1)
    pick = np.where(ok, np.where(ok[:, None], ev, -np.inf).argmax(1), 0)
    r = np.arange(len(y))
    best = np.where(ok, ev[r, pick], -np.inf)
    bet = best > edge
    won = pick == y
    profit = np.where(won, payout[r, pick] - 1.0, -1.0)
    return pd.DataFrame({"row": r[bet], "pick": pick[bet], "odds": o[r, pick][bet], "ev": best[bet],
                         "won": won[bet], "profit": profit[bet]})


def mean_se(x) -> tuple[float, float]:
    x = np.asarray(x, float)
    return (float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 1 else (np.nan, np.nan)


def bet_summary(bets: pd.DataFrame) -> dict:
    roi, se = mean_se(bets.profit)
    return {"bets": len(bets), "hit_rate": float(bets.won.mean()) if len(bets) else np.nan,
            "avg_odds": float(bets.odds.mean()) if len(bets) else np.nan, "roi": roi, "se": se,
            "t": roi / se if se else np.nan}
