"""Kronos Agent — Technical Analysis advisor.

Receives raw OHLCV candles directly from MCP and produces a structured
technical context (candlestick patterns, EMA/SMA trend, RSI, support/resistance,
CPR levels) that the AnalysisAgent uses as pre-computed context before calling
the LLM. This decouples indicator computation from LLM reasoning.

Flow:
    Orchestrator / AnalysisAgent
        → KronosAgent.handle_task(symbol, source, interval, candles?)
        → fetches candles if not provided
        → computes indicators purely in Python (no LLM)
        → returns kronos_context artifact
        → AnalysisAgent feeds this into its LLM prompt
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from backend.mcp.connector import call_mcp_tool
from backend.protocol import Artifact, Task, TaskResponse

logger = logging.getLogger(__name__)

# ── Indicator helpers ──────────────────────────────────────────────────────────

def _ema(values: list[float], period: int) -> list[float | None]:
    """Exponential moving average."""
    result: list[float | None] = [None] * len(values)
    if len(values) < period:
        return result
    k = 2 / (period + 1)
    # seed with SMA of first `period` values
    seed = sum(values[:period]) / period
    result[period - 1] = seed
    for i in range(period, len(values)):
        result[i] = values[i] * k + result[i - 1] * (1 - k)  # type: ignore[operator]
    return result


def _sma(values: list[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    for i in range(period - 1, len(values)):
        result[i] = sum(values[i - period + 1 : i + 1]) / period
    return result


def _rsi(closes: list[float], period: int = 14) -> float | None:
    """RSI of the last `period+1` closes."""
    if len(closes) < period + 1:
        return None
    tail = closes[-(period + 1):]
    gains, losses = [], []
    for i in range(1, len(tail)):
        diff = tail[i] - tail[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def _atr(candles: list[dict], period: int = 14) -> float | None:
    """Average True Range."""
    if len(candles) < period + 1:
        return None
    trs = []
    for i in range(1, len(candles)):
        h = float(candles[i].get("high", 0))
        l = float(candles[i].get("low", 0))
        pc = float(candles[i - 1].get("close", 0))
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return round(sum(trs[-period:]) / period, 4) if trs else None


def _pivot_cpr(candles: list[dict]) -> dict[str, float]:
    """Central Pivot Range from the previous candle."""
    if not candles:
        return {}
    prev = candles[-1]
    h = float(prev.get("high", 0))
    l = float(prev.get("low", 0))
    c = float(prev.get("close", 0))
    pivot = (h + l + c) / 3
    bc    = (h + l) / 2        # bottom central pivot
    tc    = (pivot - bc) + pivot  # top central pivot
    r1    = 2 * pivot - l
    s1    = 2 * pivot - h
    r2    = pivot + (h - l)
    s2    = pivot - (h - l)
    return {
        "pivot": round(pivot, 4),
        "bc":    round(bc, 4),
        "tc":    round(tc, 4),
        "r1":    round(r1, 4), "r2": round(r2, 4),
        "s1":    round(s1, 4), "s2": round(s2, 4),
    }


def _support_resistance(candles: list[dict], lookback: int = 20) -> dict[str, list[float]]:
    """Simple swing-high / swing-low support and resistance zones."""
    tail = candles[-lookback:] if len(candles) >= lookback else candles
    highs  = sorted({round(float(c.get("high", 0)), 2) for c in tail}, reverse=True)
    lows   = sorted({round(float(c.get("low",  0)), 2) for c in tail})
    last_close = float(candles[-1].get("close", 0)) if candles else 0
    resistance = [h for h in highs if h > last_close][:3]
    support    = [l for l in lows  if l < last_close][:3]
    return {"resistance": resistance, "support": support}


# ── Candlestick patterns (single + two-candle) ─────────────────────────────────

def _classify_candles(candles: list[dict]) -> list[str]:
    """Return list of detected pattern names from recent candles."""
    patterns: list[str] = []
    if not candles:
        return patterns

    def body(c: dict) -> float:
        return abs(float(c.get("close", 0)) - float(c.get("open", 0)))

    def upper_wick(c: dict) -> float:
        return float(c.get("high", 0)) - max(float(c.get("close", 0)), float(c.get("open", 0)))

    def lower_wick(c: dict) -> float:
        return min(float(c.get("close", 0)), float(c.get("open", 0))) - float(c.get("low", 0))

    def candle_range(c: dict) -> float:
        return float(c.get("high", 0)) - float(c.get("low", 0))

    # Single-candle patterns on the last candle
    c = candles[-1]
    b, uw, lw, rng = body(c), upper_wick(c), lower_wick(c), candle_range(c)
    if rng == 0:
        rng = 1e-9
    bullish = float(c.get("close", 0)) > float(c.get("open", 0))

    # Doji
    if b / rng < 0.1:
        patterns.append("doji")
    # Hammer / Hanging Man
    if lw >= 2 * b and uw < b * 0.5:
        patterns.append("hammer" if bullish else "hanging_man")
    # Inverted Hammer / Shooting Star
    if uw >= 2 * b and lw < b * 0.5:
        patterns.append("inverted_hammer" if bullish else "shooting_star")
    # Marubozu
    if b / rng > 0.9:
        patterns.append("bullish_marubozu" if bullish else "bearish_marubozu")
    # Spinning Top
    if 0.1 <= b / rng <= 0.4 and uw > b and lw > b:
        patterns.append("spinning_top")

    # Two-candle patterns
    if len(candles) >= 2:
        p, cu = candles[-2], candles[-1]
        p_bull = float(p.get("close", 0)) > float(p.get("open", 0))
        c_bull = float(cu.get("close", 0)) > float(cu.get("open", 0))
        p_body = body(p)
        c_body = body(cu)

        # Engulfing
        if not p_bull and c_bull and c_body > p_body:
            if float(cu.get("open", 0)) <= float(p.get("close", 0)) and float(cu.get("close", 0)) >= float(p.get("open", 0)):
                patterns.append("bullish_engulfing")
        if p_bull and not c_bull and c_body > p_body:
            if float(cu.get("open", 0)) >= float(p.get("close", 0)) and float(cu.get("close", 0)) <= float(p.get("open", 0)):
                patterns.append("bearish_engulfing")

        # Harami
        if p_bull and not c_bull and c_body < p_body:
            patterns.append("bearish_harami")
        if not p_bull and c_bull and c_body < p_body:
            patterns.append("bullish_harami")

    # Three-candle patterns
    if len(candles) >= 3:
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        c1_bull = float(c1.get("close", 0)) > float(c1.get("open", 0))
        c3_bull = float(c3.get("close", 0)) > float(c3.get("open", 0))

        # Morning Star
        if not c1_bull and body(c2) < body(c1) * 0.3 and c3_bull and body(c3) > body(c1) * 0.5:
            patterns.append("morning_star")
        # Evening Star
        if c1_bull and body(c2) < body(c1) * 0.3 and not c3_bull and body(c3) > body(c1) * 0.5:
            patterns.append("evening_star")
        # Three White Soldiers
        if all(float(cx.get("close", 0)) > float(cx.get("open", 0)) for cx in [c1, c2, c3]):
            patterns.append("three_white_soldiers")
        # Three Black Crows
        if all(float(cx.get("close", 0)) < float(cx.get("open", 0)) for cx in [c1, c2, c3]):
            patterns.append("three_black_crows")

    return patterns


# ── Trend classification ───────────────────────────────────────────────────────

def _trend_from_emas(ema9: list, ema21: list, ema50: list) -> str:
    """Classify trend from last EMA values."""
    vals = [(v if v is not None else 0) for v in [ema9[-1], ema21[-1], ema50[-1]]]
    e9, e21, e50 = vals
    if e9 > e21 > e50:
        return "bullish"
    if e9 < e21 < e50:
        return "bearish"
    if abs(e9 - e21) / max(e21, 1) < 0.003:
        return "sideways"
    return "bullish" if e9 > e50 else "bearish"


# ── Volume analysis ────────────────────────────────────────────────────────────

def _volume_signal(candles: list[dict], lookback: int = 10) -> str:
    if len(candles) < lookback + 1:
        return "insufficient_data"
    tail  = candles[-(lookback + 1):]
    avg_v = sum(float(c.get("volume", 0)) for c in tail[:-1]) / lookback
    last_v = float(tail[-1].get("volume", 0))
    if avg_v == 0:
        return "no_volume_data"
    ratio = last_v / avg_v
    if ratio > 1.5:
        return "high_volume"
    if ratio < 0.5:
        return "low_volume"
    return "normal_volume"


# ── KronosAgent ────────────────────────────────────────────────────────────────

class KronosAgent:
    """Pure technical analysis agent — no LLM, only math.

    Produces a structured `kronos_context` artifact consumed by AnalysisAgent.
    """

    agent_name = "kronos_agent"

    async def handle_task(self, task: Task) -> TaskResponse:
        inp      = task.input or {}
        symbol   = inp.get("symbol", "BTCUSDT")
        source   = inp.get("source", "delta")
        interval = inp.get("interval", "5m")
        candles  = inp.get("candles")  # optional — pre-fetched

        logger.info("kronos_agent START  symbol=%s  source=%s  interval=%s",
                    symbol, source, interval)

        # Fetch candles if not provided
        if not candles:
            raw = await call_mcp_tool(
                source, "fetch_candles",
                {"symbol": symbol, "interval": interval, "lookback_days": 7},
            )
            data    = json.loads(raw)
            candles = data.get("candles", [])
            if "error" in data:
                logger.warning("kronos_agent: candle fetch error — %s", data["error"])

        if not candles:
            return TaskResponse(
                task_id=task.task_id, agent=self.agent_name, status="failed",
                error=f"No candle data for {symbol} {interval}",
            )

        closes  = [float(c.get("close", 0)) for c in candles]
        last    = candles[-1]
        context = self._compute(candles, closes, symbol, interval)

        logger.info("kronos_agent END  symbol=%s  trend=%s  rsi=%.1f  patterns=%s",
                    symbol, context.get("trend"), context.get("rsi") or 0,
                    context.get("patterns"))

        return TaskResponse(
            task_id=task.task_id, agent=self.agent_name, status="completed",
            artifacts=[Artifact(type="kronos_context", data=context)],
        )

    def _compute(self, candles: list[dict], closes: list[float],
                 symbol: str, interval: str) -> dict[str, Any]:
        ema9   = _ema(closes, 9)
        ema21  = _ema(closes, 21)
        ema50  = _ema(closes, 50)
        sma200 = _sma(closes, 200)
        rsi    = _rsi(closes, 14)
        atr    = _atr(candles, 14)
        cpr    = _pivot_cpr(candles[:-1] if len(candles) > 1 else candles)
        sr     = _support_resistance(candles, 20)
        patterns = _classify_candles(candles[-5:])  # last 5 candles
        trend  = _trend_from_emas(ema9, ema21, ema50)
        vol_sig = _volume_signal(candles, 10)

        last_close = closes[-1] if closes else 0

        # EMA values (last only)
        def _last(lst: list) -> float | None:
            for v in reversed(lst):
                if v is not None:
                    return round(v, 4)
            return None

        return {
            "symbol":   symbol,
            "interval": interval,
            "candle_count": len(candles),
            "last_close":   round(last_close, 4),

            # Trend
            "trend":    trend,

            # Indicators
            "ema9":     _last(ema9),
            "ema21":    _last(ema21),
            "ema50":    _last(ema50),
            "sma200":   _last(sma200),
            "rsi":      rsi,
            "atr":      atr,

            # CPR (Central Pivot Range)
            "cpr":      cpr,

            # Support / Resistance
            "support":    sr["support"],
            "resistance": sr["resistance"],

            # Candlestick patterns
            "patterns": patterns,

            # Volume
            "volume_signal": vol_sig,

            # Composite signal
            "signal": self._composite_signal(trend, rsi, patterns, vol_sig, cpr, last_close),
        }

    @staticmethod
    def _composite_signal(
        trend: str, rsi: float | None, patterns: list[str],
        vol_sig: str, cpr: dict, last_close: float,
    ) -> dict[str, Any]:
        """Combine indicators into a single actionable bias."""
        bull_patterns = {
            "hammer", "inverted_hammer", "morning_star",
            "bullish_engulfing", "bullish_harami", "three_white_soldiers",
            "bullish_marubozu",
        }
        bear_patterns = {
            "shooting_star", "hanging_man", "evening_star",
            "bearish_engulfing", "bearish_harami", "three_black_crows",
            "bearish_marubozu",
        }
        bull_p = [p for p in patterns if p in bull_patterns]
        bear_p = [p for p in patterns if p in bear_patterns]

        # RSI zone
        rsi_zone = "neutral"
        if rsi is not None:
            if rsi < 30:
                rsi_zone = "oversold"
            elif rsi > 70:
                rsi_zone = "overbought"

        # CPR position
        cpr_position = "unknown"
        if cpr and last_close:
            if last_close > cpr.get("tc", 0):
                cpr_position = "above_cpr"
            elif last_close < cpr.get("bc", 0):
                cpr_position = "below_cpr"
            else:
                cpr_position = "inside_cpr"

        # Score
        score = 0
        if trend == "bullish":   score += 2
        if trend == "bearish":   score -= 2
        score += len(bull_p)
        score -= len(bear_p)
        if rsi_zone == "oversold":    score += 1
        if rsi_zone == "overbought":  score -= 1
        if cpr_position == "above_cpr":  score += 1
        if cpr_position == "below_cpr":  score -= 1
        if vol_sig == "high_volume":     score = int(score * 1.2)

        if score >= 3:    bias = "strong_buy"
        elif score >= 1:  bias = "buy"
        elif score <= -3: bias = "strong_sell"
        elif score <= -1: bias = "sell"
        else:             bias = "neutral"

        return {
            "bias":          bias,
            "score":         score,
            "rsi_zone":      rsi_zone,
            "cpr_position":  cpr_position,
            "bull_patterns": bull_p,
            "bear_patterns": bear_p,
        }
