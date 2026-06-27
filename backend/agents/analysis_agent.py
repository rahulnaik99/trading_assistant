"""Analysis Agent — fetches market data via MCP and produces structured analysis."""

import asyncio
import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.mcp.connector import call_mcp_tool
from backend.protocol import Artifact, Task, TaskResponse

logger = logging.getLogger(__name__)

_SYSTEM = """You are an expert financial analyst specialising in Indian equities (NSE/BSE) and crypto perpetuals (Delta Exchange).

Given market data (OHLCV candles, perpetual metrics), produce a concise structured analysis:
- Trend direction and strength
- Key support / resistance levels
- Trade type recommendation: intraday | swing | longterm
- Entry zone, stop-loss, take-profit
- Risk:Reward ratio
- Confidence score 0-1

Respond ONLY with valid JSON matching this schema:
{
  "symbol": string,
  "trade_type": "intraday|swing|longterm",
  "trend": "bullish|bearish|sideways",
  "strength": "strong|moderate|weak",
  "last_price": float,
  "entry_zone": {"low": float, "high": float},
  "stop_loss": float,
  "targets": [float, float],
  "rr_ratio": float,
  "confidence": float,
  "summary": string,
  "key_levels": {"support": [float], "resistance": [float]},
  "news_context": string
}"""

_AUTH_KEYWORDS = ("expired", "invalid", "auth", "token", "unauthorized", "403", "401")

# Use lowercase "1d" — consistent with DeltaSource.INTERVAL_MAP
_INTERVAL_MAP = {
    ("swing",    "5m"):  "1h",
    ("longterm", "5m"):  "1d",
    ("longterm", "1h"):  "1d",
}


def _resolve_interval(trade_type: str, interval: str) -> str:
    return _INTERVAL_MAP.get((trade_type, interval), interval)


def _parse_json_response(raw: str, symbol: str) -> dict[str, Any]:
    cleaned = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {"symbol": symbol, "summary": raw, "confidence": 0.5}


def _build_prompt(
    symbol: str, trade_type: str, interval: str,
    candle_data: dict, market_extra: dict, news_context: str,
    kronos: dict | None = None,
) -> str:
    candles = candle_data.get("candles") or []
    recent  = candles[-20:]
    if recent:
        highs  = [c.get("high", 0)  for c in recent]
        lows   = [c.get("low", 0)   for c in recent]
        closes = [c.get("close", 0) for c in recent]
        period_high = max(highs)
        period_low  = min(lows)
        avg_close   = sum(closes) / len(closes)
    else:
        period_high = period_low = avg_close = "N/A"

    lines = [
        f"Symbol: {symbol}  Trade type: {trade_type.upper()}  Interval: {interval}",
        f"Last close: {candle_data.get('last_close', 'N/A')}",
        f"Period range ({len(recent)} bars): High={period_high}  Low={period_low}  Avg close={avg_close:.2f}" if recent else "No candle data",
        f"Recent {len(recent)} candles (OHLCV): {json.dumps(recent)}",
    ]

    # Kronos technical context — pre-computed indicators, patterns, CPR
    if kronos:
        sig = kronos.get("signal", {})
        lines += [
            "",
            "=== KRONOS TECHNICAL CONTEXT ===",
            f"Trend (EMA9/21/50): {kronos.get('trend','?').upper()}",
            f"EMA9={kronos.get('ema9')}  EMA21={kronos.get('ema21')}  EMA50={kronos.get('ema50')}  SMA200={kronos.get('sma200')}",
            f"RSI(14): {kronos.get('rsi')}  ({sig.get('rsi_zone','?').upper()})",
            f"ATR(14): {kronos.get('atr')}",
            f"Support: {kronos.get('support')}  Resistance: {kronos.get('resistance')}",
            f"CPR: {json.dumps(kronos.get('cpr', {}))}  Position: {sig.get('cpr_position','?').upper()}",
            f"Volume signal: {kronos.get('volume_signal','?').upper()}",
            f"Candlestick patterns: {kronos.get('patterns', [])}",
            f"  Bullish: {sig.get('bull_patterns',[])}",
            f"  Bearish: {sig.get('bear_patterns',[])}",
            f"Composite bias: {sig.get('bias','?').upper()}  (score={sig.get('score',0)})",
            "=================================",
        ]

    if market_extra.get("funding_signal"):
        lines += [
            "",
            "Perpetual metrics:",
            f"  Funding: {market_extra.get('funding_rate_8h_pct','?')}%  "
            f"Signal: {market_extra.get('funding_signal','?').upper()}",
            f"  OI: {market_extra.get('oi','?')}  Mark: {market_extra.get('mark_price','?')}",
        ]
    if market_extra.get("ltp"):
        lines.append(f"\nCurrent LTP: {market_extra['ltp']}")
    lines += [
        f"\nNews context: {news_context}",
        f"\nProduce {trade_type} trade analysis in the required JSON format.",
    ]
    return "\n".join(lines)


def _get_llm(provider: str):
    from backend.llm.factory import LLMFactory
    return LLMFactory.get_llm(provider or "openai")


class AnalysisAgent:
    """Fetches data via MCP and calls LLM to produce trade analysis."""

    agent_name = "analysis_agent"

    async def handle_task(self, task: Task) -> TaskResponse:
        inp          = task.input or {}
        symbol       = inp.get("symbol", "BTCUSDT")
        trade_type   = inp.get("trade_type", "intraday")
        source       = inp.get("source", "delta")
        interval     = _resolve_interval(trade_type, inp.get("interval", "5m"))
        llm_provider = inp.get("llm_provider", "openai")

        logger.info("analysis_agent START  symbol=%s  type=%s  source=%s  interval=%s  llm=%s",
                    symbol, trade_type, source, interval, llm_provider)

        # Fetch candles, market extras, news, and Kronos TA — all in parallel
        (candle_data, auth_err), market_extra, news_context, kronos_ctx = await asyncio.gather(
            self._fetch_candles(symbol, source, interval),
            self._fetch_market_extra(symbol, source),
            self._fetch_news(symbol),
            self._fetch_kronos(symbol, source, interval),
        )

        if auth_err:
            return TaskResponse(task_id=task.task_id, agent=self.agent_name,
                                status="failed", error=auth_err)

        prompt = _build_prompt(symbol, trade_type, interval,
                               candle_data, market_extra, news_context,
                               kronos=kronos_ctx)
        logger.info("► Calling LLM  provider=%s  candles=%d  kronos=%s",
                    llm_provider, len(candle_data.get("candles") or []),
                    bool(kronos_ctx))
        llm      = _get_llm(llm_provider)
        response = await llm.ainvoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt),
        ])

        analysis = _parse_json_response(response.content.strip(), symbol)
        analysis.update({
            "symbol":        symbol,
            "trade_type":    trade_type,
            "news_context":  news_context,
            "raw_candles":   (candle_data.get("candles") or [])[-10:],
            "kronos":        kronos_ctx,  # pass through for frontend display
        })

        logger.info("analysis_agent END  symbol=%s  trend=%s  confidence=%s",
                    symbol, analysis.get("trend"), analysis.get("confidence"))

        return TaskResponse(
            task_id=task.task_id, agent=self.agent_name, status="completed",
            artifacts=[Artifact(type="analysis", data=analysis)],
        )

    async def _fetch_candles(self, symbol: str, source: str, interval: str) -> tuple[dict, str | None]:
        logger.info("► Fetching candles  source=%s  symbol=%s  interval=%s", source, symbol, interval)
        raw  = await call_mcp_tool(source, "fetch_candles",
                                   {"symbol": symbol, "interval": interval, "lookback_days": 7})
        data = json.loads(raw)
        if "error" in data:
            err = data["error"]
            if any(kw in err.lower() for kw in _AUTH_KEYWORDS):
                return {}, (
                    f"Authentication failed for {source.upper()}. "
                    f"Please refresh your access token and update .env.\nDetails: {err}"
                )
            logger.warning("analysis_agent: candle fetch warning — %s", err)
            return {"error": err, "candles": [], "last_close": None}, None
        return data, None

    async def _fetch_market_extra(self, symbol: str, source: str) -> dict:
        if source == "delta":
            logger.info("► Fetching perpetual metrics  symbol=%s", symbol)
            raw = await call_mcp_tool("delta", "fetch_perpetual_metrics", {"symbol": symbol})
        elif source == "fyers":
            logger.info("► Fetching quote  symbol=%s", symbol)
            raw = await call_mcp_tool("fyers", "get_quote", {"symbol": symbol})
        else:
            return {}
        result = json.loads(raw)
        return {} if "error" in result else result

    async def _fetch_news(self, symbol: str) -> str:
        logger.info("► Fetching news  symbol=%s", symbol)
        clean = symbol.replace("NSE:", "").replace("-EQ", "").split("/")[0]
        raw   = await call_mcp_tool("tavily", "search_news",
                                    {"query": f"{clean} stock analysis", "max_results": 3})
        data  = json.loads(raw)
        headlines = [r.get("title", "") for r in data.get("results", []) if r.get("title")]
        return " | ".join(headlines[:3]) or "No recent news"

    async def _fetch_kronos(self, symbol: str, source: str, interval: str) -> dict | None:
        """Call KronosAgent via A2A (port 8103); fall back to in-process on failure."""
        from backend.config import settings
        from backend.a2a.client import A2AClient
        from backend.protocol import Task as _Task

        input_data = {"symbol": symbol, "source": source, "interval": interval}

        # Try A2A first
        client = A2AClient(settings.KRONOS_AGENT_URL, timeout=30.0, retries=1)
        resp   = await client.send(
            agent="kronos_agent", input_data=input_data, task_id=f"k-{symbol}"
        )
        if resp.status == "completed":
            return next((a.data for a in resp.artifacts if a.type == "kronos_context"), None)

        # Fallback: in-process
        logger.warning("KronosAgent A2A unavailable — running in-process")
        try:
            from backend.agents.kronos_agent import KronosAgent
            agent = KronosAgent()
            local = await agent.handle_task(
                _Task(task_id=f"k-local-{symbol}", agent="kronos_agent", input=input_data)
            )
            if local.status == "completed":
                return next((a.data for a in local.artifacts if a.type == "kronos_context"), None)
        except Exception as exc:
            logger.warning("KronosAgent in-process fallback failed: %s", exc)
        return None
