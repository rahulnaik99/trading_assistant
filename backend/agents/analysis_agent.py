"""Analysis Agent — fetches market data via MCP and produces structured analysis."""

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.llm.factory import LLMFactory
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

_INTERVAL_MAP = {
    ("swing",    "5m"): "1h",
    ("longterm", "5m"): "1D",
    ("longterm", "1h"): "1D",
}


def _resolve_interval(trade_type: str, interval: str) -> str:
    return _INTERVAL_MAP.get((trade_type, interval), interval)


def _parse_json_response(raw: str, symbol: str) -> dict[str, Any]:
    """Parse LLM JSON response with fallback to regex extraction."""
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
) -> str:
    recent = (candle_data.get("candles") or [])[-20:]
    lines = [
        f"Symbol: {symbol}  Trade type: {trade_type.upper()}  Interval: {interval}",
        f"Last close: {candle_data.get('last_close', 'N/A')}",
        f"Recent candles (last {len(recent)}): {json.dumps(recent[:5])} ... [{len(recent)} bars total]",
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
    lines += [f"\nNews context: {news_context}",
               f"\nProduce {trade_type} trade analysis in the required JSON format."]
    return "\n".join(lines)


class AnalysisAgent:
    """Fetches data via MCP and calls LLM to produce trade analysis."""

    agent_name = "analysis_agent"

    def __init__(self, llm_provider: str = "") -> None:
        self._llm_provider = llm_provider
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = LLMFactory.get_llm(self._llm_provider)
        return self._llm

    async def handle_task(self, task: Task) -> TaskResponse:
        inp        = task.input or {}
        symbol     = inp.get("symbol", "BTCUSDT")
        trade_type = inp.get("trade_type", "intraday")
        source     = inp.get("source", "delta")
        interval   = _resolve_interval(trade_type, inp.get("interval", "5m"))

        logger.info("analysis_agent START  symbol=%s  type=%s  source=%s  interval=%s",
                    symbol, trade_type, source, interval)

        # ── Fetch market data ─────────────────────────────────────────────────
        candle_data, auth_err = await self._fetch_candles(symbol, source, interval)
        if auth_err:
            return TaskResponse(task_id=task.task_id, agent=self.agent_name,
                                status="failed", error=auth_err)

        market_extra = await self._fetch_market_extra(symbol, source)
        news_context = await self._fetch_news(symbol)

        # ── Build prompt + call LLM ───────────────────────────────────────────
        prompt = _build_prompt(symbol, trade_type, interval,
                               candle_data, market_extra, news_context)
        logger.info("► Calling LLM  provider=%s", self._llm_provider or "default")
        response = await self.llm.ainvoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt),
        ])

        analysis = _parse_json_response(response.content.strip(), symbol)
        analysis.update({
            "symbol":       symbol,
            "trade_type":   trade_type,
            "news_context": news_context,
            "raw_candles":  (candle_data.get("candles") or [])[-10:],
        })

        logger.info("analysis_agent END  symbol=%s  trend=%s  confidence=%s",
                    symbol, analysis.get("trend"), analysis.get("confidence"))

        return TaskResponse(
            task_id=task.task_id, agent=self.agent_name, status="completed",
            artifacts=[Artifact(type="analysis", data=analysis)],
        )

    # ── Data-fetch helpers ────────────────────────────────────────────────────

    async def _fetch_candles(
        self, symbol: str, source: str, interval: str
    ) -> tuple[dict, str | None]:
        """Returns (candle_data, error_message_or_None)."""
        logger.info("► Fetching candles  source=%s  symbol=%s  interval=%s",
                    source, symbol, interval)
        raw = await call_mcp_tool(
            server=source, tool="fetch_candles",
            arguments={"symbol": symbol, "interval": interval, "lookback_days": 7},
        )
        data = json.loads(raw)
        if "error" in data:
            err = data["error"]
            if any(kw in err.lower() for kw in _AUTH_KEYWORDS):
                return {}, (
                    f"Authentication failed for {source.upper()}. "
                    f"Please refresh your access token and update .env.\n"
                    f"Details: {err}"
                )
            logger.warning("analysis_agent: candle fetch warning — %s", err)
        return data, None

    async def _fetch_market_extra(self, symbol: str, source: str) -> dict:
        """Fetch perpetual metrics (delta) or LTP quote (fyers)."""
        if source == "delta":
            logger.info("► Fetching perpetual metrics  symbol=%s", symbol)
            raw = await call_mcp_tool("delta", "fetch_perpetual_metrics",
                                      {"symbol": symbol})
        elif source == "fyers":
            logger.info("► Fetching quote  symbol=%s", symbol)
            raw = await call_mcp_tool("fyers", "get_quote", {"symbol": symbol})
        else:
            return {}
        result = json.loads(raw)
        return {} if "error" in result else result

    async def _fetch_news(self, symbol: str) -> str:
        """Fetch recent news headlines for the symbol."""
        logger.info("► Fetching news  symbol=%s", symbol)
        clean = symbol.replace("NSE:", "").replace("-EQ", "").split("/")[0]
        raw = await call_mcp_tool("tavily", "search_news",
                                  {"query": f"{clean} stock analysis", "max_results": 3})
        data = json.loads(raw)
        headlines = [r.get("title", "") for r in data.get("results", []) if r.get("title")]
        return " | ".join(headlines[:3]) or "No recent news"
