"""Analysis Agent — fetches market data via MCP and produces structured analysis."""

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.llm.factory import LLMFactory, traceable
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
        inp = task.input or {}
        symbol     = inp.get("symbol", "BTCUSDT")
        trade_type = inp.get("trade_type", "intraday")   # intraday | swing | longterm
        source     = inp.get("source", "delta")          # delta | fyers
        interval   = inp.get("interval", "5m")

        # Map trade_type → sensible interval
        if trade_type == "swing"    and interval == "5m":  interval = "1h"
        if trade_type == "longterm" and interval in ("5m","1h"): interval = "1D"

        logger.info("analysis_agent START  symbol=%s  type=%s  source=%s  interval=%s",
                    symbol, trade_type, source, interval)

        # ── 1. Fetch candles via MCP ──────────────────────────────────────────
        logger.info("► Fetching candles via %s MCP  symbol=%s  interval=%s", source, symbol, interval)
        candles_json = await call_mcp_tool(
            server=source,
            tool="fetch_candles",
            arguments={"symbol": symbol, "interval": interval, "lookback_days": 7},
        )
        candle_data = json.loads(candles_json)

        # ── 2. Fetch perp metrics or quote ────────────────────────────────────
        market_extra: dict = {}
        if source == "delta":
            logger.info("► Fetching perpetual metrics  symbol=%s", symbol)
            metrics_json = await call_mcp_tool(
                server="delta", tool="fetch_perpetual_metrics",
                arguments={"symbol": symbol},
            )
            market_extra = json.loads(metrics_json)
        elif source == "fyers":
            logger.info("► Fetching quote  symbol=%s", symbol)
            quote_json = await call_mcp_tool(
                server="fyers", tool="get_quote",
                arguments={"symbol": symbol},
            )
            market_extra = json.loads(quote_json)

        # ── 3. News context ───────────────────────────────────────────────────
        logger.info("► Fetching news  symbol=%s", symbol)
        clean_sym = symbol.replace("NSE:", "").replace("-EQ", "").split("/")[0]
        news_json = await call_mcp_tool(
            server="tavily", tool="search_news",
            arguments={"query": f"{clean_sym} stock analysis", "max_results": 3},
        )
        news_data = json.loads(news_json)
        headlines = [r.get("title","") for r in news_data.get("results",[]) if r.get("title")]
        news_context = " | ".join(headlines[:3]) or "No recent news"

        # ── 4. Build LLM prompt ───────────────────────────────────────────────
        recent = (candle_data.get("candles") or [])[-20:]
        prompt = (
            f"Symbol: {symbol}  Trade type: {trade_type.upper()}  Interval: {interval}\n"
            f"Last close: {candle_data.get('last_close', 'N/A')}\n"
            f"Recent candles (last {len(recent)}): {json.dumps(recent[:5])} ... "
            f"[{len(recent)} bars total]\n"
        )
        if market_extra.get("funding_signal"):
            prompt += (
                f"\nPerpetual metrics:\n"
                f"  Funding: {market_extra.get('funding_rate_8h_pct','?')}%  "
                f"Signal: {market_extra.get('funding_signal','?').upper()}\n"
                f"  OI: {market_extra.get('oi','?')}  Mark: {market_extra.get('mark_price','?')}\n"
            )
        if market_extra.get("ltp"):
            prompt += f"\nCurrent LTP: {market_extra['ltp']}\n"
        prompt += f"\nNews context: {news_context}\n"
        prompt += f"\nProduce {trade_type} trade analysis in the required JSON format."

        # ── 5. LLM analysis ───────────────────────────────────────────────────
        logger.info("► Calling LLM for analysis  provider=%s", self._llm_provider or "default")
        response = await self.llm.ainvoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip()

        # Parse JSON
        analysis: dict[str, Any] = {}
        try:
            cleaned = re.sub(r"```json|```", "", raw).strip()
            analysis = json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    analysis = json.loads(m.group())
                except Exception:
                    pass
            if not analysis:
                analysis = {"symbol": symbol, "summary": raw, "confidence": 0.5}

        analysis["symbol"]       = symbol
        analysis["trade_type"]   = trade_type
        analysis["news_context"] = news_context
        analysis["raw_candles"]  = recent[:10]  # keep last 10 for execution agent

        logger.info("analysis_agent END  symbol=%s  trend=%s  confidence=%s",
                    symbol, analysis.get("trend"), analysis.get("confidence"))

        return TaskResponse(
            task_id=task.task_id, agent=self.agent_name, status="completed",
            artifacts=[Artifact(type="analysis", data=analysis)],
        )
