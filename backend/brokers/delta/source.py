"""Delta Exchange data source — public market data and option chain."""

from __future__ import annotations

import hashlib
import hmac
import logging

logger = logging.getLogger(__name__)
import time
from datetime import datetime, timezone
from typing import Any

import httpx
import pandas as pd


class DeltaSource:
    """Fetch OHLCV and option chain data from Delta Exchange.

    Public market data (candles, tickers, products) requires NO authentication.
    Private endpoints (orders, positions, wallet) require API key + HMAC signature.
    """

    BASE_URL = "https://api.delta.exchange/v2"
    INDIA_URL = "https://api.india.delta.exchange/v2"

    # Interval string to Delta resolution code
    # Delta API uses human-readable strings: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 1d, 1w
    INTERVAL_MAP: dict[str, str] = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "12h": "1h",   # no 12h — fall back to 1h
        "1D": "1d",
        "1d": "1d",
        "1w": "1w",
    }

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Convert TradingView-style Delta symbols to the Delta API format.

        TradingView uses a .P suffix for perpetuals (e.g. BTCUSD.P, BTCUSDT.P).
        Delta's own API just wants the bare symbol (BTCUSD, BTCUSDT).
        Strip it so users can paste TV symbols directly into the UI.
        """
        return symbol.upper().removesuffix(".P")

    def __init__(self, api_key: str = "", api_secret: str = "", region: str = "global"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.region = region  # "global" or "india"
        self.base_url = self.INDIA_URL if region == "india" else self.BASE_URL
        self._session: httpx.AsyncClient | None = None

    async def _get_session(self) -> httpx.AsyncClient:
        """Lazy-init httpx.AsyncClient."""
        if self._session is None:
            self._session = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
                headers={
                    "User-Agent": "TradingAnalyst/1.0",
                    "Accept": "application/json",
                },
            )
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session is not None:
            await self._session.aclose()
            self._session = None

    # ── Signature (for authenticated endpoints) ───────────────────────────────

    def _generate_signature(
        self, method: str, path: str, query: str, payload: str
    ) -> tuple[str, str]:
        """Generate HMAC-SHA256 signature for authenticated requests.

        Delta Exchange signature spec (v2):
            message = method + timestamp + path + query_string + body
        Returns (timestamp, signature).
        """
        timestamp = str(int(time.time()))
        # Delta spec: METHOD + timestamp + path + query + body  (in that order)
        message = method + timestamp + path + query + payload
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return timestamp, signature

    async def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        body: dict | None = None,
        authenticated: bool = False,
    ) -> dict[str, Any]:
        """Make an async HTTP request to Delta API.

        If authenticated=True, add api-key, timestamp, signature headers.
        """
        session = await self._get_session()
        headers: dict[str, str] = {}

        if authenticated:
            if not self.api_key or not self.api_secret:
                raise ValueError(
                    "Delta API key and secret are required for authenticated requests"
                )
            query = ""
            if params:
                query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            payload_str = ""
            if body:
                import json

                payload_str = json.dumps(body, separators=(",", ":"))
            timestamp, signature = self._generate_signature(method, path, query, payload_str)
            headers["api-key"] = self.api_key
            headers["timestamp"] = timestamp
            headers["signature"] = signature

        url = f"{self.base_url}{path}"
        try:
            response = await session.request(
                method=method,
                url=url,
                params=params,
                json=body,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            logging.warning(
                "Delta API HTTP error %s for %s %s: %s",
                exc.response.status_code,
                method,
                url,
                exc.response.text[:200],
            )
            return {"success": False, "error": exc.response.text[:200]}
        except Exception as exc:
            logging.warning("Delta API request failed for %s %s: %s", method, url, exc)
            return {"success": False, "error": str(exc)}

    # ── OHLC ──────────────────────────────────────────────────────────────────

    async def fetch_ohlc(
        self,
        symbol: str,
        from_dt: datetime,
        to_dt: datetime,
        interval: str = "1m",
    ) -> pd.DataFrame:
        """Fetch historical candles from Delta.

        Parameters
        ----------
        symbol : str
            Delta perpetual symbol, e.g. "BTCUSDT" (BTC/USD), "XAUUSDT" (Gold/USD).
            Check https://www.delta.exchange/app/futures for live symbol list.
        from_dt : datetime
            Start datetime.
        to_dt : datetime
            End datetime.
        interval : str
            One of: "1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1D".
            Mapped to Delta resolution codes: "1", "5", "15", "30", "60", "120",
            "240", "360", "720", "D".

        Returns
        -------
        pd.DataFrame
            Columns: datetime (index), open, high, low, close, volume
        """
        resolution = self.INTERVAL_MAP.get(interval)
        if resolution is None:
            logging.warning("Unknown interval %r for Delta, falling back to 1m", interval)
            resolution = "1m"

        # Guard: if datetimes arrive tz-naive (stripped by pandas), treat as UTC
        def _to_unix(dt: datetime) -> int:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())

        start_unix = _to_unix(from_dt)
        end_unix   = _to_unix(to_dt)

        symbol = self._normalize_symbol(symbol)
        params: dict[str, Any] = {
            "symbol": symbol,
            "resolution": resolution,
            "start": start_unix,
            "end": end_unix,
        }

        data = await self._request("GET", "/history/candles", params=params)

        if not data.get("success"):
            logging.warning(
                "Delta fetch_ohlc failed for %s: success=%s  error=%s  full=%s",
                symbol, data.get("success"), data.get("error"), str(data)[:300],
            )
            return self._empty_ohlc_df()

        raw_candles: list[dict] = data.get("result", [])
        if not raw_candles:
            return self._empty_ohlc_df()

        records = []
        for candle in raw_candles:
            records.append(
                {
                    "datetime": datetime.fromtimestamp(candle["time"], tz=timezone.utc),
                    "open": candle["open"],
                    "high": candle["high"],
                    "low": candle["low"],
                    "close": candle["close"],
                    "volume": candle["volume"],
                }
            )

        df = pd.DataFrame(records)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        return df[["open", "high", "low", "close", "volume"]]

    # ── Option Chain ──────────────────────────────────────────────────────────

    async def fetch_option_chain(
        self,
        underlying: str,
        expiry: str | None = None,
    ) -> pd.DataFrame:
        """Fetch option chain from Delta.

        Parameters
        ----------
        underlying : str
            Underlying asset symbol: "BTC", "ETH", "SOL", etc.
        expiry : str | None
            Expiry date in "DD-MM-YYYY" format (e.g. "05-06-2025").
            If None, fetches all live options.

        Returns
        -------
        pd.DataFrame
            Columns: strike, ce_oi, pe_oi, ce_iv, pe_iv, ce_ltp, pe_ltp
        """
        underlying = self._normalize_symbol(underlying)
        # /v2/tickers only supports contract_types filter; underlying and expiry
        # must be filtered client-side.
        params: dict[str, str] = {
            "contract_types": "call_options,put_options",
        }

        data = await self._request("GET", "/tickers", params=params)

        if not data.get("success"):
            logging.warning("Delta fetch_option_chain failed: %s", data.get("error", "unknown"))
            return self._empty_option_chain_df()

        raw_tickers: list[dict] = data.get("result", [])
        if not raw_tickers:
            return self._empty_option_chain_df()

        # Filter by underlying asset symbol (symbol starts with underlying, e.g. "C-BTC-...")
        underlying_upper = underlying.upper()
        raw_tickers = [
            t for t in raw_tickers
            if underlying_upper in t.get("symbol", "").upper()
        ]

        # Filter by expiry date if provided (stored as "settlement_time" or in symbol)
        if expiry:
            # expiry format "DD-MM-YYYY" → try matching against symbol or settlement_time
            raw_tickers = [
                t for t in raw_tickers
                if expiry in t.get("symbol", "") or expiry.replace("-", "") in t.get("symbol", "")
            ]

        # Group by strike
        strike_map: dict[float, dict[str, Any]] = {}

        for ticker in raw_tickers:
            try:
                strike_str = ticker.get("strike_price", "")
                if not strike_str:
                    continue
                strike = float(strike_str)
            except (ValueError, TypeError):
                continue

            contract_type = ticker.get("contract_type", "")
            is_call = contract_type == "call_options"
            is_put = contract_type == "put_options"
            if not (is_call or is_put):
                continue

            # OI
            try:
                oi = float(ticker.get("oi") or 0)
            except (ValueError, TypeError):
                oi = 0.0

            # Mark price (LTP)
            try:
                mark = float(ticker.get("mark_price") or 0)
            except (ValueError, TypeError):
                mark = 0.0

            # IV — use average of ask_iv and bid_iv from quotes
            iv = 0.0
            quotes = ticker.get("quotes", {}) or {}
            try:
                ask_iv_str = quotes.get("ask_iv", "0") or "0"
                bid_iv_str = quotes.get("bid_iv", "0") or "0"
                if ask_iv_str and bid_iv_str:
                    ask_iv = float(ask_iv_str)
                    bid_iv = float(bid_iv_str)
                    iv = (ask_iv + bid_iv) / 2.0
            except (ValueError, TypeError):
                pass

            if strike not in strike_map:
                strike_map[strike] = {
                    "strike": strike,
                    "ce_oi": 0.0,
                    "pe_oi": 0.0,
                    "ce_iv": 0.0,
                    "pe_iv": 0.0,
                    "ce_ltp": 0.0,
                    "pe_ltp": 0.0,
                }

            if is_call:
                strike_map[strike]["ce_oi"] = oi
                strike_map[strike]["ce_ltp"] = mark
                strike_map[strike]["ce_iv"] = iv
            elif is_put:
                strike_map[strike]["pe_oi"] = oi
                strike_map[strike]["pe_ltp"] = mark
                strike_map[strike]["pe_iv"] = iv

        rows = list(strike_map.values())
        if not rows:
            return self._empty_option_chain_df()

        df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
        return df[["strike", "ce_oi", "pe_oi", "ce_iv", "pe_iv", "ce_ltp", "pe_ltp"]]

    # ── Order placement (authenticated) ──────────────────────────────────────

    async def place_order(
        self,
        symbol: str,
        side: str,
        size: int,
        order_type: str = "market_order",
        limit_price: float = 0.0,
        sl: float = 0.0,
        target: float = 0.0,
    ) -> dict[str, Any]:
        """Place a single order (with optional bracket SL/target).

        Parameters
        ----------
        symbol     : str   e.g. "BTCUSD", "XAUUSD"
        side       : str   "buy" or "sell"
        size       : int   Contract size
        order_type : str   "market_order" | "limit_order"
        limit_price: float Limit price (required for limit_order)
        sl         : float Stop-loss price (absolute); 0 = no SL
        target     : float Take-profit price (absolute); 0 = no TP
        """
        # Resolve product_id from symbol via products endpoint (paginated)
        symbol = self._normalize_symbol(symbol)
        product_id: int | None = None
        page = 1
        while product_id is None:
            products_data = await self._request("GET", "/products",
                                                params={"page": page, "page_size": 100})
            results = products_data.get("result", [])
            for p in results:
                if (p.get("symbol") or "").upper() == symbol:
                    product_id = int(p["id"])
                    break
            if product_id is not None or len(results) < 100:
                break
            page += 1
        if product_id is None:
            return {"success": False, "error": f"Symbol {symbol!r} not found in Delta products"}

        body: dict[str, Any] = {
            "product_id": product_id,
            "order_type": order_type,
            "side": side.lower(),
            "size": size,
        }
        if order_type == "limit_order" and limit_price:
            body["limit_price"] = str(round(limit_price, 2))
        if sl:
            body["bracket_order"] = True
            body["bracket_stop_loss_price"] = str(round(sl, 2))
        if target:
            body["bracket_order"] = True
            body["bracket_take_profit_price"] = str(round(target, 2))

        logging.info(
            "DeltaSource.place_order: symbol=%s side=%s size=%d order_type=%s sl=%s target=%s",
            symbol, side, size, order_type, sl or "—", target or "—",
        )
        return await self._request("POST", "/orders", body=body, authenticated=True)

    async def get_positions(self) -> list[dict[str, Any]]:
        """Return open margined positions."""
        data = await self._request("GET", "/positions/margined", authenticated=True)
        return data.get("result", []) or []

    async def cancel_order(self, order_id: int | str) -> dict[str, Any]:
        """Cancel an order by its ID."""
        logging.info("DeltaSource.cancel_order: id=%s", order_id)
        return await self._request("DELETE", f"/orders/{order_id}", authenticated=True)

    # ── Funding Rate & OI ────────────────────────────────────────────────────────

    async def fetch_perpetual_metrics(self, symbol: str) -> dict[str, Any]:
        """Fetch funding rate and open interest for a perpetual futures symbol.

        Returns a dict with:
            funding_rate        : float  — current 8h funding rate (e.g. 0.0001 = 0.01%)
            funding_rate_8h_pct : float  — same, as percentage
            funding_signal      : str    — "bullish"|"bearish"|"neutral"
            funding_note        : str    — human-readable interpretation
            oi                  : float  — open interest in contracts
            oi_change_pct       : float  — OI change vs 24h ago (if available)
            oi_signal           : str    — "building"|"unwinding"|"neutral"
            mark_price          : float  — current mark price
        """
        symbol = self._normalize_symbol(symbol)
        result: dict[str, Any] = {
            "symbol": symbol,
            "funding_rate": None,
            "funding_rate_8h_pct": None,
            "funding_signal": "neutral",
            "funding_note": "",
            "oi": None,
            "oi_change_pct": None,
            "oi_signal": "neutral",
            "mark_price": None,
        }

        # ── Ticker (mark price, OI, funding rate — all in one call) ─────────────
        try:
            ticker_data = await self._request("GET", f"/tickers/{symbol}")
            t = ticker_data.get("result", {}) or {}

            mark = t.get("mark_price")
            if mark:
                result["mark_price"] = round(float(mark), 4)

            # Open Interest in contracts
            oi_raw = t.get("oi") or t.get("open_interest")
            if oi_raw:
                result["oi"] = float(oi_raw)

            # OI in USD (more meaningful than contract count)
            oi_usd = t.get("oi_value_usd") or t.get("oi_value")
            if oi_usd:
                result["oi_usd"] = round(float(oi_usd), 2)

            # Funding rate — available directly on the ticker (no product lookup needed)
            fr_direct = t.get("funding_rate")
            if fr_direct is not None:
                fr_float = float(fr_direct)
                result["funding_rate"] = round(fr_float, 8)
                result["funding_rate_8h_pct"] = round(fr_float * 100, 4)
                if fr_float > 0.0005:
                    result["funding_signal"] = "bearish"
                    result["funding_note"] = (
                        f"High positive funding {fr_float*100:.3f}% — longs overleveraged, "
                        f"expect long squeeze / price pullback"
                    )
                elif fr_float > 0.0001:
                    result["funding_signal"] = "neutral"
                    result["funding_note"] = (
                        f"Moderate positive funding {fr_float*100:.3f}% — bullish market bias"
                    )
                elif fr_float < -0.0005:
                    result["funding_signal"] = "bullish"
                    result["funding_note"] = (
                        f"Negative funding {fr_float*100:.3f}% — shorts overleveraged, "
                        f"expect short squeeze / price bounce"
                    )
                elif fr_float < -0.0001:
                    result["funding_signal"] = "neutral"
                    result["funding_note"] = (
                        f"Slightly negative funding {fr_float*100:.3f}% — bearish market bias"
                    )
                else:
                    result["funding_signal"] = "neutral"
                    result["funding_note"] = (
                        f"Neutral funding {fr_float*100:.3f}% — balanced market"
                    )
                logger.info(
                    "DeltaSource.fetch_perpetual_metrics: ticker  symbol=%s  mark=%.2f  "
                    "funding=%.4f%%  oi=%.2f  signal=%s",
                    symbol, result.get("mark_price", 0),
                    result.get("funding_rate_8h_pct", 0),
                    result.get("oi", 0),
                    result.get("funding_signal"),
                )

            # 24h OI change — only use if it looks like a percentage/delta (not absolute USD)
            oi_24h = t.get("oi_value_symbol_24h") or t.get("oi_change_usd_24h")
            # Skip oi_change_usd_6h — it's absolute USD which gives nonsense % vs contract count
            if oi_24h and result.get("oi"):
                try:
                    oi_change = float(oi_24h)
                    pct = oi_change / result["oi"] * 100
                    # Sanity check: ignore if % looks like absolute USD (>1000%)
                    if abs(pct) < 200:
                        result["oi_change_pct"] = round(pct, 2)
                        if pct > 5:
                            result["oi_signal"] = "building"
                        elif pct < -5:
                            result["oi_signal"] = "unwinding"
                except (ValueError, ZeroDivisionError):
                    pass

        except Exception as exc:
            logger.warning("DeltaSource.fetch_perpetual_metrics: ticker failed — %s", exc)

        # ── Funding Rate fallback — only if ticker didn't have funding_rate ─────
        if result.get("funding_rate") is None:
            try:
                products_data = await self._request("GET", "/products",
                                                    params={"contract_types": "perpetual_futures",
                                                            "page": 1, "page_size": 100})
                product_id = None
                for p in products_data.get("result", []):
                    if (p.get("symbol") or "").upper() == symbol:
                        product_id = p.get("id")
                        break
                if product_id:
                    fr_data = await self._request("GET", f"/products/{product_id}/funding/current")
                    fr_val  = (fr_data.get("result") or {}).get("rate") or \
                               (fr_data.get("result") or {}).get("funding_rate")
                    if fr_val is not None:
                        fr_float = float(fr_val)
                        result["funding_rate"]       = round(fr_float, 8)
                        result["funding_rate_8h_pct"] = round(fr_float * 100, 4)
                        result["funding_signal"] = (
                            "bearish" if fr_float > 0.0005 else
                            "bullish" if fr_float < -0.0005 else
                            "neutral"
                        )
                        result["funding_note"] = f"Funding {fr_float*100:.3f}%/8h"
            except Exception as exc:
                logger.warning("DeltaSource.fetch_perpetual_metrics: funding fallback failed — %s", exc)

        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_ohlc_df() -> pd.DataFrame:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        ).set_index(pd.DatetimeIndex([], name="datetime"))

    @staticmethod
    def _empty_option_chain_df() -> pd.DataFrame:
        return pd.DataFrame(
            columns=["strike", "ce_oi", "pe_oi", "ce_iv", "pe_iv", "ce_ltp", "pe_ltp"]
        )