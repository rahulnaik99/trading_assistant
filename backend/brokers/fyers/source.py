"""Fyers data source using fyers_apiv3 SDK with access token auth."""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Bug #2 fix — interval string translation map
_INTERVAL_MAP: dict[str, str] = {
    "1m":  "1",
    "2m":  "2",
    "3m":  "3",
    "5m":  "5",
    "10m": "10",
    "15m": "15",
    "20m": "20",
    "30m": "30",
    "1h":  "60",
    "1H":  "60",
    "2h":  "120",
    "4h":  "240",
    "1D":  "1D",
    "1d":  "1D",
}


class FyersSource:
    """Fetch OHLCV data from Fyers API v3 with access token auth."""

    API_BASE = "https://api-t1.fyers.in/api/v3"
    DATA_URL = "https://api-t1.fyers.in/data/history"

    def __init__(
        self,
        client_id: str = "",
        access_token: str = "",
    ) -> None:
        self.client_id = client_id
        self.access_token = access_token
        self._client = None  # lazy-loaded
        logger.info("FyersSource: client_id=%s  token_set=%s", client_id, bool(access_token))

    def _composite_token(self, token: str = "") -> str:
        """Format token as client_id:token for Fyers v3 API."""
        token = token or self.access_token
        if not token:
            raise ValueError("access_token is required")
        if ":" in token:
            return token  # already composite
        if not self.client_id:
            raise ValueError("client_id is required to build composite token")
        composite = f"{self.client_id}:{token}"
        logger.info("Fyers _composite_token: formatted composite token")
        return composite

    def check_auth(self) -> tuple[bool, str]:
        """Check if Fyers access token is valid via /profile API."""
        if not self.access_token:
            return False, "FYERS_ACCESS_TOKEN not configured"

        try:
            auth_header = self._composite_token()
            profile_resp = requests.get(
                f"{self.API_BASE}/profile",
                headers={"Authorization": auth_header},
                timeout=10,
            )
            data = profile_resp.json()
            logger.info("Fyers check_auth: /profile response s=%s code=%s", data.get("s"), data.get("code"))
            if data.get("s") == "ok" or data.get("code") == 200:
                return True, "Authenticated"
            return False, "Token invalid or expired"
        except Exception as exc:
            logger.exception("Fyers check_auth: exception")
            return False, f"Fyers auth check failed: {exc}"

    def fetch_ohlc(
        self,
        symbol: str,
        from_dt: datetime,
        to_dt: datetime,
        interval: str = "1",
        access_token: str = "",
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Fyers historical API v3.

        Parameters
        ----------
        symbol : str
            Fyers symbol e.g. "NSE:RELIANCE-EQ".
        from_dt : datetime
            Start datetime.
        to_dt : datetime
            End datetime.
        interval : str
            Human-readable interval (1m, 5m, 15m, 1h, 1D) — auto-translated to Fyers format.
        access_token : str
            User's Fyers access token (falls back to self.access_token).
        """
        # Bug #2 fix — translate interval to Fyers API format
        resolution = _INTERVAL_MAP.get(interval, interval)

        token = access_token or self.access_token
        if not token:
            raise ValueError("access_token is required for Fyers historical data fetch")
        logger.info(
            "Fyers fetch_ohlc: symbol=%s interval=%s (resolution=%s) from=%s to=%s",
            symbol, interval, resolution, from_dt, to_dt,
        )

        data_payload = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "0",  # epoch seconds
            "range_from": int(from_dt.timestamp()),
            "range_to": int(to_dt.timestamp()),
            "cont_flag": "1",
        }
        auth_header = self._composite_token(token)
        data_resp = requests.get(
            self.DATA_URL,
            headers={"Authorization": auth_header},
            params=data_payload,
            timeout=30,
        )
        response = data_resp.json()
        logger.info(
            "Fyers fetch_ohlc: response s=%s candles=%s",
            response.get("s"), len(response.get("candles", [])),
        )

        if response.get("s") != "ok" or not response.get("candles"):
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            ).set_index(pd.DatetimeIndex([], name="datetime"))

        candles = response["candles"]
        df = pd.DataFrame(
            candles,
            columns=["datetime", "open", "high", "low", "close", "volume"],
        )
        df["datetime"] = pd.to_datetime(df["datetime"], unit="s").dt.tz_localize("UTC")
        df.set_index("datetime", inplace=True)
        logger.info("Fyers fetch_ohlc: returning %s rows  columns=%s", len(df), list(df.columns))

        return df[["open", "high", "low", "close", "volume"]].astype(float)

    # ── Order placement ───────────────────────────────────────────────────────

    def place_order(self, data: dict) -> dict:
        """Place an order via fyers_apiv3 SDK.

        data must conform to the fyers_apiv3 place_order schema:
            symbol, qty, type (1=limit/2=market), side (1=buy/-1=sell),
            productType (CNC/INTRADAY/BO/CO), limitPrice, stopPrice,
            validity (DAY/IOC), stopLoss, takeProfit (for BO)
        """
        from fyers_apiv3 import fyersModel  # type: ignore[import]

        token = (
            self.access_token
            if ":" in self.access_token
            else f"{self.client_id}:{self.access_token}"
        )
        fyers = fyersModel.FyersModel(
            client_id=self.client_id,
            token=token,
            log_path="",
        )
        logger.info("FyersSource.place_order: symbol=%s productType=%s side=%s qty=%s",
                    data.get("symbol"), data.get("productType"),
                    data.get("side"), data.get("qty"))
        response = fyers.place_order(data=data)
        logger.info("FyersSource.place_order: response=%s", response)
        return response