"""Trade Assistant — TradingView chart + Analysis summary frontend."""

import requests
import streamlit as st
import streamlit.components.v1 as components

API_BASE = "http://localhost:8100"

st.set_page_config(
    page_title="Trade Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Remove default padding for max chart space */
.block-container { padding-top: 1rem; padding-bottom: 0.5rem; padding-left: 1rem; padding-right: 1rem; }
.stChatMessage { font-size: 13px; }

/* Summary card */
.summary-card {
  background: #0d1117;
  border: 1px solid #30363d;
  border-radius: 10px;
  padding: 14px 16px;
  margin: 6px 0;
  font-size: 12px;
  color: #c9d1d9;
}
.summary-title {
  font-size: 13px;
  font-weight: 700;
  color: #58a6ff;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 10px;
  border-bottom: 1px solid #21262d;
  padding-bottom: 6px;
}
.metric-grid { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; }
.metric-box {
  background: #161b22;
  border: 1px solid #21262d;
  border-radius: 6px;
  padding: 6px 10px;
  min-width: 70px;
  text-align: center;
  flex: 1;
}
.metric-label { font-size: 10px; color: #8b949e; margin-bottom: 2px; }
.metric-value { font-size: 13px; font-weight: 700; }
.bull { color: #3fb950; }
.bear { color: #f85149; }
.neutral { color: #d29922; }
.broker-badge {
  display: inline-block;
  background: #1f2937;
  border: 1px solid #374151;
  border-radius: 4px;
  padding: 2px 8px;
  font-size: 11px;
  font-weight: 600;
  color: #93c5fd;
  margin-left: 6px;
}
.targets-list { list-style: none; padding: 0; margin: 4px 0; }
.targets-list li { padding: 2px 0; font-size: 12px; }
.summary-text {
  font-size: 12px;
  color: #8b949e;
  line-height: 1.5;
  margin-top: 8px;
}
.no-analysis {
  text-align: center;
  color: #8b949e;
  font-size: 12px;
  padding: 30px 10px;
}
</style>
""", unsafe_allow_html=True)


# ── Symbol helpers ────────────────────────────────────────────────────────────
_DELTA_MAP = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
    "BNB": "BNBUSDT", "XAU": "XAUUSDT", "DOGE": "DOGEUSDT",
    "AVAX": "AVAXUSDT", "MATIC": "MATICUSDT", "LINK": "LINKUSDT",
}
_DELTA_SYMBOLS = set(_DELTA_MAP.keys()) | set(_DELTA_MAP.values())

_NSE_TV_MAP = {
    "NIFTY": "NSE:NIFTY",
    "NIFTY50": "NSE:NIFTY",
    "BANKNIFTY": "NSE:BANKNIFTY",
    "SENSEX": "BSE:SENSEX",
}


def detect_broker(raw: str) -> str:
    """Return 'delta' for crypto symbols, 'fyers' for NSE/Indian equities."""
    r = raw.upper().strip().removesuffix(".P")
    if r in _DELTA_SYMBOLS or r.endswith("USDT") or r.endswith("PERP"):
        return "delta"
    for k in _DELTA_MAP:
        if r == k:
            return "delta"
    return "fyers"


def to_tradingview_symbol(raw: str) -> str:
    """Convert raw user input to a TradingView symbol string."""
    r = raw.upper().strip().removesuffix(".P")
    # Exact NSE overrides
    if r in _NSE_TV_MAP:
        return _NSE_TV_MAP[r]
    # Crypto short names
    if r in _DELTA_MAP:
        return f"BINANCE:{_DELTA_MAP[r]}"
    # Already has USDT suffix
    if r.endswith("USDT"):
        return f"BINANCE:{r}"
    # Assume NSE equity
    return f"NSE:{r}"


def to_backend_symbol(raw: str) -> str:
    """Convert raw input to a symbol the backend orchestrator understands."""
    r = raw.upper().strip().removesuffix(".P")
    if r in _DELTA_MAP:
        return _DELTA_MAP[r]
    return r


def render_tradingview(tv_symbol: str, height: int = 580) -> None:
    chart_html = f"""
    <div class="tradingview-widget-container" style="height:{height}px;width:100%">
      <div class="tradingview-widget-container__widget" style="height:calc(100% - 32px);width:100%"></div>
      <script type="text/javascript"
        src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js"
        async>
      {{
        "autosize": true,
        "symbol": "{tv_symbol}",
        "interval": "D",
        "timezone": "Asia/Kolkata",
        "theme": "dark",
        "style": "1",
        "locale": "in",
        "withdateranges": true,
        "hide_side_toolbar": false,
        "allow_symbol_change": false,
        "studies": ["STD;MACD", "STD;RSI"],
        "support_host": "https://www.tradingview.com"
      }}
      </script>
    </div>
    """
    components.html(chart_html, height=height + 5)


# ── API helpers ───────────────────────────────────────────────────────────────
def _api(method: str, path: str, token: str = "", **kwargs) -> tuple[dict | None, str | None]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = requests.request(method, f"{API_BASE}{path}", headers=headers, timeout=120, **kwargs)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text[:200])
            except Exception:
                detail = r.text[:200]
            return None, f"HTTP {r.status_code}: {detail}"
        return r.json(), None
    except Exception as e:
        return None, str(e)


def _get(path, token="", **kw):  return _api("GET",  path, token, **kw)
def _post(path, token="", **kw): return _api("POST", path, token, **kw)


# ── Session state init ────────────────────────────────────────────────────────
for k, v in {
    "token": "", "username": "", "session_id": "",
    "messages": [], "total_tokens": 0,
    "llm_provider": "openai", "trade_mode": "paper",
    "current_symbol": "BTCUSDT",
    "current_analysis": None,
    "current_broker": "delta",
}.items():
    st.session_state.setdefault(k, v)


# ── Sidebar (auth + settings) ─────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Trade Assistant")

    health, err = _get("/health")
    if health and health.get("status") == "ok":
        st.success("● Backend connected", icon="🟢")
    else:
        st.error(f"● Backend offline: {err}", icon="🔴")
        st.info("Start backend:\n```\nuvicorn backend.main:app --port 8100 --reload\n```")
        st.stop()

    st.divider()

    if not st.session_state["token"]:
        st.subheader("Login / Register")
        auth_tab = st.radio("", ["Login", "Register"], horizontal=True, label_visibility="collapsed")
        uname = st.text_input("Username")
        pwd   = st.text_input("Password", type="password")

        if auth_tab == "Login" and st.button("Login", type="primary", use_container_width=True):
            data, err = _post("/auth/login", json=None, data={"username": uname, "password": pwd})
            if err:
                st.error(err)
            else:
                st.session_state["token"]    = data["access_token"]
                st.session_state["username"] = data["username"]
                st.rerun()

        if auth_tab == "Register" and st.button("Register", type="primary", use_container_width=True):
            data, err = _post("/auth/register", json={"username": uname, "password": pwd})
            if err:
                st.error(err)
            else:
                st.success("Account created — please login")

    else:
        st.markdown(f"👤 **{st.session_state['username']}**")
        if st.button("Logout", use_container_width=True):
            for k in ("token", "username", "session_id", "messages", "total_tokens"):
                st.session_state[k] = "" if isinstance(st.session_state[k], str) else \
                                       [] if isinstance(st.session_state[k], list) else 0
            st.session_state["current_analysis"] = None
            st.rerun()

        st.divider()
        st.subheader("Settings")
        st.session_state["llm_provider"] = st.selectbox(
            "LLM Provider", ["openai", "ollama"],
            index=0 if st.session_state["llm_provider"] == "openai" else 1,
        )
        st.session_state["trade_mode"] = st.selectbox(
            "Trade Mode", ["paper", "real"],
            index=0 if st.session_state["trade_mode"] == "paper" else 1,
        )
        if st.session_state["trade_mode"] == "real":
            st.warning("⚠️ Real trades will be placed!", icon="⚠️")

        st.divider()
        st.subheader("Tokens")
        st.markdown(
            f'<div style="font-size:22px;font-weight:800;color:#58a6ff">'
            f'{st.session_state["total_tokens"]:,}</div>'
            f'<div style="font-size:11px;color:#8b949e">this session</div>',
            unsafe_allow_html=True,
        )
        cost = st.session_state["total_tokens"] * 0.000_000_375
        st.markdown(f'<div style="font-size:12px;color:#8b949e">~${cost:.4f} est. cost</div>',
                    unsafe_allow_html=True)

        st.divider()
        st.subheader("Sessions")
        if st.button("＋ New Session", use_container_width=True, type="primary"):
            data, err = _post("/sessions/new", token=st.session_state["token"])
            if not err:
                st.session_state["session_id"]    = data["session_id"]
                st.session_state["messages"]      = []
                st.session_state["total_tokens"]  = 0
                st.session_state["current_analysis"] = None
                st.rerun()

        sessions_data, _ = _get("/sessions", token=st.session_state["token"])
        for sess in (sessions_data or {}).get("sessions", [])[:8]:
            sid = sess["session_id"][:8]
            if st.button(f"{sid}... ({sess['created'][:10]})",
                         key=f"sess_{sess['session_id']}", use_container_width=True):
                st.session_state["session_id"] = sess["session_id"]
                msgs_data, _ = _get(f"/sessions/{sess['session_id']}/messages",
                                     token=st.session_state["token"])
                if msgs_data:
                    st.session_state["messages"]     = msgs_data["messages"]
                    st.session_state["total_tokens"] = msgs_data["total_tokens"]
                st.rerun()


# ── Guard: must be logged in ──────────────────────────────────────────────────
if not st.session_state["token"]:
    st.markdown("## 📈 Trade Assistant")
    st.info("Please login from the sidebar (→) to start.")
    st.stop()

# Ensure active session
if not st.session_state["session_id"]:
    data, err = _post("/sessions/new", token=st.session_state["token"])
    if not err:
        st.session_state["session_id"] = data["session_id"]


# ── Main layout: chart (left) | symbol + summary (right) ─────────────────────
col_chart, col_panel = st.columns([13, 7], gap="medium")

# ── RIGHT PANEL ───────────────────────────────────────────────────────────────
with col_panel:
    st.markdown("#### Symbol")

    sym_input = st.text_input(
        "Symbol",
        value=st.session_state["current_symbol"],
        placeholder="e.g. BTCUSDT, RELIANCE, NIFTY",
        label_visibility="collapsed",
    )

    broker = detect_broker(sym_input)
    broker_label = "Delta (Crypto)" if broker == "delta" else "Fyers (NSE)"
    broker_color = "#60a5fa" if broker == "delta" else "#34d399"
    st.markdown(
        f'<span style="font-size:11px;color:#8b949e;">Broker: </span>'
        f'<span style="font-size:12px;font-weight:700;color:{broker_color}">{broker_label}</span>',
        unsafe_allow_html=True,
    )

    btn_col1, btn_col2 = st.columns(2)
    analyse_clicked = btn_col1.button("📊 Analyse", type="primary", use_container_width=True)
    execute_clicked = btn_col2.button("⚡ Execute", use_container_width=True)

    # Update symbol on any button click
    if analyse_clicked or execute_clicked:
        st.session_state["current_symbol"] = sym_input.upper().strip()
        st.session_state["current_broker"] = broker

    st.divider()

    # ── Analysis / Execute action ─────────────────────────────────────────────
    if analyse_clicked or execute_clicked:
        backend_sym = to_backend_symbol(sym_input)
        intent = "Analyse" if analyse_clicked else "Execute trade on"
        prompt = f"{intent} {backend_sym} for intraday trade"

        with st.spinner("Fetching analysis…"):
            data, err = _post(
                "/chat",
                token=st.session_state["token"],
                json={
                    "message":      prompt,
                    "session_id":   st.session_state["session_id"],
                    "llm_provider": st.session_state["llm_provider"],
                    "trade_mode":   st.session_state["trade_mode"],
                },
            )

        if err:
            st.error(f"Error: {err}")
        else:
            st.session_state["total_tokens"] = data.get("session_tokens", 0)
            analysis = data.get("analysis") or {}
            execution = data.get("execution") or {}
            st.session_state["current_analysis"] = {
                "analysis":  analysis,
                "execution": execution,
                "reply":     data.get("reply", ""),
                "tokens":    data.get("tokens_used", 0),
            }

        st.rerun()

    # ── Summary panel ─────────────────────────────────────────────────────────
    cached = st.session_state.get("current_analysis")

    if not cached:
        st.markdown(
            '<div class="no-analysis">Enter a symbol above and click<br>'
            '<strong>Analyse</strong> to load the trade setup summary.</div>',
            unsafe_allow_html=True,
        )
    else:
        a = cached.get("analysis", {})
        e = cached.get("execution", {})

        # ── Trade Setup Summary ──────────────────────────────────────────────
        if a:
            trend     = str(a.get("trend", "—")).upper()
            strength  = a.get("strength", "—")
            price     = a.get("last_price", "—")
            entry     = a.get("entry_zone", "—")
            sl        = a.get("stop_loss", "—")
            targets   = a.get("targets", [])
            rr        = a.get("rr_ratio", "—")
            conf      = a.get("confidence", "—")
            summary   = a.get("summary", "")
            key_levels = a.get("key_levels", {})
            trade_type = str(a.get("trade_type", "")).upper()

            trend_class = "bull" if "bull" in trend.lower() or "up" in trend.lower() \
                          else "bear" if "bear" in trend.lower() or "down" in trend.lower() \
                          else "neutral"
            trend_icon  = "↑" if trend_class == "bull" else "↓" if trend_class == "bear" else "→"

            # Header row
            st.markdown(
                f'<div class="summary-card">'
                f'<div class="summary-title">📊 Trade Setup — {a.get("symbol", sym_input.upper())}</div>'
                f'<div class="metric-grid">'
                f'  <div class="metric-box">'
                f'    <div class="metric-label">TREND</div>'
                f'    <div class="metric-value {trend_class}">{trend} {trend_icon}</div>'
                f'  </div>'
                f'  <div class="metric-box">'
                f'    <div class="metric-label">STRENGTH</div>'
                f'    <div class="metric-value">{strength}</div>'
                f'  </div>'
                f'  <div class="metric-box">'
                f'    <div class="metric-label">PRICE</div>'
                f'    <div class="metric-value">{price}</div>'
                f'  </div>'
                f'  <div class="metric-box">'
                f'    <div class="metric-label">CONFIDENCE</div>'
                f'    <div class="metric-value">{conf}</div>'
                f'  </div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Entry / SL / Targets
            targets_html = "".join(
                f'<li>T{i+1}: <strong>{t}</strong></li>' for i, t in enumerate(targets)
            ) if targets else "<li>—</li>"

            st.markdown(
                f'<div class="summary-card">'
                f'<div class="summary-title">🎯 Levels</div>'
                f'<div class="metric-grid">'
                f'  <div class="metric-box" style="flex:2">'
                f'    <div class="metric-label">ENTRY ZONE</div>'
                f'    <div class="metric-value neutral">{entry}</div>'
                f'  </div>'
                f'  <div class="metric-box">'
                f'    <div class="metric-label">STOP LOSS</div>'
                f'    <div class="metric-value bear">{sl}</div>'
                f'  </div>'
                f'  <div class="metric-box">'
                f'    <div class="metric-label">R:R</div>'
                f'    <div class="metric-value bull">{rr}</div>'
                f'  </div>'
                f'</div>'
                f'<div style="margin-top:8px;">'
                f'  <div class="metric-label" style="margin-bottom:4px">TARGETS</div>'
                f'  <ul class="targets-list">{targets_html}</ul>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Key levels if present
            if key_levels:
                kl_items = "  ".join(
                    f'<span style="color:#8b949e">{k}:</span> <strong>{v}</strong>'
                    for k, v in (key_levels.items() if isinstance(key_levels, dict) else [])
                )
                if kl_items:
                    st.markdown(
                        f'<div class="summary-card">'
                        f'<div class="summary-title">📐 Key Levels</div>'
                        f'<div style="font-size:12px;line-height:1.8">{kl_items}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Summary text
            if summary:
                st.markdown(
                    f'<div class="summary-card">'
                    f'<div class="summary-title">💬 Analysis</div>'
                    f'<div class="summary-text">{summary}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Execution details (if execute was called) ────────────────────────
        if e:
            action   = str(e.get("action", "—")).upper()
            qty      = e.get("qty", "—")
            mode     = str(e.get("mode", "paper")).upper()
            rationale = e.get("rationale", "")
            order_type = e.get("order_type", "—")

            mode_color = "#f85149" if mode == "REAL" else "#3fb950"
            action_color = "#3fb950" if action in ("BUY", "LONG") else "#f85149"

            st.markdown(
                f'<div class="summary-card">'
                f'<div class="summary-title">⚡ Execution Plan</div>'
                f'<div class="metric-grid">'
                f'  <div class="metric-box">'
                f'    <div class="metric-label">ACTION</div>'
                f'    <div class="metric-value" style="color:{action_color}">{action}</div>'
                f'  </div>'
                f'  <div class="metric-box">'
                f'    <div class="metric-label">QTY</div>'
                f'    <div class="metric-value">{qty}</div>'
                f'  </div>'
                f'  <div class="metric-box">'
                f'    <div class="metric-label">ORDER</div>'
                f'    <div class="metric-value">{order_type}</div>'
                f'  </div>'
                f'  <div class="metric-box">'
                f'    <div class="metric-label">MODE</div>'
                f'    <div class="metric-value" style="color:{mode_color}">{mode}</div>'
                f'  </div>'
                f'</div>'
                + (f'<div class="summary-text">{rationale}</div>' if rationale else "")
                + f'</div>',
                unsafe_allow_html=True,
            )

        # Token usage
        if cached.get("tokens"):
            st.markdown(
                f'<div style="font-size:10px;color:#8b949e;text-align:right;margin-top:4px">'
                f'~{cached["tokens"]:,} tokens used</div>',
                unsafe_allow_html=True,
            )


# ── LEFT PANEL: TradingView chart ─────────────────────────────────────────────
with col_chart:
    tv_symbol = to_tradingview_symbol(st.session_state["current_symbol"])
    render_tradingview(tv_symbol, height=640)
