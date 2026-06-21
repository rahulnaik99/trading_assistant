"""Trade Assistant — Streamlit ChatGPT-style frontend."""

import time
from datetime import datetime

import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8100"

st.set_page_config(
    page_title="Trade Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stChatMessage { font-size: 13px; }
.token-badge {
  display:inline-block; background:#161b22; border:1px solid #30363d;
  border-radius:4px; padding:2px 8px; font-size:10px; color:#8b949e;
  margin-left:8px; vertical-align:middle;
}
.analysis-card {
  background:#0d1117; border:1px solid #30363d; border-radius:8px;
  padding:12px 16px; margin:8px 0; font-size:12px;
}
.metric-row { display:flex; gap:16px; flex-wrap:wrap; margin:8px 0; }
.metric-box {
  background:#161b22; border-radius:6px; padding:8px 12px; min-width:80px;
  text-align:center;
}
.metric-label { font-size:10px; color:#8b949e; margin-bottom:2px; }
.metric-value { font-size:14px; font-weight:700; }
</style>
""", unsafe_allow_html=True)


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
    "messages": [], "total_tokens": 0, "sessions_list": [],
    "llm_provider": "openai", "trade_mode": "paper",
}.items():
    st.session_state.setdefault(k, v)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Trade Assistant")

    # Backend status
    health, err = _get("/health")
    if health and health.get("status") == "ok":
        st.success("● Backend connected", icon="🟢")
    else:
        st.error(f"● Backend offline: {err}", icon="🔴")
        st.info("Start backend:\n```\ncd trade_assistant\nuvicorn backend.main:app --port 8100 --reload\n```")
        st.stop()

    st.divider()

    # Auth
    if not st.session_state["token"]:
        st.subheader("Login / Register")
        auth_tab = st.radio("", ["Login", "Register"], horizontal=True, label_visibility="collapsed")
        uname = st.text_input("Username")
        pwd   = st.text_input("Password", type="password")

        if auth_tab == "Login" and st.button("Login", type="primary", use_container_width=True):
            data, err = _post("/auth/login", json=None,
                              data={"username": uname, "password": pwd})
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
            for k in ("token","username","session_id","messages","total_tokens"):
                st.session_state[k] = "" if isinstance(st.session_state[k], str) else \
                                       [] if isinstance(st.session_state[k], list) else 0
            st.rerun()

        st.divider()

        # Settings
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

        # Token counter
        st.divider()
        st.subheader("Session Tokens")
        st.markdown(
            f'<div style="font-size:24px;font-weight:800;color:#58a6ff">'
            f'{st.session_state["total_tokens"]:,}</div>'
            f'<div style="font-size:11px;color:#8b949e">total tokens this session</div>',
            unsafe_allow_html=True,
        )
        # Rough cost estimate (gpt-4o-mini: $0.15/1M input + $0.60/1M output)
        cost = st.session_state["total_tokens"] * 0.000_000_375  # blended avg
        st.markdown(
            f'<div style="font-size:12px;color:#8b949e">~${cost:.4f} estimated cost</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # Session management
        st.subheader("Sessions")
        if st.button("＋ New Session", use_container_width=True, type="primary"):
            data, err = _post("/sessions/new", token=st.session_state["token"])
            if not err:
                st.session_state["session_id"] = data["session_id"]
                st.session_state["messages"]   = []
                st.session_state["total_tokens"] = 0
                st.rerun()

        # Load session list
        sessions_data, _ = _get("/sessions", token=st.session_state["token"])
        sessions = (sessions_data or {}).get("sessions", [])
        for sess in sessions[:10]:
            sid   = sess["session_id"][:8]
            label = f"{sid}... ({sess['created'][:10]})"
            if st.button(label, key=f"sess_{sess['session_id']}", use_container_width=True):
                st.session_state["session_id"] = sess["session_id"]
                # Load message history
                msgs_data, _ = _get(f"/sessions/{sess['session_id']}/messages",
                                     token=st.session_state["token"])
                if msgs_data:
                    st.session_state["messages"]     = msgs_data["messages"]
                    st.session_state["total_tokens"] = msgs_data["total_tokens"]
                st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
if not st.session_state["token"]:
    st.markdown("## 📈 Trade Assistant\nPlease login from the sidebar to start.")
    st.stop()

# Ensure active session
if not st.session_state["session_id"]:
    data, err = _post("/sessions/new", token=st.session_state["token"])
    if not err:
        st.session_state["session_id"] = data["session_id"]

# Header
col_title, col_info = st.columns([7, 3])
with col_title:
    st.markdown("## 📈 Trade Assistant")
    st.markdown(
        f'<span style="font-size:11px;color:#8b949e">'
        f'Session: {st.session_state["session_id"][:12]}...  '
        f'·  Provider: {st.session_state["llm_provider"]}  '
        f'·  Mode: {st.session_state["trade_mode"].upper()}</span>',
        unsafe_allow_html=True,
    )

with col_info:
    st.markdown(
        '<div style="font-size:11px;color:#8b949e;margin-top:14px">'
        'Try: "Analyse RELIANCE for intraday" '
        '· "Analyse BTCUSDT for swing trading" '
        '· "Execute trade on ETH"</div>',
        unsafe_allow_html=True,
    )

st.divider()

# Chat history
for msg in st.session_state["messages"]:
    role  = msg.get("role", "user")
    text  = msg.get("content", "")
    tokens= msg.get("tokens", 0)

    with st.chat_message(role):
        if role == "assistant":
            st.markdown(text)
            if tokens:
                st.markdown(
                    f'<span class="token-badge">~{tokens} tokens</span>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(text)


# Chat input
if prompt := st.chat_input("Ask anything — e.g. 'Analyse RELIANCE for intraday trade'"):
    # Show user message immediately
    st.session_state["messages"].append({"role": "user", "content": prompt, "tokens": 0})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            data, err = _post("/chat",
                token=st.session_state["token"],
                json={
                    "message":      prompt,
                    "session_id":   st.session_state["session_id"],
                    "llm_provider": st.session_state["llm_provider"],
                    "trade_mode":   st.session_state["trade_mode"],
                },
            )

        if err:
            reply = f"❌ Error: {err}"
            tokens_used = 0
        else:
            reply       = data.get("reply", "No response")
            tokens_used = data.get("tokens_used", 0)
            st.session_state["total_tokens"] = data.get("session_tokens", 0)

        st.markdown(reply)
        if tokens_used:
            st.markdown(
                f'<span class="token-badge">~{tokens_used} tokens this message</span>',
                unsafe_allow_html=True,
            )

        # Show analysis/execution cards if present
        if data and data.get("analysis"):
            a = data["analysis"]
            with st.expander("📊 Full Analysis", expanded=False):
                st.json(a)

        if data and data.get("execution"):
            e = data["execution"]
            with st.expander("⚡ Execution Details", expanded=False):
                st.json(e)

    # Persist message in session state
    st.session_state["messages"].append({
        "role": "assistant",
        "content": reply,
        "tokens": tokens_used,
    })
    st.rerun()
