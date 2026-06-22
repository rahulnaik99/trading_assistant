"""Trade Assistant — FastAPI backend."""

import logging
import logging.handlers
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

from backend.agents.orchestrator import OrchestratorAgent
from backend.auth.store import (
    create_access_token, create_session, create_user,
    decode_token, init_db, list_sessions, load_messages,
    save_message, session_total_tokens, verify_user,
)
from backend.config import settings

# ── Logging ───────────────────────────────────────────────────────────────────
_LOG_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
root = logging.getLogger()
root.setLevel(logging.INFO)

_console = logging.StreamHandler()
_console.setFormatter(logging.Formatter(_LOG_FMT, datefmt="%H:%M:%S"))
root.addHandler(_console)

Path("data/logs").mkdir(parents=True, exist_ok=True)
_file_h = logging.handlers.RotatingFileHandler(
    "data/logs/trade_assistant.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file_h.setFormatter(logging.Formatter(_LOG_FMT, datefmt="%Y-%m-%d %H:%M:%S"))
root.addHandler(_file_h)

for _noisy in ("httpcore", "httpx", "urllib3", "langsmith", "hpack"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ── Lifespan ──────────────────────────────────────────────────────────────────
_orchestrators: dict[str, OrchestratorAgent] = {}  # provider → agent


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_db()
    logger.info("Trade Assistant API started — DB initialised")
    yield
    logger.info("Trade Assistant API shutdown")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Trade Assistant API",
    description="Multi-agent AI trading assistant — analyse, plan, execute via MCP",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ── Auth dependency ───────────────────────────────────────────────────────────
def current_user(token: str = Depends(oauth2_scheme)) -> dict:
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    return payload


def _get_orchestrator(provider: str = "openai", mode: str = "paper") -> OrchestratorAgent:
    key = f"{provider}:{mode}"
    if key not in _orchestrators:
        _orchestrators[key] = OrchestratorAgent(llm_provider=provider, mode=mode)
    return _orchestrators[key]


# ── Pydantic models ───────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    message: str
    session_id: str
    llm_provider: str = "openai"
    trade_mode: str = "paper"

class NewSessionRequest(BaseModel):
    pass


# ── Auth endpoints ────────────────────────────────────────────────────────────
@app.post("/auth/register", tags=["auth"])
async def register(req: RegisterRequest):
    try:
        uid = create_user(req.username, req.password)
        return {"status": "ok", "uid": uid, "username": req.username}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auth/login", tags=["auth"])
async def login(form: OAuth2PasswordRequestForm = Depends()):
    user = verify_user(form.username, form.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token(user["uid"], user["username"])
    logger.info("auth: login  username=%s", form.username)
    return {"access_token": token, "token_type": "bearer", "username": form.username}


# ── Session endpoints ─────────────────────────────────────────────────────────
@app.post("/sessions/new", tags=["sessions"])
async def new_session(user: dict = Depends(current_user)):
    sid = create_session(user["sub"])
    return {"session_id": sid}


@app.get("/sessions", tags=["sessions"])
async def get_sessions(user: dict = Depends(current_user)):
    return {"sessions": list_sessions(user["sub"])}


@app.get("/sessions/{session_id}/messages", tags=["sessions"])
async def get_messages(session_id: str, user: dict = Depends(current_user)):
    msgs = load_messages(session_id)
    tokens = session_total_tokens(session_id)
    return {"messages": msgs, "total_tokens": tokens}


# ── Chat endpoint ─────────────────────────────────────────────────────────────
@app.post("/chat", tags=["chat"])
async def chat(req: ChatRequest, user: dict = Depends(current_user)):
    """Main chat endpoint — routes to analysis/execution agents as needed."""
    logger.info("chat: user=%s  session=%s  provider=%s  mode=%s  msg=%s",
                user.get("username"), req.session_id,
                req.llm_provider, req.trade_mode, req.message[:60])
    logger.info("chat: OPENAI_API_KEY set=%s  key_prefix=%s",
                bool(settings.OPENAI_API_KEY),
                settings.OPENAI_API_KEY[:12] + "..." if settings.OPENAI_API_KEY else "EMPTY")

    # Load conversation history for context
    history = load_messages(req.session_id)

    # Get orchestrator for this LLM provider + mode
    orch = _get_orchestrator(req.llm_provider, req.trade_mode)

    # Run
    result = await orch.handle_message(
        user_message=req.message,
        history=history,
    )

    reply = result.get("reply", "")

    # Count tokens
    import tiktoken
    try:
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
        user_tokens  = len(enc.encode(req.message))
        reply_tokens = len(enc.encode(reply))
    except Exception:
        user_tokens  = len(req.message.split()) * 2
        reply_tokens = len(reply.split()) * 2

    # Persist
    save_message(req.session_id, "user",      req.message, user_tokens)
    save_message(req.session_id, "assistant", reply,       reply_tokens)

    total_tokens = session_total_tokens(req.session_id)

    return {
        "reply":        reply,
        "intent":       result.get("intent"),
        "symbol":       result.get("symbol"),
        "analysis":     result.get("analysis"),
        "execution":    result.get("execution"),
        "tokens_used":  user_tokens + reply_tokens,
        "session_tokens": total_tokens,
    }


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok", "version": "1.0.0"}


# ── Fyers token management ────────────────────────────────────────────────────

@app.get("/fyers/login-url", tags=["fyers"])
async def fyers_login_url():
    """Get the Fyers OAuth login URL. Open it in browser to get auth code."""
    redirect = "https://www.google.com"
    url = (
        f"https://api.fyers.in/api/v2/generate-authcode"
        f"?client_id={settings.FYERS_CLIENT_ID}"
        f"&redirect_uri={redirect}"
        f"&response_type=code"
        f"&state=trade_assistant"
    )
    return {
        "login_url":   url,
        "client_id":   settings.FYERS_CLIENT_ID,
        "redirect_uri": redirect,
        "instructions": [
            "1. Open the login_url in your browser",
            "2. Login with your Fyers credentials",
            "3. After redirect, copy the 'code' parameter from the URL",
            "4. POST to /fyers/generate-token with {auth_code: 'your-code'}",
        ],
    }


class FyersTokenRequest(BaseModel):
    auth_code: str = ""
    model_config = {
        "json_schema_extra": {
            "example": {"auth_code": "paste-the-code-from-redirect-url-here"}
        }
    }


@app.post("/fyers/generate-token", tags=["fyers"])
async def fyers_generate_token(req: FyersTokenRequest):
    """Exchange Fyers auth code for access token and update .env automatically."""
    import hashlib
    import requests as _req

    if not req.auth_code:
        raise HTTPException(422, "auth_code is required")

    app_secret = settings.FYERS_SECRET_KEY
    if not app_secret:
        raise HTTPException(400,
            "FYERS_SECRET_KEY not set in .env — add it to generate tokens")

    # Build checksum: SHA256(client_id:secret_key:auth_code)
    checksum_str = f"{settings.FYERS_CLIENT_ID}:{app_secret}:{req.auth_code}"
    checksum = hashlib.sha256(checksum_str.encode()).hexdigest()

    payload = {
        "grant_type":  "authorization_code",
        "appIdHash":   checksum,
        "code":        req.auth_code,
    }
    resp = _req.post(
        "https://api-t2.fyers.in/api/v3/token",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=15,
    )
    data = resp.json()
    logger.info("fyers_generate_token: response s=%s  code=%s", data.get("s"), data.get("code"))

    if data.get("s") != "ok":
        raise HTTPException(400, f"Fyers token generation failed: {data.get('message', data)}")

    access_token = data.get("access_token", "")
    if not access_token:
        raise HTTPException(400, f"No access_token in response: {data}")

    # Auto-update .env file
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        env_txt = env_path.read_text(encoding="utf-8")
        import re as _re
        env_txt = _re.sub(
            r"FYERS_ACCESS_TOKEN=.*",
            f"FYERS_ACCESS_TOKEN={access_token}",
            env_txt,
        )
        if "FYERS_ACCESS_TOKEN=" not in env_txt:
            env_txt += f"\nFYERS_ACCESS_TOKEN={access_token}\n"
        env_path.write_text(env_txt, encoding="utf-8")
        logger.info("fyers_generate_token: .env updated with new access_token")

    # Also update live settings so current process uses new token immediately
    settings.FYERS_ACCESS_TOKEN = access_token

    return {
        "status":         "ok",
        "message":        ".env updated. Restart services to apply everywhere.",
        "token_preview":  access_token[:20] + "...",
        "token_length":   len(access_token),
    }


@app.get("/fyers/auth-status", tags=["fyers"])
async def fyers_auth_status():
    """Check if current Fyers access token is valid."""
    import base64, json as _json
    token = settings.FYERS_ACCESS_TOKEN
    if not token:
        return {"authenticated": False, "reason": "FYERS_ACCESS_TOKEN not set in .env"}

    # Decode JWT expiry without network call
    try:
        parts = token.split(".")
        if len(parts) == 3:
            payload_b64 = parts[1] + "=" * (4 - len(parts[1]) % 4)
            payload = _json.loads(base64.b64decode(payload_b64))
            from datetime import datetime, timezone
            exp = payload.get("exp", 0)
            exp_dt = datetime.fromtimestamp(exp, tz=timezone.utc)
            now    = datetime.now(timezone.utc)
            expired = now > exp_dt
            return {
                "authenticated": not expired,
                "fy_id":         payload.get("fy_id", "?"),
                "expires_at":    exp_dt.strftime("%Y-%m-%d %H:%M UTC"),
                "expired":       expired,
                "reason":        "Token expired" if expired else "Token valid",
            }
    except Exception as exc:
        logger.warning("fyers_auth_status: JWT decode failed — %s", exc)

    # Fallback: live API check
    from backend.brokers.fyers.source import FyersSource
    fs = FyersSource(client_id=settings.FYERS_CLIENT_ID, access_token=token)
    is_auth, msg = fs.check_auth()
    return {"authenticated": is_auth, "reason": msg}


@app.get("/logs/download", tags=["meta"], response_model=None)
async def download_logs(user: dict = Depends(current_user)):
    from fastapi.responses import FileResponse
    p = Path("data/logs/trade_assistant.log")
    if not p.exists():
        raise HTTPException(404, "Log file not found")
    return FileResponse(str(p), media_type="text/plain", filename="trade_assistant.log")
