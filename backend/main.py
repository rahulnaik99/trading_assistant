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
    logger.info("chat: user=%s  session=%s  msg=%s",
                user.get("username"), req.session_id, req.message[:60])

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


@app.get("/logs/download", tags=["meta"], response_model=None)
async def download_logs(user: dict = Depends(current_user)):
    from fastapi.responses import FileResponse
    p = Path("data/logs/trade_assistant.log")
    if not p.exists():
        raise HTTPException(404, "Log file not found")
    return FileResponse(str(p), media_type="text/plain", filename="trade_assistant.log")
