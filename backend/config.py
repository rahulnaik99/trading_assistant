from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to this file's location (project root/backend/../.env)
# so it works regardless of which directory uvicorn is started from
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    OPENAI_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    DEFAULT_LLM_PROVIDER: str = "openai"

    # Delta Exchange
    DELTA_API_KEY: str = ""
    DELTA_API_SECRET: str = ""
    DELTA_REGION: str = "global"

    # Fyers
    FYERS_CLIENT_ID: str = ""
    FYERS_ACCESS_TOKEN: str = ""

    # Tavily
    TAVILY_API_KEY: str = ""

    # MCP servers (spawn as subprocess)
    DELTA_MCP_COMMAND: str = "python"
    DELTA_MCP_ARGS: str = "-m,mcp_servers.delta_server"
    FYERS_MCP_COMMAND: str = "python"
    FYERS_MCP_ARGS: str = "-m,mcp_servers.fyers_server"
    TAVILY_MCP_COMMAND: str = "python"
    TAVILY_MCP_ARGS: str = "-m,mcp_servers.tavily_server"

    # LangSmith
    LANGCHAIN_TRACING_V2: bool = False
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "trade-assistant"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

    # Auth
    SECRET_KEY: str = "change-me-in-production-32-chars-min"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24h

    # API
    API_PORT: int = 8100

    # A2A Agent service URLs
    ANALYSIS_AGENT_URL:  str = "http://localhost:8101"
    EXECUTION_AGENT_URL: str = "http://localhost:8102"
    KRONOS_AGENT_URL:    str = "http://localhost:8103"


settings = Settings()
