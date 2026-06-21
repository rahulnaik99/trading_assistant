"""Disable LangSmith pytest plugin during tests — prevents hanging on mocked LLM calls."""
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_API_KEY"] = ""

# Remove langsmith plugin if loaded
import sys
for key in list(sys.modules.keys()):
    if "langsmith" in key and "pytest" in key:
        del sys.modules[key]
