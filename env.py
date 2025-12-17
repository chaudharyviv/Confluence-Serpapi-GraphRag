import os
from dotenv import load_dotenv

load_dotenv()

REQUIRED_VARS = [
    "OPENAI_API_KEY",
    "CONFLUENCE_EMAIL",
    "CONFLUENCE_API_TOKEN",
]

def validate_env():
    missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")
