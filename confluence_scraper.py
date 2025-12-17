import os
import requests
from config import BASE_URL, ENGINEERING_SPACE_KEY

def fetch_engineering_pages(limit=50):
    url = f"{BASE_URL}/rest/api/content"
    params = {
        "spaceKey": ENGINEERING_SPACE_KEY,
        "type": "page",
        "expand": "body.storage,version",
        "limit": limit
    }

    r = requests.get(
        url,
        params=params,
        auth=(os.getenv("CONFLUENCE_EMAIL"), os.getenv("CONFLUENCE_API_TOKEN")),
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["results"]
