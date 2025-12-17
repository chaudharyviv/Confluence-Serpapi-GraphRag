import os
import requests

SERPAPI_URL = "https://serpapi.com/search.json"

def google_search(query, num_results=5):
    params = {
        "engine": "google",
        "q": query,
        "num": num_results,
        "api_key": os.getenv("SERPAPI_API_KEY")
    }

    r = requests.get(SERPAPI_URL, params=params, timeout=20)
    r.raise_for_status()

    data = r.json()
    results = []

    for item in data.get("organic_results", []):
        results.append({
            "title": item.get("title"),
            "snippet": item.get("snippet"),
            "link": item.get("link")
        })

    return results
