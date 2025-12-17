import os
import requests
from openai import OpenAI

client = OpenAI()

SERPAPI_URL = "https://serpapi.com/search"

def serpapi_search(query: str, top_k=5):
    params = {
        "engine": "google",
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "num": top_k,
    }

    r = requests.get(SERPAPI_URL, params=params, timeout=20)
    r.raise_for_status()
    return r.json().get("organic_results", [])

def answer_with_serpapi(question: str):
    results = serpapi_search(question)

    if not results:
        return "No relevant public information found.", []

    snippets = []
    sources = []

    for r in results:
        snippet = r.get("snippet")
        link = r.get("link")
        if snippet and link:
            snippets.append(snippet)
            sources.append(link)

    prompt = f"""
You are answering using ONLY the following Google Search results.
Do NOT add any external knowledge.

SEARCH RESULTS:
{chr(10).join(snippets)}

QUESTION:
{question}

ANSWER:
"""

    answer = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    ).choices[0].message.content.strip()

    return answer, list(set(sources))
