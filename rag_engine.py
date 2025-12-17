import json
from openai import OpenAI
from config import (
    BASE_URL,
    ENGINEERING_SPACE_KEY,
    OPENAI_CHAT_MODEL,
    MAX_TOKENS,
)
from embeddings import embed
from vectorstore import collection, load_versions, save_versions
from serpapi_search import google_search
from text_utils import (
    structured_graph_nodes,
    add_semantic_edges,
    enforce_token_limit
)

client = OpenAI()

SYSTEM_RULES = """
You are an Engineering Documentation Assistant.

RULES:
- Use internal engineering documentation ONLY if it explicitly supports the answer
- Use external sources when internal documentation is insufficient
- Never infer missing facts
- Clearly state knowledge gaps
"""

# -------------------------------
# DOMAIN + EVIDENCE INTELLIGENCE
# -------------------------------
ENGINEERING_KEYWORDS = [
    "ceph", "storage", "ontap", "netapp", "dell", "pure",
    "san", "nas", "object", "block", "cluster",
    "replication", "snapshot", "protocol",
    "architecture", "use case", "performance"
]

def is_engineering_question(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ENGINEERING_KEYWORDS)

def extract_answer_requirements(question: str):
    q = question.lower()

    if "protocol" in q or "san" in q:
        return {
            "type": "list",
            "expects": ["iscsi", "fibre channel", "fc", "nvme", "nvme-of", "fcoe"]
        }

    if "how to" in q or "setup" in q or "configure" in q:
        return {"type": "procedural"}

    return {"type": "descriptive"}

def evidence_is_sufficient(docs, requirements):
    if not docs:
        return False

    text = " ".join(docs).lower()

    if requirements["type"] == "list":
        return any(term in text for term in requirements["expects"])

    if requirements["type"] == "procedural":
        # Procedural steps usually not in internal docs
        return False

    return True

# -------------------------------
# INDEXING (GRAPH-BASED)
# -------------------------------
def index_pages(pages, force_reindex=False):
    versions = {} if force_reindex else load_versions()
    indexed = 0

    for page in pages:
        page_id = page["id"]
        title = page["title"]
        version = page["version"]["number"]

        if not force_reindex and versions.get(page_id) == version:
            continue

        html = page["body"]["storage"]["value"]
        page_url = f"{BASE_URL}/spaces/{ENGINEERING_SPACE_KEY}/pages/{page_id}"

        nodes = structured_graph_nodes(html, title)
        nodes = add_semantic_edges(nodes)
        nodes = enforce_token_limit(nodes, MAX_TOKENS, OPENAI_CHAT_MODEL)

        for i, node in enumerate(nodes):
            text = f"{title} | {node['properties']['section']}\n{node['properties']['text']}"

            collection.add(
                documents=[text],
                embeddings=[embed(text)],
                metadatas=[{
                    "page_id": page_id,
                    "title": title,
                    "section": node["properties"]["section"],
                    "url": page_url,
                    "space": ENGINEERING_SPACE_KEY,
                    "edges": json.dumps(node["edges"])
                }],
                ids=[f"{page_id}_{version}_{i}"]
            )
            indexed += 1

        versions[page_id] = version

    save_versions(versions)
    return indexed

# -------------------------------
# QUESTION ANSWERING
# -------------------------------
def ask_engineering(question, debug=False):
    debug_info = {}

    # 0️⃣ Domain Gate
    if not is_engineering_question(question):
        return (
            "⚠️ **Out of Scope Question**\n\n"
            "This assistant answers engineering and storage-related questions only.",
            [],
            "none",
            debug_info if debug else None
        )

    # 1️⃣ Local Retrieval
    q_embed = embed(question)
    res = collection.query(
        query_embeddings=[q_embed],
        n_results=4,
        where={"space": ENGINEERING_SPACE_KEY},
        include=["documents", "metadatas"]
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    debug_info["local_chunks"] = docs
    debug_info["local_sources"] = [m.get("url") for m in metas]

    requirements = extract_answer_requirements(question)

    # 2️⃣ Local RAG (Only if sufficient)
    if evidence_is_sufficient(docs, requirements):
        context = "\n\n---\n\n".join(docs)

        prompt = f"""
{SYSTEM_RULES}

ENGINEERING CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER:
"""

        answer = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        ).choices[0].message.content.strip()

        sources = sorted(set(m.get("url") for m in metas if m.get("url")))
        return answer, sources, "local", debug_info if debug else None

    # 3️⃣ External Search (SerpAPI)
    google_results = google_search(question)
    debug_info["google_results"] = google_results

    if google_results:
        ext_context = "\n\n".join(f"{g['title']}\n{g['snippet']}" for g in google_results)
        ext_sources = [g["link"] for g in google_results]

        prompt = f"""
Use the EXTERNAL CONTEXT below.
Do NOT hallucinate.

EXTERNAL CONTEXT:
{ext_context}

QUESTION:
{question}

FINAL ANSWER:
"""

        answer = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        ).choices[0].message.content.strip()

        return answer, ext_sources, "external", debug_info if debug else None

    # 4️⃣ Knowledge Gap
    return (
        "⚠️ **Knowledge Gap Detected**\n\n"
        "No reliable internal or external information was found.",
        [],
        "none",
        debug_info if debug else None
    )
