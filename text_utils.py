from bs4 import BeautifulSoup
import numpy as np
import tiktoken
from embeddings import embed


# ============================================================
# 1️⃣ STRUCTURE → GRAPH NODES
# ============================================================
def structured_graph_nodes(html: str, page_title: str):
    soup = BeautifulSoup(html, "html.parser")

    nodes = []
    current_section = "General"
    buffer = []

    def flush():
        if buffer:
            node_id = f"{page_title}:{len(nodes)}"
            nodes.append({
                "node_id": node_id,
                "label": "Section",
                "properties": {
                    "page": page_title,
                    "section": current_section,
                    "text": "\n".join(buffer)
                },
                "edges": []
            })

    for el in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        if el.name in ["h1", "h2", "h3"]:
            flush()
            current_section = el.get_text(strip=True)
            buffer = []
        else:
            txt = el.get_text(strip=True)
            if txt:
                buffer.append(txt)

    flush()

    # Sequential structure edges
    for i in range(1, len(nodes)):
        nodes[i]["edges"].append({
            "type": "FOLLOWS",
            "target": nodes[i - 1]["node_id"]
        })

    return nodes


# ============================================================
# 2️⃣ SEMANTIC GRAPH EDGES
# ============================================================
def add_semantic_edges(nodes, similarity_threshold=0.85):
    if len(nodes) < 2:
        return nodes

    vectors = [embed(n["properties"]["text"]) for n in nodes]

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            v1, v2 = vectors[i], vectors[j]
            sim = np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2)
            )
            if sim >= similarity_threshold:
                nodes[i]["edges"].append({
                    "type": "RELATED_TO",
                    "target": nodes[j]["node_id"],
                    "score": round(sim, 3)
                })

    return nodes


# ============================================================
# 3️⃣ TOKEN LIMIT SAFETY
# ============================================================
def enforce_token_limit(nodes, max_tokens, model):
    enc = tiktoken.encoding_for_model(model)
    final_nodes = []

    for node in nodes:
        tokens = enc.encode(node["properties"]["text"])
        if len(tokens) <= max_tokens:
            final_nodes.append(node)
        else:
            for i in range(0, len(tokens), max_tokens):
                sub_text = enc.decode(tokens[i:i + max_tokens])
                final_nodes.append({
                    "node_id": f"{node['node_id']}_part{i}",
                    "label": node["label"],
                    "properties": {
                        **node["properties"],
                        "text": sub_text
                    },
                    "edges": node["edges"]
                })

    return final_nodes
