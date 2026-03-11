"""
generation.py  —  LLM answer generation
-----------------------------------------
Supported providers: groq (free), openai, anthropic, local (extractive fallback)
Get a FREE Groq key at https://console.groq.com
"""

import os, re
from typing import List, Tuple, Optional


def _load_config():
    try:
        import yaml
        if os.path.exists("config/config.yaml"):
            with open("config/config.yaml") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


def _resolve_key(api_key):
    return (api_key
            or os.environ.get("GROQ_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY"))


def _build_context(retrieved: List[Tuple[str, float, dict]], max_chunks: int = 10) -> str:
    lines = []
    for i, (text, score, meta) in enumerate(retrieved[:max_chunks], 1):
        name = meta.get("source_name", "")
        lines.append(f"[{i}] {name}")
        lines.append(text.strip())
        lines.append("")
    return "\n".join(lines)


_SYSTEM = (
    "You are a precise factual question-answering assistant.\n"
    "Rules:\n"
    "1. Read the numbered context passages carefully.\n"
    "2. Answer using ONLY information found in those passages.\n"
    "3. Be SHORT and DIRECT — give just the key fact: a name, date, number, or short phrase.\n"
    "4. Do NOT repeat the question. Do NOT write long paragraphs.\n"
    "5. If the answer truly cannot be found in the passages, output: I don't know\n"
    "6. End your answer with the citation number, e.g. [1] or [2].\n"
    "\n"
    "Good answer examples:\n"
    "  Q: Who directed Inception?                          → Christopher Nolan [4]\n"
    "  Q: When was the Eiffel Tower built?                 → 1889 [2]\n"
    "  Q: Which country has the largest population?        → China [1]\n"
    "  Q: How many Grand Slams has Federer won?            → 20 [3]\n"
)


def generate_answer(
    query: str,
    retrieved: List[Tuple[str, float, dict]],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 100,
    temperature: float = 0.0,
) -> str:
    cfg     = _load_config()
    llm_cfg = cfg.get("llm", {})

    provider = provider or llm_cfg.get("provider", "groq")
    model    = model    or llm_cfg.get("model",    "llama3-70b-8192")
    api_key  = _resolve_key(api_key or llm_cfg.get("api_key"))

    if not api_key or provider == "local":
        return _local_fallback(query, retrieved)

    context = _build_context(retrieved)
    user_msg = (
        f"Context passages:\n{context}\n"
        f"Question: {query}\n"
        f"Short direct answer (just the key fact + citation):"
    )

    try:
        if provider == "groq":
            return _call_groq(user_msg, model, api_key, max_tokens, temperature)
        elif provider == "openai":
            return _call_openai(user_msg, model, api_key, max_tokens, temperature)
        elif provider == "anthropic":
            return _call_anthropic(user_msg, model, api_key, max_tokens, temperature)
        else:
            return _local_fallback(query, retrieved)
    except Exception as exc:
        print(f"[generation] LLM failed ({exc}), using extractive fallback.")
        return _local_fallback(query, retrieved)


def _call_groq(msg, model, key, max_tokens, temp):
    from openai import OpenAI
    c = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    r = c.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":_SYSTEM},{"role":"user","content":msg}],
        max_tokens=max_tokens, temperature=temp,
    )
    return r.choices[0].message.content.strip()


def _call_openai(msg, model, key, max_tokens, temp):
    from openai import OpenAI
    c = OpenAI(api_key=key)
    r = c.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":_SYSTEM},{"role":"user","content":msg}],
        max_tokens=max_tokens, temperature=temp,
    )
    return r.choices[0].message.content.strip()


def _call_anthropic(msg, model, key, max_tokens, temp):
    import anthropic
    c = anthropic.Anthropic(api_key=key)
    r = c.messages.create(
        model=model, max_tokens=max_tokens, temperature=temp,
        system=_SYSTEM,
        messages=[{"role":"user","content":msg}],
    )
    return r.content[0].text.strip()


def _local_fallback(query: str, retrieved: List[Tuple[str, float, dict]]) -> str:
    """
    Extractive fallback when no API key is set.
    Scans every sentence across all retrieved chunks and picks the one
    with maximum keyword + named-entity overlap with the query.
    """
    if not retrieved:
        return "I don't know."

    # Query keywords (ignore stopwords)
    q_words = set(re.findall(r'\b\w{3,}\b', query.lower())) - {
        "who","what","when","where","which","how","why","did","does",
        "was","were","has","have","been","the","that","this","with",
        "from","more","than","are","can","its","and","for","not"
    }

    best_sent, best_score, best_url = None, -1.0, ""

    for text, chunk_score, meta in retrieved:
        url = meta.get("source_url", "")
        # Split into sentences
        for sent in re.split(r'(?<=[.!?])\s+', text.strip()):
            sent = sent.strip()
            if len(sent) < 15:
                continue
            s_words  = set(re.findall(r'\b\w{3,}\b', sent.lower()))
            kw_hits  = len(q_words & s_words)
            # Named-entity bonus (capitalised multi-word phrases)
            entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sent)
            ent_bonus = min(len(entities), 4) * 0.5
            score     = kw_hits + ent_bonus + chunk_score * 1.5
            if score > best_score:
                best_score = score
                best_sent  = sent
                best_url   = url

    if best_sent:
        return f"{best_sent} [Source: {best_url}]"
    top_text, _, top_meta = retrieved[0]
    return f"{top_text[:200].strip()} [Source: {top_meta.get('source_url','')}]"