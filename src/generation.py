"""
generation.py
-------------
LLM answer generation from (query, retrieved_context).

Supported providers
-------------------
- "groq"      : FREE - fast Llama 3 (get key at https://console.groq.com)
- "openai"    : GPT models
- "anthropic" : Claude models
- "local"     : Extractive fallback, no API needed
"""

import os
import re
from typing import List, Tuple, Optional


def _load_config():
    try:
        import yaml
        cfg_path = "config/config.yaml"
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


def _build_context_str(retrieved: List[Tuple[str, float, dict]], max_chunks: int = 10) -> str:
    """Format retrieved chunks into a numbered context block."""
    lines = []
    for i, (text, score, meta) in enumerate(retrieved[:max_chunks], 1):
        url  = meta.get("source_url", "unknown")
        name = meta.get("source_name", "")
        lines.append(f"[{i}] {name}")
        lines.append(text.strip())
        lines.append("")
    return "\n".join(lines)


def _system_prompt() -> str:
    return (
        "You are a factual question-answering assistant. "
        "You will be given numbered context passages and a question. "
        "Rules:\n"
        "1. Answer using ONLY the information in the context passages.\n"
        "2. Be SHORT and DIRECT. Give just the key fact: a name, date, number, or brief phrase.\n"
        "3. Do NOT repeat the question. Do NOT write full sentences unless necessary.\n"
        "4. If the answer is not in the context, output exactly: I don't know\n"
        "5. End with the citation number like [1] or [2].\n"
        "Examples of good answers:\n"
        "  Q: Who directed Inception? → Christopher Nolan [4]\n"
        "  Q: When was the Eiffel Tower built? → 1889 [2]\n"
        "  Q: Which country has the largest population? → China [1]\n"
    )


def generate_answer(
    query: str,
    retrieved: List[Tuple[str, float, dict]],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 128,
    temperature: float = 0.0,
) -> str:
    cfg     = _load_config()
    llm_cfg = cfg.get("llm", {})

    provider = provider or llm_cfg.get("provider", "groq")
    model    = model    or llm_cfg.get("model", "llama3-70b-8192")
    api_key  = (api_key
                or llm_cfg.get("api_key")
                or os.environ.get("GROQ_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("ANTHROPIC_API_KEY"))

    if not api_key or provider == "local":
        return _local_fallback(query, retrieved)

    context      = _build_context_str(retrieved)
    user_message = (
        f"Context passages:\n{context}\n"
        f"Question: {query}\n"
        f"Short answer:"
    )

    try:
        if provider == "groq":
            return _generate_groq(user_message, model, api_key, max_tokens, temperature)
        elif provider == "openai":
            return _generate_openai(user_message, model, api_key, max_tokens, temperature)
        elif provider == "anthropic":
            return _generate_anthropic(user_message, model, api_key, max_tokens, temperature)
        else:
            return _local_fallback(query, retrieved)
    except Exception as exc:
        print(f"[generation] LLM call failed: {exc}. Using local fallback.")
        return _local_fallback(query, retrieved)


def _generate_groq(user_message, model, api_key, max_tokens, temperature):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _system_prompt()},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def _generate_openai(user_message, model, api_key, max_tokens, temperature):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _system_prompt()},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def _generate_anthropic(user_message, model, api_key, max_tokens, temperature):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=_system_prompt(),
        messages=[{"role": "user", "content": user_message}],
    )
    return resp.content[0].text.strip()


def _local_fallback(query: str, retrieved: List[Tuple[str, float, dict]]) -> str:
    """
    Extractive fallback: scans every sentence of every retrieved chunk and
    returns the sentence whose words best overlap with the query keywords.
    This is used when no LLM API key is available.
    """
    if not retrieved:
        return "I don't know."

    # Strip stopwords and short words from query to get keywords
    q_words = set(re.findall(r'\b\w{3,}\b', query.lower())) - {
        "who", "what", "when", "where", "which", "how", "why", "did",
        "does", "was", "were", "has", "have", "been", "the", "that",
        "this", "with", "from", "more", "than", "are", "can", "its"
    }

    best_sent  = None
    best_val   = -1
    best_url   = ""

    for text, chunk_score, meta in retrieved:
        url       = meta.get("source_url", "unknown")
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 15:
                continue
            s_words = set(re.findall(r'\b\w{3,}\b', sent.lower()))
            overlap = len(q_words & s_words)
            # Reward sentences that contain key entities (capitalised words)
            entities = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', sent)
            entity_bonus = min(len(entities), 3)
            score = overlap + entity_bonus + chunk_score * 1.5
            if score > best_val:
                best_val  = score
                best_sent = sent
                best_url  = url

    if best_sent:
        return f"{best_sent} [Source: {best_url}]"

    top_text, _, top_meta = retrieved[0]
    return f"{top_text[:200].strip()} [Source: {top_meta.get('source_url', '')}]"