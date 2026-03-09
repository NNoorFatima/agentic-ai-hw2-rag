# RAG in the Wild — Case Study Assignment

This assignment is framed as a **case study**: you work with a real-world-style corpus (web search results across multiple domains) and implement four advanced RAG strategies—RAG Fusion, HyDE, CRAG, and Graph RAG—to see which best handles noisy retrieval and varied question types. See **ASSIGNMENT.md** for the full scenario and requirements.

## Requirements

- Python 3.9+
- Node.js 18+ (for the React frontend)

---

## Setup

### Python (backend and pipelines)

```bash
pip install -r requirements.txt
```

Copy `config/config.example.yaml` to `config/config.yaml` and set:

- `dataset_path` — path to `dataset/crag_task_1_and_2_dev_v4.jsonl`
- `embedding_model` — e.g. `all-MiniLM-L6-v2`
- `generation_model` — model name for the LLM you use for answer generation (see below)
- `top_k` — number of chunks to retrieve per query

**LLM / API policy:** **Do not use an OpenAI API key.** Use a **Groq** API key, or a **free** option such as **Google Gemini** (free tier), or another free/local LLM.

Do not commit `config.yaml` if it contains API keys.

### Frontend (React)

```bash
cd frontend
npm install
```

---

## Dataset

This assignment uses the **CRAG Task 1 & 2 dev v4** dataset. 
Download the dataset and place it in the `dataset/` folder yourself.

- **Download (Task 1 & 2, compressed):** [crag_task_1_and_2_dev_v4.jsonl.bz2](https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2)
- Decompress the file (e.g. with 7-Zip or `bzip2 -d`), then put the resulting `crag_task_1_and_2_dev_v4.jsonl` inside the `dataset/` folder.
- **Path after setup:** `dataset/crag_task_1_and_2_dev_v4.jsonl`
- **Format:** One JSON object per line. Fields: `query`, `answer`, `alt_ans`, `search_results` (list of up to 5 items; each has `page_snippet`).
- **Schema:** See `docs/dataset.md`.

All `page_snippet` texts from all rows form the global corpus. Build one embedding index from this corpus; all four pipelines retrieve from it.

---

## Running the project

**Evaluation (run from project root):**

```bash
python run_evaluation.py
```

**Frontend (run from project root):**

```bash
cd frontend
npm run dev
```

Open the URL shown (e.g. http://localhost:3000). You need a backend that loads the index and runs the selected pipeline; the React app will call that backend.

---

## Folder structure

Do not change the folder structure. Required layout and the full case-study description are in `ASSIGNMENT.md`.
