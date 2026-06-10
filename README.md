# 🎓 IGCSE Math Pro — AI-Powered Tutoring Ecosystem

> An intelligent, local-first tutoring system for IGCSE Mathematics, combining a deterministic grading engine, Retrieval-Augmented Generation (RAG), and a local LLM to deliver structured, pedagogically sound feedback — without sending student data to the cloud.

---

## ✨ Features

- 🤖 **Local LLM Tutor** — Powered by Llama 3.2 via Ollama; all inference runs on your machine
- 📚 **RAG-Enhanced Hints** — ChromaDB vector store retrieves relevant past exam questions to ground responses
- ✅ **Deterministic Answer Grading** — SymPy-based symbolic validation handles implicit multiplication, sign errors, and edge cases with mathematical precision
- 🔄 **Finite State Machine (FSM) Flow** — Structured tutoring loop: question → attempt → hint → solution, with controlled state transitions
- 🔒 **Answer Leakage Prevention** — Correct answers are regex-masked and stored in Gradio state objects, never exposed in the UI until appropriate
- 📊 **Automated Evaluation Suite** — Metrics report for testing the grading engine against known failure cases

---

## 🏗️ Architecture

```
igcse-math-pro/
├── src/
│   ├── agent/
│   │   └── ui.py               # Gradio UI + FSM controller
│   ├── rag/
│   │   └── vector_store.py     # ChromaDB ingestion & retrieval
│   └── utils/
│       └── evaluate.py         # Automated grading evaluation suite
├── data/
│   └── question_bank/          # Parsed JSON question files
├── requirements.txt
└── README.md
```

**Core pipeline:**
1. A question is sampled and presented via the Gradio UI
2. The student submits an answer; the SymPy engine validates it symbolically
3. If incorrect, the RAG pipeline retrieves similar questions and the LLM generates a targeted hint
4. The FSM controls state transitions (attempt → hint → worked solution) to prevent premature answer reveal

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) installed on your system

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/igcse-math-pro.git
cd igcse-math-pro
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> Ensure `sympy`, `gradio`, `langchain_ollama`, `langchain_core`, and `chromadb` are listed in `requirements.txt`.

### 3. Start the Local LLM

Open a new terminal and pull/run the Llama model:

```bash
ollama run llama3.2:1b
```

### 4. Initialize the Vector Database

With your parsed JSON files ready in `data/question_bank/`:

```bash
python src/rag/vector_store.py
```

### 5. Launch the Application

```bash
python src/agent/ui.py
```

The app will be available at **https://huggingface.co/spaces/wandiya39/IGCSE-Math-Tutor**

---

## 📊 Evaluation & Testing

An automated evaluation suite tests the deterministic grading engine against known failure cases, including:

- Implicit multiplication (e.g., `2x` vs `2*x`)
- Incorrect sign handling
- Non-mathematical or malformed input

To run the metrics report:

```bash
python src/utils/evaluate.py
```

---

## 🛡️ Security & Limitations

| Concern | Approach |
|---|---|
| Answer leakage | Correct answers are regex-masked and stored in Gradio's state, never rendered in the UI prematurely |
| LLM instability | The 1B model may produce formatting errors; fallback UI messages catch and handle generation failures gracefully |
| Data privacy | All LLM inference is local via Ollama — no student data leaves the machine |

---

## 🤝 Contributing

Contributions are welcome! If you'd like to:

- Improve the data extraction pipeline
- Add new syllabus topics or question types
- Enhance the grading engine for more edge cases

Please open an issue or submit a pull request. All contributions should include appropriate test cases in `src/utils/evaluate.py`.

---

## 📄 License

This project is open for educational use. See `LICENSE` for details.

---

*Built with SymPy, ChromaDB, LangChain, Ollama, and Gradio.*
