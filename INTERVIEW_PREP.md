# 🎯 OnboardIQ — Complete Interview Preparation Guide

Everything you need to confidently explain this project in any interview.

---

## 1. 🗣️ THE ELEVATOR PITCH (say this first)

> "I built OnboardIQ — an intelligent onboarding assistant that helps new employees
> instantly answer questions about company policies, benefits, and procedures.
> It uses Agentic RAG with advanced retrieval techniques including Hybrid Search,
> HyDE, Multi-Query, Re-ranking, and Self-RAG, plus a three-layer guardrail system
> for production safety. The entire pipeline runs on free, open-source models."

---

## 2. 🧠 WHAT IS RAG? (explain from scratch)

**RAG = Retrieval-Augmented Generation**

Without RAG, an LLM only knows what it was trained on.
With RAG:
1. You store your own documents in a vector database
2. When a user asks a question, you search those documents
3. You send the relevant chunks + question to the LLM
4. The LLM answers using YOUR documents, not just training data

```
User Question → Search Documents → Retrieve Relevant Chunks → LLM generates Answer
```

**Why RAG over fine-tuning?**
- Fine-tuning is expensive and slow to update
- RAG is cheap, fast, and documents can be updated instantly
- RAG answers are grounded in real documents (less hallucination)

---

## 3. 🤖 WHAT IS AGENTIC RAG?

Basic RAG does a **fixed** single retrieval. Agentic RAG lets the LLM **decide**:
- Should I search? What should I search for?
- Is the result good enough or should I search again?
- Which tool should I use — keyword search or semantic search?

```
Basic RAG:  Question → Retrieve → Answer  (always one retrieval)

Agentic RAG: Question → Agent thinks → Picks tool → Retrieves
                      → Agent thinks → Good enough? → If no: search again
                      → Final Answer
```

**ReAct Pattern** (Reason + Act):
The agent alternates between:
- **Thought**: "I need to find the leave policy"
- **Action**: search_company_docs("annual leave policy")
- **Observation**: [retrieved chunks]
- **Thought**: "I have enough info now"
- **Final Answer**: "You get 21 days of annual leave..."

---

## 4. 🔍 EVERY RETRIEVAL TECHNIQUE EXPLAINED

### 4.1 Vector Search (Semantic Search)
- Converts text to numbers (embeddings) that capture meaning
- Similar meaning = similar numbers = close in vector space
- **Model used**: `all-MiniLM-L6-v2` (runs locally, free)
- **DB**: ChromaDB (local vector database)
- **Limitation**: Misses exact keyword matches

```
"How many days off do I get?" 
→ embedding → [0.23, -0.45, 0.78, ...]
→ find closest vectors in ChromaDB
→ returns semantically similar chunks
```

### 4.2 BM25 (Keyword Search)
- Classic information retrieval algorithm (used by search engines)
- Scores documents based on term frequency and document frequency
- **Strength**: Catches exact matches like "Form 16", "UAN", "EPFO"
- **Limitation**: Misses meaning (can't understand synonyms)

```
"What is UAN?" 
→ BM25 finds chunks that contain the exact word "UAN"
```

### 4.3 Hybrid Search + RRF (Reciprocal Rank Fusion)
- Combines Vector Search + BM25 results
- **RRF Formula**: `score = 1/(k + rank)` for each result list
- A document that ranks high in BOTH lists gets a very high fused score
- Best of both worlds: meaning AND exact terms

```
Vector results: [doc_A, doc_B, doc_C]  
BM25 results:   [doc_B, doc_D, doc_A]
RRF fused:      [doc_B, doc_A, doc_C, doc_D]  ← doc_B ranked high in both
```

### 4.4 HyDE (Hypothetical Document Embeddings)
- **Problem**: Questions and answers exist in different embedding spaces
  - "What is the leave policy?" embeds differently than "Employees get 21 days of leave"
- **Solution**: Generate a fake answer first, then embed THAT
- The fake answer is closer to real document chunks in embedding space

```
User: "How many leaves do I get?"
         ↓
LLM generates: "Employees are entitled to 21 days of paid annual leave..."
         ↓
Embed the hypothetical answer (not the question)
         ↓
Search ChromaDB with that embedding → much better matches
```

### 4.5 Multi-Query Retrieval
- One query might miss relevant documents
- Generate 3 variations of the query, retrieve for each, merge results
- Deduplicates before merging

```
Original: "How do I apply for leave?"
Variation 1: "Leave application process procedure"
Variation 2: "Steps to request time off from work"
Variation 3: "Employee leave management portal instructions"
→ Retrieve for all 4 → deduplicate → much better coverage
```

### 4.6 Cross-Encoder Re-ranking
- First retrieval returns top-20 chunks (fast but approximate)
- Re-ranker scores each (query, chunk) pair together — more accurate
- **Bi-encoder** (used in retrieval): encodes query and doc separately
- **Cross-encoder** (used in re-ranking): encodes query + doc TOGETHER
- Keep only top-5 after re-ranking

```
Query: "sick leave policy"
Retrieved 20 chunks → cross-encoder scores each one jointly with query
→ Top 5 most relevant chunks sent to LLM
```

### 4.7 Contextual Compression
- Retrieved chunks often contain irrelevant sentences
- LLM reads each chunk and extracts ONLY the relevant part
- Reduces noise sent to the final LLM

```
Chunk: "The company was founded in 2010. Employees get 21 days leave. 
        The office is in Mumbai."
Query: "How many leaves?"
Compressed: "Employees get 21 days leave."
```

### 4.8 Parent-Child Chunking
- **Child chunks** (small, ~300 chars): used for retrieval — more precise matching
- **Parent chunks** (large, ~1000 chars): sent to LLM — more context for answering
- Search small → answer with big

```
Parent chunk: Full paragraph about leave policy (1000 chars)
Child chunks: 3 smaller pieces of that paragraph (300 chars each)

Retrieve child chunk → look up its parent_id → send parent to LLM
```

### 4.9 Self-RAG
- LLM generates an answer, then **critiques its own answer**
- If critique says "INSUFFICIENT" → refine query → retrieve more → regenerate
- Loop runs max 2 times to avoid infinite loops

```
Answer generated → "Is this answer complete and grounded?"
→ "INSUFFICIENT: missing information about sick leave"
→ Search again with refined query → better answer
→ "SUFFICIENT" → return final answer
```

---

## 5. 🛡️ GUARDRAILS EXPLAINED

### Why Guardrails?
Production AI systems need safety layers. Without guardrails:
- Users can inject malicious prompts
- LLM might answer questions unrelated to HR
- LLM might hallucinate policies that don't exist

### 5.1 Input Guardrail
Runs BEFORE any retrieval. Checks:
- **Relevance**: Is this an HR/onboarding question? (blocks "write me a poem")
- **Toxicity**: Is the query harmful or abusive?
- **Prompt injection**: Is someone trying to override the system prompt?
- **Query length**: Too short (<5 chars) or too long (>1000 chars)?

### 5.2 Retrieval Guardrail
Runs AFTER retrieval, BEFORE sending to LLM. Checks:
- **Relevance score**: Each chunk must score above threshold (0.3)
- Filters out chunks that are technically retrieved but not actually relevant
- Prevents irrelevant context from confusing the LLM

### 5.3 Output Guardrail
Runs AFTER LLM generates answer. Checks:
- **Groundedness**: Is the answer supported by the retrieved chunks?
- **Hallucination detection**: Did the LLM make up something not in the docs?
- **Toxicity**: Is the output harmful?
- Flags suspicious answers before showing to user

---

## 6. 📊 FULL PIPELINE FLOW

```
User types question
        │
        ▼
┌─────────────────────┐
│   Input Guardrail   │ → Block if toxic/irrelevant/injection
└──────────┬──────────┘
           │
        ▼
┌──────────────────────────────────────────┐
│              ReAct Agent                 │
│                                          │
│  Thought: "I need to find leave policy"  │
│  Action: search_company_docs(query)      │
│                  ↓                       │
│    Query Rewriter → cleaner query        │
│    Multi-Query → 3 variations            │
│    HyDE → hypothetical doc               │
│                  ↓                       │
│    Hybrid Search (Vector + BM25 + RRF)   │
│                  ↓                       │
│    Cross-Encoder Re-ranking (20 → 5)     │
│                  ↓                       │
│    Parent-Child upgrade (small → big)    │
│                  ↓                       │
│  Observation: [retrieved context]        │
│  Thought: "Enough info, can answer"      │
└──────────────────┬───────────────────────┘
                   │
                ▼
┌──────────────────────────┐
│   Retrieval Guardrail    │ → Filter low-relevance chunks
└──────────────┬───────────┘
               │
            ▼
┌──────────────────────────┐
│   LLM generates Answer   │
└──────────────┬───────────┘
               │
            ▼
┌──────────────────────────┐
│     Self-RAG Critique    │ → If insufficient: re-retrieve
└──────────────┬───────────┘
               │
            ▼
┌──────────────────────────┐
│    Output Guardrail      │ → Check groundedness, flag hallucinations
└──────────────┬───────────┘
               │
            ▼
   Final Answer + Sources + Guardrail Report
```

---

## 7. 💬 TRICKY INTERVIEW QUESTIONS + ANSWERS

**Q: Why did you use ChromaDB instead of Pinecone?**
> "ChromaDB runs locally with zero setup — perfect for a portfolio project and demos.
> In production, I'd switch to Pinecone or Weaviate for scalability and managed infrastructure.
> The switch is a one-line config change since we abstracted the vector store."

**Q: What's the difference between a bi-encoder and cross-encoder?**
> "A bi-encoder encodes the query and document independently into vectors, then compares
> them with cosine similarity — fast but less accurate. A cross-encoder takes the query
> and document together as input and scores them jointly — much more accurate but too slow
> to run on thousands of documents. So we use bi-encoder for retrieval (fast, top-20)
> and cross-encoder for re-ranking (accurate, top-5)."

**Q: How do you handle hallucinations?**
> "Three layers: First, we only give the LLM retrieved context — no room to hallucinate
> from training data. Second, the output guardrail checks if the answer is grounded in
> the retrieved chunks. Third, Self-RAG critiques the answer and re-retrieves if the
> answer isn't supported by evidence."

**Q: Why use Parent-Child chunking instead of fixed chunks?**
> "Small chunks are better for retrieval precision — they match the query more tightly.
> But small chunks lack context for the LLM to generate a complete answer.
> Parent-child solves both: retrieve with small chunks, answer with large parent chunks
> that contain the full surrounding context."

**Q: What is RRF and why use it?**
> "Reciprocal Rank Fusion merges two ranked lists by scoring each document as
> the sum of 1/(k+rank) across all lists. Documents appearing high in both
> vector search AND BM25 results get a high fused score. k=60 is a constant
> that prevents the top rank from dominating too much. It's parameter-free
> and consistently outperforms weighted averaging."

**Q: What are the limitations of this system?**
> "A few honest ones: ChromaDB doesn't scale to millions of documents.
> The free Groq model (Llama 8B) is less capable than GPT-4 for complex reasoning.
> The self-RAG loop adds latency. And the contextual compressor makes an extra
> LLM call per chunk which increases cost. In production I'd use async calls
> and caching to address these."

**Q: How would you evaluate this RAG system?**
> "I'd use RAGAS — a RAG evaluation framework that measures:
> - Answer Relevance: Is the answer relevant to the question?
> - Faithfulness: Is the answer grounded in the retrieved context?
> - Context Precision: Are retrieved chunks actually relevant?
> - Context Recall: Did we retrieve all necessary information?
> I'd create a golden Q&A test set from the HR documents and run these metrics."

**Q: How does HyDE help?**
> "Questions and their answers live in different parts of the embedding space.
> 'How many leaves do I get?' embeds very differently from 'Employees receive 21 days
> of paid annual leave per year.' HyDE bridges this gap by generating a hypothetical
> answer document first, then using that embedding to search — because the hypothetical
> answer is much closer to the real document chunks than the original question."

**Q: Why Groq instead of OpenAI?**
> "Groq is completely free and significantly faster — sub-second latency thanks to
> their LPU hardware. For a portfolio project it removes all cost barriers.
> The architecture is provider-agnostic — switching to GPT-4 or Claude requires
> changing just one line in llm.py."

---

## 8. 📈 HOW TO TALK ABOUT EACH FILE

| File | What to say |
|---|---|
| `config.py` | "Single source of truth for all parameters — thresholds, model names, chunk sizes. Makes the system easy to tune." |
| `llm.py` | "LLM factory with caching — any module calls get_llm() so we can swap providers in one place." |
| `ingestor.py` | "Handles PDF, URL, and text ingestion with parent-child chunking strategy." |
| `hybrid_search.py` | "Implements RRF fusion of semantic and keyword search for best-of-both retrieval." |
| `hyde.py` | "Generates hypothetical answers to bridge the query-document embedding gap." |
| `reranker.py` | "Cross-encoder precision scoring — trades speed for accuracy on the shortlist." |
| `self_rag.py` | "Implements the critique loop — the LLM checks its own answer quality." |
| `agent.py` | "Custom ReAct loop — the LLM reasons and acts iteratively rather than one-shot retrieval." |
| `input_guard.py` | "First line of defense — relevance and safety check before any expensive operations." |
| `output_guard.py` | "Final quality gate — groundedness check prevents hallucinated answers reaching users." |

---

## 9. 🏆 KEY NUMBERS TO REMEMBER

| Parameter | Value | Why |
|---|---|---|
| Parent chunk size | 1000 chars | Enough context for LLM |
| Child chunk size | 300 chars | Precise for retrieval |
| Vector retrieval top-k | 20 | Cast wide net |
| After re-ranking | 5 | Keep only best |
| RRF constant k | 60 | Industry standard |
| Self-RAG max iterations | 2 | Balance quality vs latency |
| Relevance threshold | 0.3 | Filter poor chunks |
| Groundedness threshold | 0.5 | Catch hallucinations |

---

## 10. 🎯 ONE-LINE SUMMARIES FOR EACH TECHNIQUE

- **RAG**: Give the LLM your own documents to answer from
- **Agentic RAG**: Let the LLM decide how to search, not just what to return
- **Hybrid Search**: Semantic meaning + exact keywords = better recall
- **HyDE**: Search with a fake answer, not the question
- **Multi-Query**: Ask the same question 3 ways, merge results
- **Re-ranking**: Fast retrieval then slow-but-accurate scoring
- **Parent-Child**: Search small, answer with large
- **Self-RAG**: LLM grades its own homework and re-does it if needed
- **Guardrails**: Safety checks at input, retrieval, and output stages
