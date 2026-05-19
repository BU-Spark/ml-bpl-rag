# Lessons Learned
---

## Overview

This document captures technical, architectural, and process lessons learned during Spring 2026.

---

## 1. RAG Is a Systems Problem, Not a Model Problem

### What Happened
Early in the semester we assumed that upgrading the embedding model from `sentence-transformers/all-mpnet-base-v2` to `BAAI/bge-m3` would be a straightforward improvement. In practice, a single model swap triggered a cascade of required changes across the entire pipeline:

- Re-embedding all ~400K BPL documents from scratch
- Rebuilding all FAISS indexes (title index + full article chunk index)
- Regenerating sparse vectors (BGE-M3 produces these; E5 did not)
- Retuning the confidence gate threshold
- Re-running the full evaluation suite to establish a new baseline

### What We Learned
There are no isolated improvements in a RAG pipeline. Every component such as embedding model, chunk size, index type, confidence threshold, reranker weights is interconnected. Changing one affects all the others. Optimizing a single component in isolation produces misleading results.

### Recommendation for Future Teams
- **Change one thing at a time.** Lock every other component before varying the one you're testing. Run your full evaluation suite after each change before moving on.
- **Never delete a working index** until its replacement is fully tested and evaluated.

---

## 2. Build Your Evaluation Dataset First

### What Happened
The Fall 2025 team identified the lack of a gold standard evaluation dataset as their biggest gap at handoff. We inherited this gap. For the first several weeks of Spring 2026, we were building and improving the pipeline without a consistent way to measure whether changes were actually helping or hurting. We did not have a reliable evaluation baseline until the 50-query ground truth dataset was finalized mid-semester.

### What We Learned
Without evaluation data you are flying blind. It is impossible to make confident architectural decisions without numbers that tell you whether each change improved or degraded performance. Should we lower the threshold? Switch models? Change chunk size? 

---

## 3. Historical OCR Text Is Harder Than Modern Text

### What Happened
19th and early 20th century newspaper OCR text has systematic corruption that modern embedding models were not trained to handle:

- Character substitutions: `rn` misread as `m`, `l` misread as `1`, `|` misread as `I`
- Broken column layouts producing fragmented mid-word line breaks
- Archaic spelling: `publick`, `colour`, `despatches`
- Period-specific vocabulary and Latin phrases outside modern model training distributions
- Inconsistent naming of people, places, and regiments across issues

This means a query for *"molasses flood"* may fail to retrieve an article that OCR rendered as *"rnolasses f1ood"* — semantically identical content but unrecognizable to both sparse keyword search and dense vector retrieval.

### What We Learned
NER-based reranking is a practical mitigation for OCR noise. Named entities — proper nouns like place names, people, and dates — survive OCR corruption far better than common words. Scoring retrieval candidates by named entity overlap between query and document provides a retrieval signal that is stable even when surrounding text is heavily degraded.

### Recommendation for Future Teams
- Keep the NER reranking layer — it directly addresses OCR noise and should not be removed.

---

## 4. Compute Infrastructure Is Part of the Research

### What Happened
Embedding 400K+ documents is not feasible on a laptop. On CPU, embedding the Boston Traveler/Transcript subset alone would take 10–14 hours. On SCC GPU it takes 1–2 hours. Beyond speed, SCC home directory disk quotas (10GB) fill up quickly when pip installs CUDA packages and HuggingFace downloads model weights locally.


### What We Learned
SCC is essential for this project since embedding and indexing at BPL's scale cannot be done on a laptop. Getting SCC access and configuration sorted out in Week 1 is vital and it is a prerequisite for doing any meaningful work on the pipeline.

---

## 5. Understanding Prior Work Before Adding New Components

### What Happened
This is a multi-semester continuation project. We inherited code from Fall 2024, Spring 2025, and Fall 2025 teams where each of which made architectural decisions that were not always documented. Time spent reverse-engineering prior decisions (why pgVector instead of Pinecone, why the bronze-silver-gold pipeline, why the two-LLM architecture) was time not spent building new features.

### What We Learned
Reading the client meeting notes from prior semesters is as valuable as reading the code. The meeting notes in the project description document exactly what was tried, what failed, what the client asked for, and why specific decisions were made. This context is not in the codebase.

### Recommendation for Future Teams
Do not skip the onboarding. Understanding why decisions were made such as  which models were tried, what the client asked for, what failed and why is just as important as understanding the code itself. That context lives in the meeting notes and prior READMEs, not in the codebase.

---