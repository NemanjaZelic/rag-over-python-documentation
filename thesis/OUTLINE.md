# Kostur diplomskog rada

## 1. Uvod

- Problem: pronalaženje relevantnog konteksta i generisanje potkrepljenog odgovora nad tehničkom dokumentacijom.
- Cilj: implementirati reproduktivan RAG baseline i eksperimentalno ispitati retrieval konfiguracije.
- Predložena istraživačka pitanja:
  - RQ1: Kako se dense, BM25 i hybrid RRF razlikuju po Recall@k i MRR?
  - RQ2: Kako veličina strukturnog chunka utiče na kvalitet retrievala?
  - RQ3: Kakav je odnos kvaliteta, broja chunkova, veličine indeksa i latencije?
  - RQ4: Koliko pouzdano sistem odbija pitanja izvan korpusa?

## 2. Teorijska osnova

- RAG i podela na retrieval i generation.
- Sparse retrieval i BM25.
- Dense embedding retrieval i kosinusna sličnost.
- FAISS vektorski indeks.
- Reciprocal Rank Fusion.
- Chunkovanje, overlap i strukturne granice.
- Retrieval metrike: Recall@k i MRR.

Izvori: `references.bib`, posebno Lewis et al., Robertson i Zaragoza, Cormack et al., Johnson et al. i Thakur et al.

## 3. Korpus i priprema podataka

- CPython `Doc/` i Python PEP snapshotovi na zaključanim commitovima.
- 1.316 datoteka i 27.838.432 bajta izvornog teksta.
- RST obrada pomoću Docutils-a.
- Pretvaranje Docutils doctree strukture u blokove teksta, koda i sekcija.
- `cl100k_base` za kontrolu veličine chunka.
- Stabilni ID-jevi i izvorni URL-ovi.

Artefakti: `data/system_inventory.json`, izvorni kod `evaluation/chunk_sizes.py`.

## 4. Implementacija baseline sistema

- Fixed 512/64 chunking: 13.940 chunkova.
- Jina Embeddings v2 Small, 512 dimenzija, L2 normalizacija.
- FAISS `IndexFlatIP` kao egzaktna dense pretraga.
- SQLite FTS5/BM25 kao sparse pretraga.
- RRF sa `k=60`.
- Qwen2.5-0.5B-Instruct generator, citati i odbijanje.

Artefakti: `/Users/nemanjazelic/Projects/RAG/rag.py`, `/Users/nemanjazelic/Projects/RAG/build.py`.

## 5. Eksperimentalna metodologija

- Golden set: 20 pitanja, od toga 16 odgovorivih i 4 van domena.
- Relevantnost je trenutno označena na nivou dokumenta.
- Kontrolisane Docutils konfiguracije: 256/32, 512/64 i 768/96.
- Isti korpus, embedding model, retrieveri, RRF i pitanja za sve tri konfiguracije.
- Metrike: Recall@1, Recall@5, MRR, p50/p95 retrieval latencija, broj chunkova i veličina FAISS indeksa.

Artefakti: `data/golden_set.jsonl`, `data/experiment_matrix.csv`.

## 6. Rezultati

- Tabela baseline rezultata.
- Tabela Docutils chunk-size rezultata.
- Grafikon efektivnosti.
- Grafikon vremena i veličine indeksa.
- Analiza pitanja kod kojih metode menjaju rang relevantnog dokumenta.

Artefakti: `data/baseline_results.json`, `data/chunk_size_results.json`, `figures/`.

## 7. Diskusija

- Na malom skupu hybrid 256/32 ima najbolji MRR.
- Manji chunk povećava broj vektora, veličinu indeksa i retrieval latenciju.
- Dense i sparse metode imaju različite profile uspeha.
- Rezultat ne razdvaja efekat Docutils normalizacije, strukturnog chunkovanja i dodavanja naslova u odnosu na stari baseline; poređenje unutar tri Docutils veličine menja samo veličinu i overlap.
- Ograničenja malog golden seta i dokument-level oznaka.

## 8. Zaključak i budući rad

- Cross-encoder reranking.
- Veći i zamrznut test skup.
- Chunk-level relevance oznake.
- Evaluacija generacije, potkrepljenosti i citata.
- Bolje pravilo odbijanja.
- Deduplikacija/MMR i query rewriting.
