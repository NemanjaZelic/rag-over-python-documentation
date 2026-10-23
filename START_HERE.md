# Polazna tačka za diplomski rad

Ovaj folder je samostalan radni paket za temu **„Evaluacija i optimizacija Retrieval-Augmented Generation sistema nad tehničkom dokumentacijom“**. Sadrži izvršiv RAG sistem, korpus i indekse, golden set, rezultate eksperimenata, grafikone, bibliografiju i proverene beleške za pisanje rada na srpskom jeziku.

## Redosled čitanja

1. `START_HERE.md` — orijentacija, stanje projekta i pravila za tumačenje rezultata.
2. `rag.py` — retrieval, RRF spajanje, odbijanje odgovora, generisanje i CLI.
3. `build.py` — preuzimanje korpusa, fixed-window chunkovanje i izgradnja indeksa.
4. `evaluation/chunk_sizes.py` — Docutils parser, strukturno chunkovanje i eksperiment sa tri veličine.
5. `evaluation/golden_set.jsonl` — 20 evaluacionih pitanja i relevantni dokumenti.
6. `thesis/FACTS.md` — proverene brojke, dozvoljeni zaključci i ograničenja.
7. `thesis/OUTLINE.md` — predlog poglavlja diplomskog rada.
8. `thesis/SOURCES.md` i `thesis/references.bib` — početna bibliografija.
9. `thesis/data/` i `thesis/figures/` — zamrznuti rezultati, inventar i grafikoni.

## Kako se sistem pokreće

Iz korena ovog foldera:

```bash
python3 rag.py setup
python3 rag.py ask "What is the purpose of Python's GIL?"
python3 rag.py ask "What is the purpose of Python's GIL?" --json
python3 rag.py verify
python3 rag.py evaluate
python3 rag.py experiment
```

`ask --json` vraća odgovor, citate i sadržaj korišćenih chunkova. `evaluate` meri postojeći baseline. `experiment` gradi i meri Docutils strukturne varijante 256/32, 512/64 i 768/96.

## Tok podataka

```text
CPython Docs + Python PEPs (zaključani commitovi)
                    |
             RST/TXT dokumenti
                    |
       +------------+-------------+
       |                          |
fixed-window baseline       Docutils eksperiment
512 tokena / overlap 64     struktura + 256/512/768
       |                          |
       +------------+-------------+
                    |
       Jina embedding (512 dim, L2)
           |                     |
  FAISS IndexFlatIP       SQLite FTS5 BM25
           +----------+----------+
                      |
               RRF fusion (k=60)
                      |
       prag za odgovor ili odbijanje
                      |
       Qwen2.5-0.5B odgovor + citati
```

RRF spaja rang-liste po poziciji: za dokument na rangu `r` dodaje `1 / (60 + r)`. Ne sabiraju se sirovi cosine i BM25 skorovi.

## Gde je šta implementirano

- `rag.py`: dense i BM25 pretraga, RRF, prag odbijanja, generator, citati, evaluacija i komandna linija.
- `build.py`: preuzimanje 1.316 izvornih datoteka, `cl100k_base` tokenizacija, fixed-window chunkovi, Jina embeddings, FAISS i SQLite FTS5 indeksi.
- `evaluation/chunk_sizes.py`: parsiranje RST-a pomoću Docutilsa, očuvanje sekcija i blokova, strukturni chunkovi, tri konfiguracije, metrike, latencija i grafikoni.
- `verification/verify.py`: provera integriteta glavnih artefakata.
- `tests/test_rag.py`: automatski testovi ključne logike.
- `data/`: lokalni korpus, modeli, chunkovi, indeksi i rezultati izvršavanja.
- `thesis/evidence/legacy_pipeline/`: istorijski kod koji objašnjava poreklo ranijih bogatih baseline artefakata; nije deo aktivnog runtime-a.

## Korpus i modeli

- CPython dokumentacija, commit `7cd63cddade4c3e1de17c3443695afe1f41d8495`.
- Python PEP repozitorijum, commit `24419b92ae550bf2878716f57c257cba00d3c1a1`.
- 1.316 datoteka i 27.838.432 izvorna bajta.
- Embedding model: `jinaai/jina-embeddings-v2-small-en`, revision `44e7d1d6caec8c883c2d4b207588504d519788d0`.
- Generator: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- Dense indeks: exact FAISS `IndexFlatIP`; zbog L2 normalizacije inner product je jednak kosinusnoj sličnosti.
- Sparse indeks: SQLite FTS5 BM25.

## Glavni rezultati

Baseline sa 13.940 chunkova:

| Metoda | Recall@1 | Recall@5 | MRR |
|---|---:|---:|---:|
| Dense | 0,4375 | 0,9375 | 0,6493 |
| BM25 | 0,5000 | 0,6875 | 0,6024 |
| Hybrid | 0,5000 | 0,9375 | 0,6667 |

Docutils strukturne varijante:

| Veličina/overlap | Chunkovi | Hybrid Recall@1 | Recall@5 | MRR | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| 256/32 | 36.165 | 0,8125 | 0,9375 | 0,8642 | 48,86 ms | 64,59 ms |
| 512/64 | 25.103 | 0,7500 | 0,8750 | 0,8073 | 42,87 ms | 57,68 ms |
| 768/96 | 22.371 | 0,7500 | 0,8750 | 0,8057 | 39,72 ms | 52,38 ms |

Na ovom malom skupu 256/32 je najbolja od tri Docutils varijante po hybrid Recall@1 i MRR, uz najveći indeks i najveću retrieval latenciju.

## Važna granica reproduktivnosti

Aktivni `build.py`, kada `data/` ne postoji, pravi jednostavan baseline direktnim token-window chunkovanjem i postavlja prazan `section_paths`. Postojeći bogati baseline chunkovi u ranijoj fazi projekta bili su napravljeni ručno pisanim RST normalizatorom iz sada arhiviranog pipeline-a. Relevantni istorijski fajlovi kopirani su u `thesis/evidence/legacy_pipeline/` da bi poreklo artefakata bilo proverljivo, ali aktivni kod ih ne uvozi niti poziva.

Docutils strukturni eksperiment je potpuno implementiran u aktivnom `evaluation/chunk_sizes.py` i može se ponoviti komandom `python3 rag.py experiment` kada postoje preuzeti izvori i modeli.

Pri pisanju se zato ne sme tvrditi da baseline i Docutils poređenje izoluje samo veličinu chunka: tu se istovremeno menjaju parser, struktura i tekst koji ulazi u embedding. Poređenje veličine je kontrolisanije samo unutar tri Docutils varijante.

## Istraživačka pitanja koja podaci podržavaju

1. Kako veličina i overlap strukturnih chunkova utiču na Recall@k, MRR, broj chunkova, veličinu indeksa i retrieval latenciju?
2. Kako se dense, BM25 i RRF hybrid retrieval razlikuju na istom korpusu i golden setu?
3. Koji kompromis između retrieval kvaliteta i performansi daju konfiguracije 256/32, 512/64 i 768/96?

## Pravila za buduće pisanje

- Koristiti samo proverene vrednosti iz `thesis/FACTS.md` i JSON/CSV artefakata.
- Ne izmišljati rezultate, hardver, trajanja, citate ni eksperimentalne uslove.
- Brojke tretirati kao preliminarne zbog samo 16 odgovorivih i 4 neodgovoriva pitanja.
- Ne predstavljati accuracy praga 1,0 kao opštu pouzdanost; dobijen je na samo četiri negativna primera.
- Ne tvrditi da je evaluirana tačnost generisanih odgovora ili citata: trenutna evaluacija meri retrieval i answerability gate.
- Latencija ne uključuje query embedding. Build vreme 512/64 uključuje prekid i predstavlja gornju procenu.
- Relevantnost je označena na nivou dokumenta, ne na nivou tačnog dokaznog chunka.
- Sve akademske tvrdnje povezati sa primarnim izvorima iz `thesis/references.bib`.
- Grafikone koristiti iz `thesis/figures/`, a sirove vrednosti proveriti u `thesis/data/experiment_matrix.csv`.

## Šta još treba uraditi pre konačne verzije rada

- potvrditi konačan naslov i istraživačka pitanja sa mentorom;
- proširiti i nezavisno proveriti golden set, zatim odvojiti development i test skup;
- dodati procenu tačnosti odgovora, potkrepljenosti i ispravnosti citata;
- ponoviti clean build sa direktno instrumentisanim vremenima;
- dodati intervale poverenja ili bootstrap analizu;
- zapisati hardver, verzije softvera i završni Git commit.

Na osnovu postojećeg materijala može se početi sa nacrtom rada, ali navedene stavke treba završiti pre jakih zaključaka i konačne predaje.
