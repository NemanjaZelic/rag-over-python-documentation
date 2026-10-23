# Proverene činjenice i ograničenja

## Sistem

- Korpus: CPython dokumentacija i Python PEP dokumenti sa dva zaključana Git commita.
- Izvorni korpus: 1.316 datoteka, 27.838.432 bajta.
- Baseline: 13.940 fixed-window chunkova, najviše 512 `cl100k_base` tokena, overlap 64.
- Eksperiment: Docutils strukturni chunkovi sa konfiguracijama 256/32, 512/64 i 768/96.
- Embedding: `jinaai/jina-embeddings-v2-small-en`, 512 dimenzija, L2 normalizacija.
- Dense indeks: FAISS `IndexFlatIP`; zbog normalizacije inner product odgovara kosinusnoj sličnosti.
- Sparse indeks: SQLite FTS5 BM25 nad tekstom, naslovom i sekcijama.
- Fusion: RRF, `k=60`; sabiraju se recipročne funkcije rangova, ne sirovi BM25 i cosine skorovi.
- Generator: `Qwen/Qwen2.5-0.5B-Instruct`.

## Baseline rezultat

| Metoda | Recall@1 | Recall@5 | MRR |
|---|---:|---:|---:|
| Dense | 0,4375 | 0,9375 | 0,6493 |
| BM25 | 0,5000 | 0,6875 | 0,6024 |
| Hybrid | 0,5000 | 0,9375 | 0,6667 |

Prag za odgovor/odbijanje je pogodio svih 20 primera, ali samo četiri pitanja su van domena. Rezultat 1,0 ne sme se predstavljati kao dokaz opšte pouzdanosti.

## Docutils eksperiment

| Konfiguracija | Chunkovi | FAISS | Hybrid Recall@1 | Recall@5 | MRR | Hybrid p50 | Hybrid p95 | Prvi build raspon |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 256/32 | 36.165 | 74.065.965 B | 0,8125 | 0,9375 | 0,8642 | 48,86 ms | 64,59 ms | 11,0 min |
| 512/64 | 25.103 | 51.410.989 B | 0,7500 | 0,8750 | 0,8073 | 42,87 ms | 57,68 ms | 13,5 min |
| 768/96 | 22.371 | 45.815.853 B | 0,7500 | 0,8750 | 0,8057 | 39,72 ms | 52,38 ms | 11,5 min |

## Šta rezultat podržava

- Na ovom skupu i unutar tri Docutils varijante, 256/32 ima najbolji hybrid Recall@1 i MRR.
- Smanjenje veličine chunka povećalo je broj chunkova, FAISS indeks i retrieval latenciju.
- Hybrid je imao veći MRR od pojedinačnih dense i BM25 metoda u sve tri Docutils konfiguracije.

## Šta rezultat ne podržava

- Ne dokazuje da je 256 optimalno na drugim korpusima ili većem skupu pitanja.
- Ne meri automatski tačnost generisanih odgovora.
- Ne meri potkrepljenost tvrdnji niti preciznost citata.
- Ne izoluje Docutils parser naspram starog parsera kada se eksperimentalne varijante porede sa starim baseline-om.
- Ne meri query embedding u prikazanoj retrieval latenciji.
- Prva build vremena su izvedena iz timestamp raspona artefakata; 512 run je uključivao prekid i zato je gornja procena, ne čisto vreme računanja.

## Pretnje validnosti

- Samo 16 odgovorivih pitanja: jedno pitanje menja Recall za 0,0625.
- Relevantnost je označena na nivou dokumenta, ne dokaznog chunka.
- Pitanja je pravio autor sistema, pa je moguć selection bias.
- Nema odvojenog development/test skupa.
- Nema intervala poverenja ni testa statističke značajnosti.
- Latencija je merena na jednom računaru i zavisi od cache-a i sistemskog opterećenja.
