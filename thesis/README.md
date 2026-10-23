# Materijal za diplomski rad — RAG evaluacija i optimizacija

Ovaj direktorijum sadrži proverene rezultate i izvore za početak pisanja. Nije tekst diplomskog rada.

## Početna tačka

Najpre pročitati `../START_HERE.md`, a zatim:

1. `OUTLINE.md` — predložena struktura rada i veza sa artefaktima.
2. `FACTS.md` — proverene činjenice, rezultati i ograničenja koja se smeju tvrditi.
3. `SOURCES.md` i `references.bib` — početna bibliografija primarnih izvora.
4. `data/system_inventory.json` — modeli, verzije, commitovi i konfiguracije.
5. `data/experiment_matrix.csv` — rezultati u tabelarnom obliku.
6. `figures/` — grafikoni spremni za rad u PNG i SVG formatu.
7. `evidence/` — istorijski kod uključen samo radi sledljivosti porekla baseline artefakata.

## Reprodukcija

```bash
cd /Users/nemanjazelic/Projects/RAG
python3 rag.py setup
python3 rag.py verify
python3 rag.py evaluate
python3 rag.py experiment
```

`evaluate` meri postojeći fixed-window baseline. `experiment` meri Docutils strukturne varijante 256/32, 512/64 i 768/96.

## Pre početka konačnog pisanja

- dogovoriti sa mentorom konačna istraživačka pitanja;
- proširiti golden set sa 20 na najmanje 100–150 pitanja;
- odvojiti development i test deo pre podešavanja parametara;
- ponoviti clean build sa instrumentisanim `perf_counter` vremenima;
- dodati evaluaciju generisanih odgovora i citata;
- izračunati intervale poverenja;
- sačuvati konačni Git commit rada i hash svih rezultata.
