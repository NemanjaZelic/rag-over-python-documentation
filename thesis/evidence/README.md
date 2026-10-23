# Istorijski dokaz o baseline pipeline-u

Direktorijum `legacy_pipeline/` sadrži minimalni skup starijih skripti i konfiguracija koji objašnjava kako su u ranijoj fazi projekta nastajali normalizovani RST dokumenti i bogati chunkovi.

Ovi fajlovi su priloženi samo radi sledljivosti metodologije. Aktivni `rag.py` i `build.py` ih ne uvoze, ne pozivaju i ne zavise od njih. Ako se obriše ceo `data/` i pokrene `python3 rag.py setup`, aktivni `build.py` pravi jednostavne fixed-window chunkove direktno iz RST/TXT teksta. Ponovljivi Docutils strukturni pipeline nalazi se u `evaluation/chunk_sizes.py`.

Relevantni istorijski fajlovi:

- `legacy_pipeline/normalize_corpus.py` — ručno pisana RST normalizacija i metadata.
- `legacy_pipeline/chunk_corpus.py` — formiranje bogatih chunkova iz normalizovanih blokova.
- `legacy_pipeline/corpus_sources.py` — zajedničke definicije izvora i putanja.
- `legacy_pipeline/config/` — zaključane konfiguracije tog pipeline-a.

Ovo razdvajanje sprečava da se istorijski kod pogrešno predstavi kao deo trenutnog minimalnog proizvoda.
