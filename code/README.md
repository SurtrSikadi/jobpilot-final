# JobPilot Code

Run with one command after installing requirements:

```bash
streamlit run app.py
```

The application uses:

- `kaggle_ingest.py` to stream a real subset from the professor-recommended Techmap/Kaggle jobs dump.
- Lightweight TF-IDF plus deterministic random projection to create dense profile and job embeddings.
- Cosine-similarity nearest-candidate retrieval.
- Country-aware candidate filtering plus a multi-stage ranking score combining embedding similarity, skill overlap, role-family alignment, salary/location preferences, feedback weights, and dealbreaker penalties.
- Simulated streaming ingestion into SQLite with primary-key deduplication.
- Session-level adaptive feedback from accept/reject/skip controls.
