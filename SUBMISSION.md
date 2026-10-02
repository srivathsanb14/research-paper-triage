<!-- Technical submission package (Project 1). Replace <hf-user> after running scripts/publish_hf.sh. -->

# Interface
**Hugging Face Space:** [Demo Link](https://huggingface.co/spaces/<hf-user>/paper-triage)
*Paper Triage: describe your research in a sentence and ~1,900 recent papers from all fields are sorted into Read / Skim / Skip, each with a one-line reason and "worth-it" signals (peer review, released code or data, study design, citation impact, cautions). Try an example profile, rate a few papers (👍/👎) and watch the ranking learn; label papers in **Label** and see cross-validated accuracy in **Insights**. Runs entirely in the browser; no account. Also on GitHub Pages: https://srivathsanb14.github.io/research-paper-triage/*

# Models
- **Primary Model (trained from scratch):** [Model Card Link](https://huggingface.co/<hf-user>/paper-triage-ranker)
  *Interpretable relevance ranker: seven readable features (semantic similarity to the description, focus and keywords; exact keyword coverage; recency; excluded topics; similarity to rated papers) feed a hand-set profile score and a weighted ridge regression trained per user on their ratings and hand labels. The learned weight is chosen by 5-fold cross-validated average precision and stays 0 unless it beats the profile alone. Outputs Read/Skim/Skip with a reading-time budget. On 500 human-verified labels across three research profiles it reaches AP 0.83–1.00 for Read papers (random order: 0.01–0.03) and surfaces 80% of them within the first 3–6 papers (random: 80–160); cross-validation kept the learned weight at 0% because learning did not beat the profile score on these labels.*
- **Secondary Model (off-the-shelf):** [Model Card Link](https://huggingface.co/<hf-user>/paper-triage-minilm-embeddings)
  *all-MiniLM-L6-v2 sentence embeddings (q8 ONNX, transformers.js), unchanged. Embeds the catalog at build time and the user's interests in a browser Web Worker; cosine similarities are calibrated and used as four of the ranker's features and for near-duplicate grouping. Chosen for size (~23 MB), browser speed and quality on short English text; compared with a TF-IDF space on the same labels.*

# Data
- **Dataset:** [Dataset Link](https://huggingface.co/datasets/<hf-user>/paper-triage-dataset)
  *~1,900 recent papers (OpenAlex, CC0 metadata) across all 26 fields with computed worth-it signals, plus 500 human-verified Read/Skim/Skip labels for three written research profiles (model-proposed, reviewed by a person; provenance recorded per row) and a separate synthetic label file used only for pipeline tests. The card includes collection details, license, ethics notes and EDA. Used to evaluate and tune the ranker.*

# Code Repositories
- **Main Repo:** [GitHub Repo](https://github.com/srivathsanb14/research-paper-triage)
  *Static browser app (`web/`), the Python reference engine and Streamlit server edition (`triage/`, `app.py`), catalog collection and embedding (`scripts/build_catalog.py`, `scripts/embed_catalog.mjs`), dataset building and EDA (`scripts/build_dataset.py`), offline evaluation (`scripts/evaluate_web.mjs`), tests (pytest, node:test, Playwright end-to-end) and the GitHub Actions workflow that refreshes the catalog daily and deploys Pages and the Space. The README has run and reproduce instructions.*
