from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SKILL_VOCAB = [
    "python", "sql", "r", "tableau", "power bi", "spark", "pyspark",
    "kafka", "kubernetes", "docker", "aws", "gcp", "azure", "scikit-learn",
    "pytorch", "tensorflow", "nlp", "computer vision", "deep learning",
    "machine learning", "mlops", "airflow", "dbt", "snowflake", "excel",
    "statistics", "experimentation", "feature engineering", "microservices",
    "java", "c++", "fastapi", "streamlit", "llm", "rag", "data modeling",
    "sales", "marketing", "finance", "operations", "healthcare", "human resources",
    "project management", "customer success", "design", "teaching", "legal",
    "communication", "problem solving", "stakeholder management",
]


SENIOR_PATTERNS = re.compile(r"\b(senior|sr\.?|staff|principal|lead)\b", re.I)
JUNIOR_PATTERNS = re.compile(r"\b(junior|entry level|associate|new grad)\b", re.I)
CONTRACT_PATTERNS = re.compile(r"\b(contract|contractor|temporary|temp|unpaid)\b", re.I)
YEARS_PATTERN = re.compile(r"(\d+)\+?\s*(?:years|yrs)", re.I)
COUNTRY_ALIASES = {
    "us": "United States",
    "usa": "United States",
    "united states": "United States",
    "uk": "United Kingdom",
    "gb": "United Kingdom",
    "united kingdom": "United Kingdom",
    "england": "United Kingdom",
    "de": "Germany",
    "germany": "Germany",
    "ca": "Canada",
    "canada": "Canada",
    "au": "Australia",
    "australia": "Australia",
    "in": "India",
    "india": "India",
    "sg": "Singapore",
    "singapore": "Singapore",
    "hk": "Hong Kong",
    "hong kong": "Hong Kong",
    "ie": "Ireland",
    "ireland": "Ireland",
    "nz": "New Zealand",
    "new zealand": "New Zealand",
    "ch": "Switzerland",
    "switzerland": "Switzerland",
    "nl": "Netherlands",
    "netherlands": "Netherlands",
    "fr": "France",
    "france": "France",
    "se": "Sweden",
    "sweden": "Sweden",
    "jp": "Japan",
    "japan": "Japan",
    "kr": "South Korea",
    "south korea": "South Korea",
    "cn": "China",
    "china": "China",
    "ua": "Ukraine",
    "ukraine": "Ukraine",
    "ph": "Philippines",
    "philippines": "Philippines",
    "at": "Austria",
    "austria": "Austria",
    "lu": "Luxembourg",
    "luxembourg": "Luxembourg",
    "cz": "Czech Republic",
    "czech republic": "Czech Republic",
    "ru": "Russia",
    "russia": "Russia",
    "my": "Malaysia",
    "malaysia": "Malaysia",
    "id": "Indonesia",
    "indonesia": "Indonesia",
    "gr": "Greece",
    "greece": "Greece",
    "tw": "Taiwan",
    "taiwan": "Taiwan",
    "pt": "Portugal",
    "portugal": "Portugal",
    "be": "Belgium",
    "belgium": "Belgium",
    "ae": "United Arab Emirates",
    "united arab emirates": "United Arab Emirates",
    "th": "Thailand",
    "thailand": "Thailand",
    "es": "Spain",
    "spain": "Spain",
    "vn": "Vietnam",
    "vietnam": "Vietnam",
    "it": "Italy",
    "italy": "Italy",
}


@dataclass
class Profile:
    name: str
    background: str
    skills: list[str]
    target_roles: list[str]
    location: str = "United States"
    min_salary: int = 0
    dealbreakers: list[str] | None = None
    preferred_countries: list[str] | None = None
    visa_required: bool = False
    prefer_remote: bool = True

    @property
    def text(self) -> str:
        return " ".join([
            self.background,
            " ".join(self.skills),
            " ".join(self.target_roles),
            self.location,
            " ".join(self.preferred_countries or []),
            " ".join(self.dealbreakers or []),
        ])


PERSONAS = {
    "Aisha - ML Engineer Pivoter": Profile(
        name="Aisha",
        background="Data Analyst with 3 years in retail analytics. Wants to pivot to ML Engineering.",
        skills=["Python", "SQL", "pandas", "scikit-learn", "basic PyTorch"],
        target_roles=["ML Engineer", "Applied Scientist", "Data Scientist"],
        dealbreakers=["No Senior or Staff titles", "No defense companies", "No 5+ years ML required"],
        preferred_countries=["United States", "Canada"],
        prefer_remote=True,
    ),
    "Marcus - New Grad Broad Search": Profile(
        name="Marcus",
        background="Recent UC Davis MSBA graduate with two analytics internships.",
        skills=["Python", "R", "SQL", "Tableau", "PySpark", "basic NLP"],
        target_roles=["Data Analyst", "BI Analyst", "Junior Data Scientist", "Analytics Engineer"],
        dealbreakers=["No 3+ years required", "No unpaid roles", "No contract-only roles"],
        preferred_countries=["United States", "Canada", "United Kingdom"],
        prefer_remote=True,
    ),
    "Priya - ML Infrastructure": Profile(
        name="Priya",
        background="Senior software engineer with 7 years in fintech moving into ML/AI infrastructure.",
        skills=["Java", "Python", "Kubernetes", "microservices", "Kafka", "Spark", "TensorFlow", "AWS"],
        target_roles=["ML Platform Engineer", "MLOps Engineer", "Senior ML Engineer"],
        dealbreakers=["No Junior titles", "No companies under 100 employees", "US only"],
        preferred_countries=["United States"],
        min_salary=130000,
        prefer_remote=True,
    ),
    "Kenji - Visa-Constrained AI Search": Profile(
        name="Kenji",
        background="MS Computer Science student graduating on OPT. Published research in NLP and computer vision.",
        skills=["Python", "C++", "PyTorch", "NLP", "computer vision", "deep learning"],
        target_roles=["Research Scientist", "ML Engineer", "Applied Scientist", "AI Engineer"],
        dealbreakers=["No contract or temporary roles", "Requires H-1B sponsorship"],
        preferred_countries=["United States", "Canada"],
        visa_required=True,
        prefer_remote=False,
    ),
    "Sofia - UK Product Analytics": Profile(
        name="Sofia",
        background="Product analyst in Madrid with 4 years of SaaS experimentation experience, looking for UK or Ireland analytics roles.",
        skills=["SQL", "Python", "Tableau", "experimentation", "statistics", "data modeling"],
        target_roles=["Product Analyst", "Data Analyst", "Analytics Engineer"],
        location="United Kingdom, Ireland",
        dealbreakers=["No contract-only roles", "No roles requiring security clearance"],
        preferred_countries=["United Kingdom", "Ireland"],
        min_salary=70000,
        prefer_remote=True,
    ),
    "Lukas - Germany Data Engineer": Profile(
        name="Lukas",
        background="Backend engineer in Berlin moving into data engineering and analytics platforms.",
        skills=["Python", "SQL", "Spark", "Kafka", "Docker", "AWS"],
        target_roles=["Data Engineer", "Analytics Engineer", "Platform Engineer"],
        location="Germany, Switzerland, Netherlands",
        dealbreakers=["No junior titles", "No unpaid roles", "No contract-only roles"],
        preferred_countries=["Germany", "Switzerland", "Netherlands"],
        min_salary=85000,
        prefer_remote=True,
    ),
    "Mei - Singapore BI Analyst": Profile(
        name="Mei",
        background="Marketing operations analyst in Singapore seeking BI or analytics roles in Southeast Asia.",
        skills=["SQL", "Power BI", "Excel", "Python", "data modeling"],
        target_roles=["BI Analyst", "Data Analyst", "Business Intelligence Analyst"],
        location="Singapore, Hong Kong, Australia",
        dealbreakers=["No 5+ years required", "No contract-only roles"],
        preferred_countries=["Singapore", "Hong Kong", "Australia"],
        min_salary=60000,
        prefer_remote=False,
    ),
    "Arjun - India MLOps Engineer": Profile(
        name="Arjun",
        background="Cloud engineer in Bengaluru with Kubernetes and Python experience, targeting MLOps roles.",
        skills=["Python", "Kubernetes", "Docker", "AWS", "Spark", "MLOps"],
        target_roles=["MLOps Engineer", "ML Platform Engineer", "Cloud Data Engineer"],
        location="India, Singapore",
        dealbreakers=["No junior titles", "No unpaid roles", "No contract-only roles"],
        preferred_countries=["India", "Singapore"],
        min_salary=50000,
        prefer_remote=True,
    ),
    "Nora - Canada Healthcare Data Scientist": Profile(
        name="Nora",
        background="Healthcare analyst in Toronto with Python and statistics experience, looking for regulated-industry data science.",
        skills=["Python", "SQL", "statistics", "scikit-learn", "Tableau"],
        target_roles=["Data Scientist", "Healthcare Data Scientist", "Machine Learning Analyst"],
        location="Canada",
        dealbreakers=["No defense companies", "No contract-only roles"],
        preferred_countries=["Canada"],
        min_salary=80000,
        prefer_remote=True,
    ),
    "Elena - UK Marketing Manager": Profile(
        name="Elena",
        background="Brand marketing specialist with five years in consumer products, seeking campaign and growth marketing roles.",
        skills=["marketing", "project management", "communication", "stakeholder management"],
        target_roles=["Marketing Manager", "Growth Marketing Manager", "Brand Manager"],
        location="United Kingdom, Ireland",
        dealbreakers=["No unpaid roles", "No contract-only roles"],
        preferred_countries=["United Kingdom", "Ireland"],
        min_salary=65000,
        prefer_remote=True,
    ),
    "David - Germany Finance Analyst": Profile(
        name="David",
        background="Corporate finance associate with budgeting, forecasting, and reporting experience.",
        skills=["finance", "Excel", "SQL", "stakeholder management"],
        target_roles=["Financial Analyst", "FP&A Analyst", "Finance Business Partner"],
        location="Germany, Switzerland",
        dealbreakers=["No sales roles", "No contract-only roles"],
        preferred_countries=["Germany", "Switzerland"],
        min_salary=75000,
        prefer_remote=False,
    ),
    "Leila - UAE Healthcare Operations": Profile(
        name="Leila",
        background="Hospital operations coordinator seeking healthcare operations or patient-services management roles.",
        skills=["healthcare", "operations", "project management", "communication"],
        target_roles=["Healthcare Operations Manager", "Clinical Operations Analyst", "Patient Services Manager"],
        location="United Arab Emirates, United Kingdom",
        dealbreakers=["No night-shift-only roles", "No contract-only roles"],
        preferred_countries=["United Arab Emirates", "United Kingdom"],
        min_salary=65000,
        prefer_remote=False,
    ),
    "Maya - International HR Recruiter": Profile(
        name="Maya",
        background="Recruiting coordinator with HR operations experience, open to talent acquisition roles in India, the UK, or Vietnam.",
        skills=["human resources", "communication", "stakeholder management", "project management"],
        target_roles=["Recruiter", "Talent Acquisition Specialist", "HR Generalist"],
        location="India, United Kingdom, Vietnam",
        dealbreakers=["No commission-only roles", "No contract-only roles"],
        preferred_countries=["India", "United Kingdom", "Vietnam"],
        min_salary=55000,
        prefer_remote=True,
    ),
    "Omar - Canada Sales Account Executive": Profile(
        name="Omar",
        background="B2B sales representative with SaaS prospecting and account-management experience.",
        skills=["sales", "customer success", "communication", "stakeholder management"],
        target_roles=["Account Executive", "Sales Manager", "Customer Success Manager"],
        location="Canada, United States",
        dealbreakers=["No unpaid roles", "No door-to-door sales", "No contract-only roles"],
        preferred_countries=["Canada", "United States"],
        min_salary=60000,
        prefer_remote=True,
    ),
}


def normalize_country(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    key = raw.lower().strip()
    return COUNTRY_ALIASES.get(key, raw)


def parse_country_preferences(profile: Profile) -> set[str]:
    countries = profile.preferred_countries or []
    if not countries and profile.location:
        countries = re.split(r"[,;/|]", profile.location)
    normalized = {normalize_country(c) for c in countries if normalize_country(c)}
    if any(c.lower() in {"any", "global", "worldwide", "remote"} for c in normalized):
        return set()
    return normalized


def normalize_skill(skill: str) -> str:
    return skill.lower().strip()


def extract_skills(text: str) -> list[str]:
    text_l = text.lower()
    found = []
    for skill in SKILL_VOCAB:
        if re.search(rf"\b{re.escape(skill)}\b", text_l):
            found.append(skill)
    return sorted(set(found))


def max_years_required(text: str) -> int:
    years = [int(m.group(1)) for m in YEARS_PATTERN.finditer(text or "")]
    return max(years) if years else 0


def load_jobs(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return prepare_jobs(df)


def prepare_jobs(df: pd.DataFrame) -> pd.DataFrame:
    if "country" not in df.columns:
        df["country"] = df["location"].fillna("").apply(lambda loc: normalize_country(str(loc).split(",")[-1]))
    df["country"] = df["country"].fillna("").apply(normalize_country)
    if "country_code" not in df.columns:
        df["country_code"] = df["country"].str.lower()
    df["skills_list"] = df["skills"].fillna("").apply(lambda s: [x.strip().lower() for x in s.split(";") if x.strip()])
    df["job_text"] = (
        df["title"].fillna("") + " " + df["company"].fillna("") + " " +
        df["description"].fillna("") + " " + df["skills"].fillna("") + " " +
        df["location"].fillna("")
    )
    df["years_required"] = df["description"].apply(max_years_required)
    return df


class SimpleEmbeddingModel:
    def __init__(self, max_features: int = 1800, n_components: int = 96):
        self.max_features = max_features
        self.n_components = n_components
        self.vocab_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None
        self.components_: np.ndarray | None = None

    @staticmethod
    def _tokens(text: str) -> list[str]:
        words = re.findall(r"[a-zA-Z][a-zA-Z+#.-]{1,}", (text or "").lower())
        bigrams = [f"{a} {b}" for a, b in zip(words, words[1:])]
        return words + bigrams

    def _term_matrix(self, texts: list[str], fit: bool = False) -> np.ndarray:
        docs = [self._tokens(t) for t in texts]
        if fit:
            counts: dict[str, int] = {}
            docfreq: dict[str, int] = {}
            for toks in docs:
                seen = set()
                for tok in toks:
                    counts[tok] = counts.get(tok, 0) + 1
                    if tok not in seen:
                        docfreq[tok] = docfreq.get(tok, 0) + 1
                        seen.add(tok)
            features = sorted(
                [tok for tok, df in docfreq.items() if df >= 2],
                key=lambda tok: counts[tok],
                reverse=True,
            )[: self.max_features]
            self.vocab_ = {tok: i for i, tok in enumerate(features)}
            n = max(1, len(docs))
            self.idf_ = np.array([np.log((1 + n) / (1 + docfreq[tok])) + 1 for tok in features])

        matrix = np.zeros((len(texts), len(self.vocab_)), dtype=float)
        for row, toks in enumerate(docs):
            for tok in toks:
                col = self.vocab_.get(tok)
                if col is not None:
                    matrix[row, col] += 1.0
        if self.idf_ is not None and matrix.size:
            matrix *= self.idf_
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1, norms)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        tfidf = self._term_matrix(texts, fit=True)
        if tfidf.shape[1] == 0:
            self.components_ = np.zeros((0, 0))
            return tfidf
        rng = np.random.default_rng(423)
        k = min(self.n_components, tfidf.shape[1])
        self.components_ = rng.normal(0, 1 / np.sqrt(k), size=(tfidf.shape[1], k))
        dense = tfidf @ self.components_
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        return dense / np.where(norms == 0, 1, norms)

    def transform(self, texts: list[str]) -> np.ndarray:
        tfidf = self._term_matrix(texts, fit=False)
        if self.components_ is None or self.components_.size == 0:
            return tfidf
        dense = tfidf @ self.components_
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        return dense / np.where(norms == 0, 1, norms)


def fit_embeddings(jobs: pd.DataFrame):
    corpus = jobs["job_text"].tolist()
    model = SimpleEmbeddingModel()
    job_vectors = model.fit_transform(corpus)
    return model, job_vectors, None


def _dealbreaker_penalty(job: pd.Series, profile: Profile) -> tuple[float, list[str]]:
    penalty = 0.0
    flags = []
    title_desc = f"{job.title} {job.description}"
    if any("senior" in d.lower() or "staff" in d.lower() for d in profile.dealbreakers or []):
        if SENIOR_PATTERNS.search(title_desc):
            penalty += 0.35
            flags.append("senior/staff title conflicts with target level")
        if job.years_required >= 5:
            penalty += 0.25
            flags.append("requires 5+ years")
    if any("3+" in d.lower() for d in profile.dealbreakers or []):
        if job.years_required >= 3:
            penalty += 0.30
            flags.append("requires 3+ years")
    if any("contract" in d.lower() or "temp" in d.lower() for d in profile.dealbreakers or []):
        if CONTRACT_PATTERNS.search(title_desc) or str(job.employment_type).lower() == "contract":
            penalty += 0.30
            flags.append("contract/temp conflicts with preference")
    if any("junior" in d.lower() for d in profile.dealbreakers or []):
        if JUNIOR_PATTERNS.search(title_desc):
            penalty += 0.25
            flags.append("junior role is under-target")
    if any("100" in d.lower() for d in profile.dealbreakers or []):
        if int(job.company_size) < 100:
            penalty += 0.25
            flags.append("company appears below 100 employees")
    if profile.visa_required:
        text_l = f"{job.title} {job.description}".lower()
        if "no sponsorship" in text_l or "unable to sponsor" in text_l:
            penalty += 0.30
            flags.append("posting says sponsorship is unavailable")
        elif bool(job.sponsors_visa):
            penalty -= 0.06
    if profile.min_salary and int(job.salary_min) < profile.min_salary:
        penalty += 0.12
        flags.append("salary below preference")
    return penalty, flags


def rank_jobs(
    jobs: pd.DataFrame,
    profile: Profile,
    model,
    job_vectors: np.ndarray,
    nn,
    feedback_weights: dict[str, float] | None = None,
    top_k: int = 25,
) -> pd.DataFrame:
    feedback_weights = feedback_weights or {}
    profile_vector = model.transform([profile.text])
    preferred_countries = parse_country_preferences(profile)
    candidate_pool = jobs
    pool_vectors = job_vectors
    if preferred_countries:
        mask = jobs["country"].isin(preferred_countries).to_numpy()
        if mask.sum() >= max(top_k, 20):
            candidate_pool = jobs.loc[mask]
            pool_vectors = job_vectors[mask]
    n_neighbors = min(max(250, top_k * 8), len(candidate_pool))
    all_sim = (pool_vectors @ profile_vector.T).ravel()
    candidate_idx = np.argsort(all_sim)[::-1][:n_neighbors]
    sim = all_sim[candidate_idx]
    candidates = candidate_pool.iloc[candidate_idx].copy()
    candidates["embedding_score"] = sim

    profile_skills = {normalize_skill(s) for s in profile.skills}
    target_text = " ".join(profile.target_roles).lower()
    rows = []
    for _, job in candidates.iterrows():
        job_skills = set(job.skills_list)
        overlap = len(profile_skills & job_skills) / max(1, len(job_skills))
        target_bonus = 0.16 if any(role.lower().split()[0] in job.title.lower() for role in profile.target_roles) else 0
        location_bonus = 0.0
        if preferred_countries and normalize_country(job.country) in preferred_countries:
            location_bonus += 0.16
        if profile.prefer_remote and str(job.remote).lower() == "true":
            location_bonus += 0.05
        salary_bonus = min(0.08, max(0, (int(job.salary_max) - 90000) / 500000))
        feedback_bonus = sum(feedback_weights.get(skill, 0) for skill in job_skills) / max(1, len(job_skills))
        penalty, flags = _dealbreaker_penalty(job, profile)
        score = (
            0.56 * float(job.embedding_score) +
            0.26 * overlap +
            target_bonus +
            location_bonus +
            salary_bonus +
            feedback_bonus -
            penalty
        )
        if any(t in job.job_text.lower() for t in target_text.split()):
            score += 0.02
        rows.append((job.job_id, overlap, target_bonus, location_bonus, salary_bonus, feedback_bonus, penalty, flags, score))

    scored = candidates.merge(
        pd.DataFrame(rows, columns=[
            "job_id", "skill_overlap", "target_bonus", "location_bonus", "salary_bonus",
            "feedback_bonus", "dealbreaker_penalty", "dealbreaker_flags", "match_score"
        ]),
        on="job_id",
    )
    return scored.sort_values("match_score", ascending=False).head(top_k).reset_index(drop=True)


def explain_job(job: pd.Series, profile: Profile) -> str:
    profile_skills = {normalize_skill(s) for s in profile.skills}
    matched = sorted(profile_skills & set(job.skills_list))
    missing = sorted(set(job.skills_list) - profile_skills)[:5]
    parts = [
        f"Dense-vector similarity: {job.embedding_score:.2f}",
        f"Skill overlap: {job.skill_overlap:.0%}" + (f" ({', '.join(matched[:6])})" if matched else ""),
    ]
    if job.target_bonus > 0:
        parts.append("Title aligns with target role family")
    if job.location_bonus > 0:
        parts.append(f"Location fit: {job.country}" + (" / remote-friendly" if str(job.remote).lower() == "true" else ""))
    if job.dealbreaker_flags:
        parts.append("Caution: " + "; ".join(job.dealbreaker_flags))
    if missing:
        parts.append("Skills to tailor or learn: " + ", ".join(missing))
    return " | ".join(parts)


def update_feedback_weights(weights: dict[str, float], job: pd.Series, action: str) -> dict[str, float]:
    delta = {"accept": 0.035, "reject": -0.045, "skip": -0.005}.get(action, 0.0)
    updated = dict(weights)
    for skill in job.skills_list:
        updated[skill] = float(np.clip(updated.get(skill, 0.0) + delta, -0.20, 0.20))
    return updated


def tailored_resume(profile: Profile, job: pd.Series) -> str:
    matched = sorted({normalize_skill(s) for s in profile.skills} & set(job.skills_list))
    missing = sorted(set(job.skills_list) - {normalize_skill(s) for s in profile.skills})[:4]
    bullets = [
        f"Built analytics and machine-learning workflows using {', '.join(matched[:4]) or 'Python and SQL'} for business decision support.",
        f"Translated stakeholder needs into measurable model and dashboard outputs relevant to {job.title} responsibilities.",
        "Prepared clean, reproducible datasets and communicated findings through concise technical documentation.",
    ]
    if "kafka" in job.skills_list or "spark" in job.skills_list:
        bullets.append("Positioned distributed data processing experience as a foundation for production ML infrastructure.")
    if profile.name == "Kenji":
        bullets.insert(0, "Led applied AI research projects in NLP/computer vision with publication-ready experimentation.")
    if profile.name == "Marcus":
        bullets.insert(0, "Recent MSBA graduate with internships spanning analytics, reporting automation, and data storytelling.")
    if profile.name == "Aisha":
        bullets.insert(0, "Reframed retail analytics experience around model development, Python workflows, and ML experimentation.")
    return "\n".join([
        f"{profile.name} - Tailored Resume Summary",
        "",
        f"Target role: {job.title} at {job.company}",
        f"Summary: {profile.background}",
        "",
        "Selected Skills: " + ", ".join(profile.skills),
        "Job Keywords to Mirror: " + ", ".join(job.skills_list[:8]),
        "",
        "Experience Bullets:",
        *[f"- {b}" for b in bullets],
        "",
        "ATS Tailoring Notes:",
        f"- Emphasize matched skills: {', '.join(matched) if matched else 'role-relevant analytics and ML skills'}.",
        f"- Address gaps carefully: {', '.join(missing) if missing else 'no major keyword gaps detected'}.",
        "- Keep dealbreaker-sensitive wording honest; do not claim unearned production experience.",
    ])


def stream_to_sqlite(csv_path: str | Path, db_path: str | Path, batch_size: int = 75) -> dict[str, int]:
    jobs = pd.read_csv(csv_path)
    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)
    existing_cols = [
        row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
    ] if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'").fetchone() else []
    if existing_cols and "country" not in existing_cols:
        conn.execute("DROP TABLE jobs")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            title TEXT, company TEXT, location TEXT, country TEXT, country_code TEXT,
            salary_min INTEGER, salary_max INTEGER,
            skills TEXT, description TEXT, apply_link TEXT, employment_type TEXT,
            remote INTEGER, sponsors_visa INTEGER, company_size INTEGER, source TEXT
        )
        """
    )
    inserted = 0
    duplicates = 0
    def as_bool(value) -> int:
        return int(str(value).strip().lower() in {"true", "1", "yes", "y"})

    for start in range(0, len(jobs), batch_size):
        batch = jobs.iloc[start:start + batch_size]
        for _, row in batch.iterrows():
            try:
                conn.execute(
                    "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        row.job_id, row.title, row.company, row.location,
                        getattr(row, "country", ""), getattr(row, "country_code", ""),
                        int(row.salary_min), int(row.salary_max), row.skills, row.description, row.apply_link,
                        row.employment_type, as_bool(row.remote), as_bool(row.sponsors_visa),
                        int(row.company_size), row.source,
                    ),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                duplicates += 1
        conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    conn.close()
    return {"inserted": inserted, "duplicates": duplicates, "stored": total}


def ensure_sqlite_store(csv_path: str | Path, db_path: str | Path) -> dict[str, int | str]:
    csv_path = Path(csv_path)
    db_path = Path(db_path)
    expected = sum(1 for _ in csv_path.open("r", encoding="utf-8", errors="ignore")) - 1
    needs_rebuild = not db_path.exists()
    if not needs_rebuild:
        try:
            conn = sqlite3.connect(db_path)
            row = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()
            existing = int(row[0]) if row else 0
            conn.close()
            needs_rebuild = existing != expected
        except sqlite3.Error:
            needs_rebuild = True

    if needs_rebuild:
        if db_path.exists():
            db_path.unlink()
        result = stream_to_sqlite(csv_path, db_path)
        result["mode"] = "rebuilt"
        return result

    return {"inserted": 0, "duplicates": 0, "stored": expected, "mode": "existing"}


def load_jobs_from_sqlite(db_path: str | Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM jobs", conn)
    conn.close()
    for col in ["remote", "sponsors_visa"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return prepare_jobs(df)


def analytics(jobs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    skill_rows = []
    for skills in jobs["skills_list"]:
        skill_rows.extend(skills)
    top_skills = pd.Series(skill_rows).value_counts().head(15).reset_index()
    top_skills.columns = ["skill", "count"]
    salary = jobs.groupby("role_family").agg(
        postings=("job_id", "count"),
        median_salary_min=("salary_min", "median"),
        median_salary_max=("salary_max", "median"),
    ).reset_index().sort_values("postings", ascending=False)
    locations = jobs.groupby("location").size().sort_values(ascending=False).head(12).reset_index()
    locations.columns = ["location", "postings"]
    countries = jobs.groupby("country").size().sort_values(ascending=False).head(15).reset_index()
    countries.columns = ["country", "postings"]
    return {"top_skills": top_skills, "salary_by_role": salary, "demand_by_location": locations, "demand_by_country": countries}


def persona_pass_table(jobs: pd.DataFrame, model, vectors, nn) -> pd.DataFrame:
    rows = []
    for persona_name, profile in PERSONAS.items():
        ranked = rank_jobs(jobs, profile, model, vectors, nn, top_k=10)
        titles = " ".join(ranked["title"].tolist())
        desc = " ".join(ranked["description"].tolist())
        country_ok = True
        prefs = parse_country_preferences(profile)
        if prefs:
            country_ok = ranked["country"].isin(prefs).all()
        if profile.name == "Aisha":
            passed = country_ok and not SENIOR_PATTERNS.search(titles) and ranked["role_family"].str.contains("ML|Data Scientist|Applied", regex=True).any()
            criterion = "No Senior/Staff and ML-related recommendations"
        elif profile.name == "Marcus":
            passed = country_ok and ranked["years_required"].max() < 3 and not ranked["employment_type"].str.lower().eq("contract").any()
            criterion = "No 3+ year or contract-only roles"
        elif profile.name == "Priya":
            passed = country_ok and not JUNIOR_PATTERNS.search(titles) and ranked["company_size"].min() >= 100
            criterion = "No Junior roles or tiny startups"
        elif profile.name == "Kenji":
            ai_roles = ranked["role_family"].isin(["Research", "Applied AI", "Data Scientist", "ML Infrastructure"]).mean()
            large_company_share = (ranked["company_size"] >= 100).mean()
            passed = country_ok and not CONTRACT_PATTERNS.search(desc) and large_company_share >= 0.7 and ai_roles >= 0.2
            criterion = "No contract roles; favors large AI/research employers"
        else:
            target_tokens = set()
            for role in profile.target_roles:
                target_tokens.update(t for t in re.findall(r"[a-z]+", role.lower()) if len(t) > 3)
            role_text = " ".join(ranked["title"].tolist() + ranked["role_family"].tolist()).lower()
            target_fit = any(t in role_text for t in target_tokens)
            passed = country_ok and target_fit and not ranked["employment_type"].str.lower().eq("contract").any()
            criterion = "Top-10 stays within preferred countries, target field, and avoids contract-only roles"
        rows.append({
            "persona": profile.name,
            "countries": ", ".join(profile.preferred_countries or []),
            "criterion": criterion,
            "top10_pass": "PASS" if passed else "PARTIAL",
            "mean_score": round(float(ranked["match_score"].mean()), 3),
        })
    return pd.DataFrame(rows)


def benchmark(jobs: pd.DataFrame, model, vectors, nn) -> pd.DataFrame:
    def violations(frame: pd.DataFrame, profile: Profile) -> int:
        count = 0
        for _, job in frame.iterrows():
            _, flags = _dealbreaker_penalty(job, profile)
            hard_flags = [f for f in flags if "salary below preference" not in f]
            if hard_flags:
                count += 1
        return count

    rows = []
    for persona_name, profile in PERSONAS.items():
        profile_skills = {normalize_skill(s) for s in profile.skills}
        keyword = jobs.copy()
        keyword["keyword_score"] = keyword["skills_list"].apply(lambda s: len(profile_skills & set(s)) / max(1, len(s)))
        keyword_top = keyword.sort_values("keyword_score", ascending=False).head(10)
        ranked = rank_jobs(jobs, profile, model, vectors, nn, top_k=10)
        rows.append({
            "persona": profile.name,
            "keyword_top10_avg_skill_overlap": round(float(keyword_top["skills_list"].apply(lambda s: len(profile_skills & set(s)) / max(1, len(s))).mean()), 3),
            "multistage_top10_avg_skill_overlap": round(float(ranked["skill_overlap"].mean()), 3),
            "keyword_dealbreaker_violations": violations(keyword_top, profile),
            "multistage_dealbreaker_violations": violations(ranked, profile),
            "multistage_avg_score": round(float(ranked["match_score"].mean()), 3),
        })
    return pd.DataFrame(rows)
