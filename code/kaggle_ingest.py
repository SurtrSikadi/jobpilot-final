from __future__ import annotations

import csv
import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path
from zipfile import ZipFile


ZIP_PATH = Path(os.environ.get("TECHMAP_JOBS_ZIP", r"C:\Users\Rui Wang\Downloads\techmap-jobs-dump-2021-09.json.zip"))
OUT = Path(os.environ.get("JOBPILOT_OUT_CSV", "data/job_postings_sample.csv"))
MAX_ROWS = 27423
SCAN_LIMIT = 900000
random.seed(423)


def safe_print(message: str) -> None:
    try:
        print(message, flush=True)
    except OSError:
        pass

COUNTRY_NAMES = {
    "us": "United States",
    "uk": "United Kingdom",
    "gb": "United Kingdom",
    "de": "Germany",
    "ca": "Canada",
    "au": "Australia",
    "in": "India",
    "sg": "Singapore",
    "hk": "Hong Kong",
    "ie": "Ireland",
    "nz": "New Zealand",
    "ch": "Switzerland",
    "nl": "Netherlands",
    "fr": "France",
    "se": "Sweden",
    "jp": "Japan",
    "kr": "South Korea",
    "cn": "China",
    "ua": "Ukraine",
    "ph": "Philippines",
    "at": "Austria",
    "lu": "Luxembourg",
    "cz": "Czech Republic",
    "ru": "Russia",
    "my": "Malaysia",
    "id": "Indonesia",
    "gr": "Greece",
    "tw": "Taiwan",
    "pt": "Portugal",
    "be": "Belgium",
    "ae": "United Arab Emirates",
    "th": "Thailand",
    "es": "Spain",
    "vn": "Vietnam",
    "it": "Italy",
}

COUNTRY_CAPS = {
    "us": 18000,
    "uk": 14000,
    "gb": 14000,
    "de": 12000,
    "au": 7000,
    "in": 7000,
    "sg": 5000,
    "ca": 5000,
    "hk": 4500,
    "ie": 3500,
    "nz": 3000,
    "ch": 3000,
    "nl": 2500,
    "fr": 2500,
    "se": 2000,
    "jp": 1800,
    "kr": 1800,
    "cn": 1800,
}

SKILL_PATTERNS = [
    ("python", r"\bpython\b"),
    ("sql", r"\bsql\b"),
    ("r", r"\br\b"),
    ("tableau", r"\btableau\b"),
    ("power bi", r"\bpower\s*bi\b"),
    ("spark", r"\bspark\b"),
    ("pyspark", r"\bpyspark\b"),
    ("kafka", r"\bkafka\b"),
    ("kubernetes", r"\bkubernetes\b"),
    ("docker", r"\bdocker\b"),
    ("aws", r"\baws|amazon web services\b"),
    ("gcp", r"\bgcp|google cloud\b"),
    ("azure", r"\bazure\b"),
    ("scikit-learn", r"\bscikit[- ]learn|sklearn\b"),
    ("pytorch", r"\bpytorch\b"),
    ("tensorflow", r"\btensorflow\b"),
    ("nlp", r"\bnlp|natural language processing\b"),
    ("computer vision", r"\bcomputer vision\b"),
    ("deep learning", r"\bdeep learning\b"),
    ("machine learning", r"\bmachine learning\b"),
    ("mlops", r"\bmlops|model operations\b"),
    ("airflow", r"\bairflow\b"),
    ("dbt", r"\bdbt\b"),
    ("snowflake", r"\bsnowflake\b"),
    ("excel", r"\bexcel\b"),
    ("statistics", r"\bstatistics|statistical\b"),
    ("experimentation", r"\bab test|a/b test|experiment"),
    ("feature engineering", r"\bfeature engineering\b"),
    ("microservices", r"\bmicroservices\b"),
    ("java", r"\bjava\b"),
    ("c++", r"\bc\+\+\b"),
    ("fastapi", r"\bfastapi\b"),
    ("streamlit", r"\bstreamlit\b"),
    ("llm", r"\bllm|large language model\b"),
    ("rag", r"\brag|retrieval augmented\b"),
    ("data modeling", r"\bdata model"),
    ("sales", r"\bsales|account executive|business development\b"),
    ("marketing", r"\bmarketing|seo|campaign|brand|content strategy\b"),
    ("finance", r"\bfinance|financial|accounting|fp&a|investment\b"),
    ("operations", r"\boperations|supply chain|logistics|process improvement\b"),
    ("healthcare", r"\bhealthcare|clinical|patient|medical|hospital\b"),
    ("human resources", r"\bhuman resources|recruiting|talent acquisition|hr\b"),
    ("project management", r"\bproject manager|program manager|scrum|agile\b"),
    ("customer success", r"\bcustomer success|customer support|client success\b"),
    ("design", r"\bux|ui designer|product designer|graphic design\b"),
    ("teaching", r"\bteacher|lecturer|instructional|curriculum\b"),
    ("legal", r"\blegal|attorney|paralegal|compliance\b"),
]


def clean_text(value, limit: int = 6000) -> str:
    text = str(value or "")[:limit]
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def nested(obj, *keys, default=""):
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def classify_role(title: str, text: str) -> str:
    hay = f"{title} {text}".lower()
    if re.search(r"\b(account executive|sales|business development)\b", hay):
        return "Sales"
    if re.search(r"\b(marketing|seo|brand|content strategist|campaign)\b", hay):
        return "Marketing"
    if re.search(r"\b(finance|financial analyst|accounting|fp&a|investment)\b", hay):
        return "Finance"
    if re.search(r"\b(operations|supply chain|logistics|procurement)\b", hay):
        return "Operations"
    if re.search(r"\b(healthcare|clinical|patient|medical|hospital|nurse)\b", hay):
        return "Healthcare"
    if re.search(r"\b(human resources|recruiter|talent acquisition|hr manager)\b", hay):
        return "Human Resources"
    if re.search(r"\b(project manager|program manager|product manager)\b", hay):
        return "Product/Project"
    if re.search(r"\b(customer success|customer support|client success)\b", hay):
        return "Customer Success"
    if re.search(r"\b(ux|ui designer|product designer|graphic designer)\b", hay):
        return "Design"
    if re.search(r"\b(teacher|lecturer|instructional|curriculum)\b", hay):
        return "Education"
    if re.search(r"\b(attorney|paralegal|legal|compliance)\b", hay):
        return "Legal/Compliance"
    if "platform" in hay or "mlops" in hay or "kubernetes" in hay:
        return "ML Infrastructure"
    if "research scientist" in hay:
        return "Research"
    if "applied scientist" in hay or "ai engineer" in hay:
        return "Applied AI"
    if "data engineer" in hay or "analytics engineer" in hay:
        return "Analytics Engineering"
    if "data scientist" in hay or "machine learning" in hay or "ml engineer" in hay:
        return "Data Scientist"
    if "business intelligence" in hay or "bi analyst" in hay or "tableau" in hay or "power bi" in hay:
        return "Analytics"
    if re.search(r"\b(software engineer|developer|engineer)\b", hay):
        return "Software Engineering"
    return "General Business"


def extract_skills(text: str) -> list[str]:
    found = [name for name, pattern in SKILL_PATTERNS if re.search(pattern, text, re.I)]
    if not found:
        found = ["communication", "problem solving", "stakeholder management"]
    return sorted(set(found))


def salary_for(role_family: str, title: str, text: str) -> tuple[int, int]:
    hay = f"{title} {text}".lower()
    base = {
        "Analytics": (70000, 115000),
        "Analytics Engineering": (90000, 145000),
        "Data Scientist": (100000, 165000),
        "Applied AI": (115000, 185000),
        "Research": (125000, 210000),
        "ML Infrastructure": (125000, 205000),
        "Software Engineering": (100000, 170000),
        "Sales": (65000, 130000),
        "Marketing": (65000, 120000),
        "Finance": (75000, 140000),
        "Operations": (65000, 120000),
        "Healthcare": (65000, 130000),
        "Human Resources": (60000, 110000),
        "Product/Project": (85000, 155000),
        "Customer Success": (55000, 105000),
        "Design": (70000, 130000),
        "Education": (50000, 100000),
        "Legal/Compliance": (70000, 145000),
        "General Business": (60000, 115000),
    }.get(role_family, (80000, 130000))
    if re.search(r"\b(senior|sr\.?|staff|principal|lead)\b", hay):
        base = (base[0] + 25000, base[1] + 35000)
    if re.search(r"\b(junior|entry level|new grad|associate)\b", hay):
        base = (max(50000, base[0] - 25000), max(70000, base[1] - 30000))
    jitter = random.randint(-6000, 9000)
    return max(45000, base[0] + jitter), max(65000, base[1] + jitter)


def country_from_obj(obj: dict, schema: dict) -> tuple[str, str]:
    source_cc = str(obj.get("sourceCC") or "").lower()
    loc = schema.get("jobLocation", {})
    if isinstance(loc, list):
        loc = loc[0] if loc else {}
    address = loc.get("address", {}) if isinstance(loc, dict) else {}
    if isinstance(address, list):
        address = address[0] if address else {}
    if not isinstance(address, dict):
        address = {}
    country = address.get("addressCountry") or source_cc or ""
    if isinstance(country, dict):
        country = country.get("name") or country.get("@id") or source_cc
    country_s = str(country).strip()
    country_key = country_s.lower()
    if country_key in COUNTRY_NAMES:
        return country_key, COUNTRY_NAMES[country_key]
    if source_cc in COUNTRY_NAMES:
        return source_cc, COUNTRY_NAMES[source_cc]
    return source_cc or "unknown", country_s or "Unknown"


def location_from_schema(schema: dict, country_name: str) -> str:
    loc = schema.get("jobLocation", {})
    if isinstance(loc, list):
        loc = loc[0] if loc else {}
    address = loc.get("address", {}) if isinstance(loc, dict) else {}
    if isinstance(address, list):
        address = address[0] if address else {}
    if not isinstance(address, dict):
        address = {}
    city = address.get("addressLocality") or ""
    state = address.get("addressRegion") or ""
    country = address.get("addressCountry") or ""
    if isinstance(country, dict):
        country = country.get("name") or country.get("@id") or ""
    if isinstance(state, dict):
        state = state.get("name") or ""
    if isinstance(city, dict):
        city = city.get("name") or ""
    city, state, country = str(city), str(state), str(country)
    if country.lower() in COUNTRY_NAMES:
        country = COUNTRY_NAMES[country.lower()]
    if not country or country == "Unknown":
        country = country_name
    pieces = [x for x in [city, state, country] if x]
    return ", ".join(pieces) if pieces else country_name


def parse_stream(zip_path: Path):
    decoder = json.JSONDecoder()
    with ZipFile(zip_path) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            buffer = ""
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk and not buffer.strip():
                    break
                buffer += chunk.decode("utf-8", errors="ignore")
                while buffer:
                    buffer = buffer.lstrip()
                    try:
                        obj, idx = decoder.raw_decode(buffer)
                    except json.JSONDecodeError:
                        break
                    yield obj
                    buffer = buffer[idx:]
                if not chunk:
                    break


def make_row(obj: dict, n: int) -> dict | None:
    schema = nested(obj, "json", "schemaOrg", default={})
    title = clean_text(schema.get("title") or obj.get("title") or "", limit=300)
    raw_text = obj.get("text") or schema.get("description") or ""
    text = clean_text(raw_text, limit=3000)
    if not title or not text or len(text) < 250:
        return None
    company = clean_text(nested(schema, "hiringOrganization", "name", default="Unknown Company"))
    url = schema.get("url") or obj.get("url") or f"https://example.com/kaggle/{n}"
    role_family = classify_role(title, text)
    skills = extract_skills(f"{title} {text}")
    country_code, country_name = country_from_obj(obj, schema)
    salary_min, salary_max = salary_for(role_family, title, text)
    hay = f"{title} {text}".lower()
    remote = bool(re.search(r"\bremote|work from home|hybrid\b", hay))
    sponsor = bool(re.search(r"\bh-?1b|visa sponsorship|sponsor", hay))
    employment = "Contract" if re.search(r"\bcontract|contractor|temporary|temp\b", hay) else "Full-time"
    digest = hashlib.md5(f"{title}|{company}|{url}".encode("utf-8")).hexdigest()[:12]
    return {
        "job_id": f"KAG-{digest}",
        "title": title[:140],
        "role_family": role_family,
        "company": company[:100] or "Unknown Company",
        "location": location_from_schema(schema, country_name),
        "country": country_name,
        "country_code": country_code,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "skills": ";".join(skills),
        "description": text[:3000],
        "apply_link": url,
        "employment_type": employment,
        "remote": remote,
        "sponsors_visa": sponsor,
        "company_size": random.choice([80, 120, 250, 600, 1400, 3200]),
        "source": "techmap_kaggle_2021_09",
    }


def main(zip_path: Path = ZIP_PATH, out: Path = OUT):
    tmp_out = out.with_suffix(".csv.tmp")
    seen = set()
    country_counts = {}
    scanned = 0
    out.parent.mkdir(exist_ok=True)
    fieldnames = None
    collected = 0
    with tmp_out.open("w", newline="", encoding="utf-8") as f:
        writer = None
        for obj in parse_stream(zip_path):
            scanned += 1
            row = make_row(obj, scanned)
            if row and row["job_id"] not in seen:
                cc = row["country_code"]
                cap = COUNTRY_CAPS.get(cc, 200)
                if country_counts.get(cc, 0) >= cap and scanned < SCAN_LIMIT:
                    continue
                if writer is None:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                writer.writerow(row)
                collected += 1
                seen.add(row["job_id"])
                country_counts[cc] = country_counts.get(cc, 0) + 1
                if collected % 1000 == 0:
                    f.flush()
                if collected % 5000 == 0:
                    top = sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:8]
                    safe_print(f"collected={collected} scanned={scanned} top_countries={top}")
                if collected >= MAX_ROWS:
                    break
            if scanned >= SCAN_LIMIT and collected >= MAX_ROWS:
                break

    if collected < MAX_ROWS:
        raise RuntimeError(f"Only collected {collected} relevant jobs after scanning {scanned} records")

    tmp_out.replace(out)
    safe_print(f"wrote {collected} rows after scanning {scanned} records to {out}")
    safe_print("country_counts=" + str(sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:30]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a normalized JobPilot sample from the Techmap/Kaggle jobs zip.")
    parser.add_argument("--zip", type=Path, default=ZIP_PATH, help="Path to techmap-jobs-dump-2021-09.json.zip")
    parser.add_argument("--out", type=Path, default=OUT, help="Output CSV path")
    args = parser.parse_args()
    main(args.zip, args.out)
