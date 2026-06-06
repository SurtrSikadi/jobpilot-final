from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from jobpilot_core import (
    PERSONAS,
    Profile,
    analytics,
    benchmark,
    explain_job,
    fit_embeddings,
    load_jobs,
    persona_pass_table,
    rank_jobs,
    stream_to_sqlite,
    tailored_resume,
    update_feedback_weights,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "job_postings_sample.csv"
DB_PATH = ROOT / "data" / "jobpilot_stream.sqlite"


st.set_page_config(page_title="JobPilot", page_icon="JP", layout="wide")


def inject_styles():
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1180px;
            padding-top: 1.5rem;
        }
        [data-testid="stMetric"] {
            min-width: 0;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 10px;
        }
        .job-card-title {
            font-size: clamp(1.35rem, 2.5vw, 2rem);
            line-height: 1.18;
            font-weight: 700;
            color: #262730;
            overflow-wrap: anywhere;
            margin: 0 0 .5rem 0;
        }
        .job-meta {
            color: #3f4250;
            font-size: clamp(.95rem, 1.7vw, 1.08rem);
            line-height: 1.45;
            overflow-wrap: anywhere;
            margin-bottom: .55rem;
        }
        .job-explain {
            color: #6b6f7a;
            font-size: .95rem;
            line-height: 1.55;
            overflow-wrap: anywhere;
            margin: .55rem 0 .75rem 0;
        }
        div.stButton > button,
        div.stDownloadButton > button,
        div[data-testid="stLinkButton"] > a {
            width: 100%;
            min-height: 2.5rem;
            white-space: nowrap;
            text-align: center;
        }
        @media (max-width: 760px) {
            .block-container {
                padding-left: .8rem;
                padding-right: .8rem;
            }
            .job-card-title {
                font-size: 1.25rem;
            }
            .job-meta {
                font-size: .95rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def cached_jobs():
    return load_jobs(DATA_PATH)


@st.cache_resource(show_spinner=False)
def cached_models(jobs_df: pd.DataFrame):
    return fit_embeddings(jobs_df)


def profile_from_inputs() -> Profile:
    persona_label = st.sidebar.selectbox("Persona", list(PERSONAS.keys()))
    base = PERSONAS[persona_label]
    resume = st.sidebar.text_area("Resume / profile text", value=base.background, height=120)
    skills = st.sidebar.text_input("Skills", value=", ".join(base.skills))
    targets = st.sidebar.text_input("Target roles", value=", ".join(base.target_roles))
    countries = st.sidebar.text_input("Preferred countries", value=", ".join(base.preferred_countries or [base.location]))
    min_salary = st.sidebar.number_input("Minimum salary", min_value=0, max_value=250000, value=base.min_salary, step=5000)
    dealbreakers = st.sidebar.text_area("Dealbreakers", value="\n".join(base.dealbreakers or []), height=90)
    return Profile(
        name=base.name,
        background=resume,
        skills=[s.strip() for s in skills.split(",") if s.strip()],
        target_roles=[r.strip() for r in targets.split(",") if r.strip()],
        location=countries,
        min_salary=int(min_salary),
        dealbreakers=[d.strip() for d in dealbreakers.splitlines() if d.strip()],
        preferred_countries=[c.strip() for c in countries.split(",") if c.strip()],
        visa_required=base.visa_required,
        prefer_remote=base.prefer_remote,
    )


def ensure_feedback_state():
    if "feedback_weights" not in st.session_state:
        st.session_state.feedback_weights = {}
    if "feedback_log" not in st.session_state:
        st.session_state.feedback_log = []


def main():
    inject_styles()
    ensure_feedback_state()
    jobs = cached_jobs()
    model, vectors, nn = cached_models(jobs)
    profile = profile_from_inputs()

    st.title("JobPilot")
    st.caption("Smart job matcher with simulated streaming ingestion, dense retrieval, multi-stage ranking, feedback learning, explanations, analytics, and resume tailoring.")

    summary_cols = st.columns(4)
    summary_cols[0].metric("Stored postings", f"{len(jobs):,}")
    summary_cols[1].metric("Countries", f"{jobs.country.nunique():,}")
    summary_cols[2].metric("Median salary max", f"${int(jobs.salary_max.median()):,}")
    summary_cols[3].metric("Visa sponsor share", f"{jobs.sponsors_visa.mean():.0%}")

    tabs = st.tabs(["Recommendations", "Analytics", "Pipeline", "Benchmarks"])

    with tabs[0]:
        top_k = st.slider("Recommendations to show", 5, 40, 15)
        ranked = rank_jobs(jobs, profile, model, vectors, nn, st.session_state.feedback_weights, top_k=top_k)

        export_cols = ["job_id", "title", "company", "location", "country", "salary_min", "salary_max", "apply_link", "description"]
        st.download_button(
            "Download top jobs CSV",
            ranked[export_cols].to_csv(index=False).encode("utf-8"),
            file_name="jobpilot_top_jobs.csv",
            mime="text/csv",
        )

        for i, job in ranked.iterrows():
            with st.container(border=True):
                title = escape(str(job.title))
                company = escape(str(job.company))
                location = escape(str(job.location))
                country = escape(str(job.country))
                explanation = escape(explain_job(job, profile))
                st.markdown(
                    f"<div class='job-card-title'>{i + 1}. {title}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    (
                        "<div class='job-meta'>"
                        f"{company} - {location} - {country} - "
                        f"${int(job.salary_min):,}-${int(job.salary_max):,}"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
                st.progress(max(0.0, min(1.0, float(job.match_score))), text=f"Match score {job.match_score:.2f}")
                st.markdown(
                    f"<div class='job-explain'>{explanation}</div>",
                    unsafe_allow_html=True,
                )
                action_cols = st.columns([1.1, 1, 1, 1, 2.4])
                action_cols[0].link_button("Apply", job.apply_link)
                if action_cols[1].button("Accept", key=f"accept-{job.job_id}"):
                    st.session_state.feedback_weights = update_feedback_weights(st.session_state.feedback_weights, job, "accept")
                    st.session_state.feedback_log.append((job.job_id, "accept"))
                    st.rerun()
                if action_cols[2].button("Skip", key=f"skip-{job.job_id}"):
                    st.session_state.feedback_weights = update_feedback_weights(st.session_state.feedback_weights, job, "skip")
                    st.session_state.feedback_log.append((job.job_id, "skip"))
                    st.rerun()
                if action_cols[3].button("Reject", key=f"reject-{job.job_id}"):
                    st.session_state.feedback_weights = update_feedback_weights(st.session_state.feedback_weights, job, "reject")
                    st.session_state.feedback_log.append((job.job_id, "reject"))
                    st.rerun()
                with st.expander("Generate Resume"):
                    resume = tailored_resume(profile, job)
                    st.text_area("Tailored resume draft", value=resume, height=260, key=f"resume-{job.job_id}")
                    st.download_button(
                        "Download resume draft",
                        resume.encode("utf-8"),
                        file_name=f"resume_{profile.name}_{job.job_id}.txt",
                        mime="text/plain",
                        key=f"download-{job.job_id}",
                    )

    with tabs[1]:
        insight = analytics(jobs)
        col1, col2 = st.columns(2)
        col1.subheader("Top Skills")
        col1.bar_chart(insight["top_skills"].set_index("skill"))
        col2.subheader("Demand by Location")
        col2.bar_chart(insight["demand_by_location"].set_index("location"))
        st.subheader("Demand by Country")
        st.bar_chart(insight["demand_by_country"].set_index("country"))
        st.subheader("Salary by Role Family")
        st.dataframe(insight["salary_by_role"], use_container_width=True)

    with tabs[2]:
        st.subheader("Streaming Ingestion Simulation")
        st.write("This button replays the offline snapshot into SQLite in batches, deduplicating by `job_id` as a stand-in for a real Pub/Sub or Kafka stream.")
        if st.button("Run stream replay"):
            result = stream_to_sqlite(DATA_PATH, DB_PATH)
            st.success(f"Inserted {result['inserted']} records, skipped {result['duplicates']} duplicates, stored {result['stored']} total.")
        st.code("python -m jobpilot_core  # core module used by Streamlit app", language="bash")
        st.write("Production path: replace CSV replay with Adzuna/JSearch fetcher, publish records to Pub/Sub, and keep the same dedup/store/rank boundary.")

    with tabs[3]:
        st.subheader("Persona Pass Table")
        st.dataframe(persona_pass_table(jobs, model, vectors, nn), use_container_width=True)
        st.subheader("Technique Benchmark")
        st.dataframe(benchmark(jobs, model, vectors, nn), use_container_width=True)
        st.write(f"Feedback events this session: {len(st.session_state.feedback_log)}")


if __name__ == "__main__":
    main()
