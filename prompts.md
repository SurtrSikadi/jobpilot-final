# Key AI Prompts

1. **Initial weak prompt:** "Build my JobPilot final project."  
   Purpose: established that the request was too broad. I replaced it with a rubric-grounded implementation plan.

2. **Refined project prompt:** "Read the JobPilot assignment and lecture notes, identify required deliverables, then build a runnable prototype that covers all six core capabilities."  
   Purpose: made the output align with the grading rubric instead of producing a generic job-board demo.

3. **Production-aware implementation prompt:** "Create a Streamlit app with a real Kaggle job-posting snapshot, simulated streaming ingestion, deduplication, dense embedding retrieval, multi-stage ranking, feedback learning, explanations, CSV export, analytics, persona benchmarks, and resume generation."  
   Purpose: forced coverage of data, model, serving/UI, and evaluation layers from Lecture 10.

4. **Kaggle data prompt:** "Stream the professor-recommended Techmap jobs zip without extracting the full JSON file, create a 27,423-posting multi-country sample across many fields, normalize schema.org fields, and write a deployable CSV sample with country fields."  
   Purpose: replaced the narrower MSBA/data sample with a broad real job-posting pool while keeping the project runnable.

5. **Location-aware prompt:** "Make recommendations respect each persona's preferred countries before ranking, then add international personas with diverse country constraints."  
   Purpose: fixed the issue where location was only a light preference rather than a real job-search constraint.

6. **Benchmark prompt:** "Compare keyword skill matching against the dense retrieval plus ranking pipeline for each persona, and report measurable differences in the technical brief."  
   Purpose: tied model choices to a benchmark rather than just naming techniques.

7. **Brief prompt:** "Write a max-four-page technical brief with architecture, BAX-423 technique choices, benchmark results, persona pass/fail table, limitations, and production deployment path."  
   Purpose: converted implementation details into a demo-ready submission artifact.
