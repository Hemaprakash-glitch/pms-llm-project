import streamlit as st
import json
import re
import time
import email
from email import policy
from email.parser import BytesParser
from llama_cpp import Llama

# PAGE CONFIGURATION
st.set_page_config(page_title="PMS Intelligence System", layout="wide")

#DESIGN

st.markdown("""
<style>
.block-container {
    max-width: 95% !important;
    padding-top: 4rem !important;
}
html, body, [class*="css"]  {
    background-color: #0F172A;
    color: #E2E8F0;
}
h1 {
    color: #38BDF8;
    font-weight: 700;
}
div[data-testid="stButton"] button {
    height: 45px;
    font-weight: 600;
    border-radius: 8px;
}
.stTextArea textarea {
    background-color: #020617 !important;
    color: #CBD5E1 !important;
    border-radius: 8px !important;
    border: 1px solid #334155 !important;
}
.stFileUploader {
    background-color: #020617;
    border-radius: 8px;
    padding: 10px;
}
.runtime-box {
    background: #064E3B;
    color: #34D399;
    padding: 6px 14px;
    border-radius: 6px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# SESSION STATE
if "results" not in st.session_state:
    st.session_state.results = None
if "runtime" not in st.session_state:
    st.session_state.runtime = None
if "uploader_key" not in st.session_state:  
    st.session_state.uploader_key = 0

def reset_dashboard():
    current_key = st.session_state.get("uploader_key", 0)
    st.session_state.clear()
    st.session_state["uploader_key"] = current_key + 1
    st.rerun()

# MODEL LOADING (RUNPOD VERSION)
@st.cache_resource
def load_model():
    return Llama(
        model_path="/workspace/pms_mistral.q4_k_m.gguf",
        n_ctx=2048,
        n_threads=8,
        n_batch=512,
        n_gpu_layers=-1,
        verbose=False
    )

llm = load_model()

# UI SETUP
st.title("PMS Semantic Extractor")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Upload Email / Complaint (.txt / .eml)",
        type=["txt", "eml"],
        key=st.session_state.uploader_key  
    )

with col2:
    pasted_input = st.text_area(
        "Or Paste Complaint Narrative Manually:",
        height=250
    )

colA, colB = st.columns([0.85, 0.15])

with colA:
    analyze_btn = st.button("RUN ANALYSIS", use_container_width=True)

with colB:
    st.button("RESET", on_click=reset_dashboard, use_container_width=True)

user_input = ""

# FILE PARSING 
if uploaded_file is not None:

    if uploaded_file.name.endswith(".txt"):
        user_input = uploaded_file.getvalue().decode("utf-8")

    elif uploaded_file.name.endswith(".eml"):
        msg = BytesParser(policy=policy.default).parsebytes(uploaded_file.getvalue())

        if msg.is_multipart():
            parts = []
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    parts.append(part.get_content())
            user_input = "\n".join(parts)
        else:
            user_input = msg.get_content()

    st.info("Using content from uploaded file.")

else:
    user_input = pasted_input

# CORE LOGIC 
def override_fields(result_obj, text):
    text_lower = text.lower()

    lot_matches = re.findall(r"\b(?:LOT|BATCH)[:\s-]+([A-Z0-9-]+)\b", text, re.I)
    if result_obj.get("lot_number") in ["Not Available", "", None]:
        if len(set(lot_matches)) == 1:
            result_obj["lot_number"] = lot_matches[0]

    mat_matches = re.findall(r"\b(?:MAT|REF)[:\s-]+([A-Z0-9-]+)\b", text, re.I)
    if result_obj.get("material_number") in ["Not Available", "", None]:
        if len(set(mat_matches)) == 1:
            result_obj["material_number"] = mat_matches[0]

    if result_obj.get("occurrence_count") in ["Not Available", "", None]:
        count_match = re.search(r"\b(\d+)\s+(unit|units|case|cases)\b", text_lower)
        if count_match:
            result_obj["occurrence_count"] = count_match.group(1)

    if any(x in text_lower for x in ["discarded", "not returned", "thrown away"]):
        result_obj["sample_available"] = "No"
    elif any(x in text_lower for x in ["sample returned", "samples available", "retained", "preserved"]):
        result_obj["sample_available"] = "Yes"

    if any(x in text_lower for x in [
        "death", "expired", "fatal", "deceased",
        "not revived", "could not be revived",
        "cardiac arrest and could not be revived"
    ]):
        result_obj["death_occurred"] = "Yes"
    elif "no death" in text_lower:
        result_obj["death_occurred"] = "No"

    return result_obj

# EXTRACTION 
if analyze_btn and user_input.strip():
    with st.spinner("Processing..."):
        start_time = time.time()

        prompt = f"""<s>[INST]
You are a Medical Device Quality Specialist.

Extract the following 13 fields into STRICTLY VALID JSON.
Use DOUBLE QUOTES only.
Return ONLY JSON. No explanations.

Fields:
reporter_name, reporter_address, country, date_received,
date_event_occurred, death_occurred, device_problem,
occurrence_count, material_number, material_description,
lot_number, sample_available, patient_harm.

If multiple complaints exist, return JSON list.
If single complaint, return JSON object.

TEXT:
{user_input}
[/INST]"""

        res = llm(
            prompt,
            max_tokens=1200,
            temperature=0,
            repeat_penalty=1.1,
            stop=["</s>"]
        )

        raw_output = res["choices"][0]["text"].strip()

        try:
            start_candidates = [i for i in [raw_output.find("{"), raw_output.find("[")] if i != -1]
            start_bracket = min(start_candidates)
            end_bracket = max(raw_output.rfind("}"), raw_output.rfind("]")) + 1

            json_str = raw_output[start_bracket:end_bracket]
            results = json.loads(json_str)

            if isinstance(results, list):
                updated_results = []
                for obj in results:
                    updated_results.append(override_fields(obj, user_input))
                results = updated_results
            else:
                results = override_fields(results, user_input)

            st.session_state.results = results
            st.session_state.runtime = round(time.time() - start_time, 2)

        except Exception:
            st.session_state.results = {
                "error": "JSON Parsing Error",
                "raw_output": raw_output
            }

# DISPLAY
if st.session_state.results:

    st.markdown(
        f'<div class="runtime-box">Execution Time: {st.session_state.runtime} seconds</div>',
        unsafe_allow_html=True
    )

    if isinstance(st.session_state.results, dict) and "error" in st.session_state.results:
        st.error("Model struggled. Raw output below:")
        st.code(st.session_state.results["raw_output"])
    else:
        tab1, tab2 = st.tabs(["JSON View", "Table View"])

        with tab1:
            st.json(st.session_state.results)

        with tab2:
            if isinstance(st.session_state.results, list):
                st.dataframe(st.session_state.results, use_container_width=True)
            else:
                st.dataframe([st.session_state.results], use_container_width=True)