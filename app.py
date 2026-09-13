import streamlit as st
import joblib
import json
import os
import re
import base64
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader

# Optional Gemini Generative AI support
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="FinePrint",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------
# LOAD BACKGROUND IMAGE
# ---------------------------------------------------
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


bg_path = "assets/background.jpg"
bg_base64 = get_base64_image(bg_path) if os.path.exists(bg_path) else ""


# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background:
        linear-gradient(rgba(8, 18, 30, 0.82), rgba(8, 18, 30, 0.88)),
        url("data:image/jpg;base64,{bg_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #F5F7FA;
}}

section[data-testid="stSidebar"] {{
    background: rgba(10, 18, 30, 0.88);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255,255,255,0.08);
}}

h1, h2, h3 {{
    color: #F5F7FA !important;
    font-weight: 800 !important;
    letter-spacing: 0.3px;
}}

p, label, div {{
    color: #E6EDF5 !important;
}}

.hero-card {{
    background: rgba(14, 25, 40, 0.72);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    backdrop-filter: blur(12px);
    margin-bottom: 20px;
}}

.metric-card {{
    background: rgba(20, 32, 48, 0.72);
    border: 1px solid rgba(147, 197, 253, 0.14);
    border-radius: 18px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 8px 22px rgba(0,0,0,0.28);
    backdrop-filter: blur(10px);
}}

.metric-title {{
    color: #AFC4D9 !important;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}}

.metric-value {{
    color: #F8FAFC !important;
    font-size: 28px;
    font-weight: 800;
}}

.result-card {{
    background: rgba(14, 25, 40, 0.76);
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 5px solid #7FA9D6;
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}}

.badge {{
    display: inline-block;
    background: linear-gradient(90deg, #6A89B6, #93A9C6);
    color: white;
    font-weight: 700;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 12px;
    margin-right: 8px;
    margin-bottom: 8px;
}}

.plain-box {{
    background: rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 14px;
    border: 1px solid rgba(255,255,255,0.06);
    margin-top: 10px;
    margin-bottom: 10px;
}}

.stTextArea textarea {{
    background: rgba(255,255,255,0.06) !important;
    color: #F5F7FA !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
}}

.stFileUploader {{
    background: rgba(255,255,255,0.04);
    padding: 8px;
    border-radius: 14px;
}}

.stButton > button {{
    background: linear-gradient(90deg, #5E7CA8, #88A1BF);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 1.25rem;
    font-weight: 700;
    font-size: 15px;
    box-shadow: 0 8px 18px rgba(0,0,0,0.25);
}}

.stButton > button:hover {{
    background: linear-gradient(90deg, #6D8AB4, #9CB3CF);
    color: white;
}}

hr {{
    border: none;
    height: 1px;
    background: rgba(255,255,255,0.12);
    margin: 1rem 0;
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# LOAD MODEL ARTIFACTS
# ---------------------------------------------------
@st.cache_resource
def load_artifacts():
    tfidf = joblib.load("fineprint_artifacts/tfidf_vectorizer.joblib")
    model = joblib.load("fineprint_artifacts/fineprint_classifier.joblib")
    thresholds = joblib.load("fineprint_artifacts/thresholds.joblib")

    with open("fineprint_artifacts/label_names.json", "r", encoding="utf-8") as f:
        label_names = json.load(f)

    return tfidf, model, thresholds, label_names


tfidf, model, thresholds, label_names = load_artifacts()


# ---------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------
def split_into_sentences(text):
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def analyze_clause(text):
    if not isinstance(text, str) or not text.strip():
        return []

    text_vector = tfidf.transform([text.strip()])
    probabilities = model.predict_proba(text_vector)[0]

    results = []

    for i, label_name in enumerate(label_names):
        prob = float(probabilities[i])
        threshold = float(thresholds[i])

        if prob >= threshold:
            results.append({
                "category": label_name,
                "confidence": prob,
                "threshold": threshold
            })

    results = sorted(results, key=lambda x: x["confidence"], reverse=True)
    return results


def extract_pdf_text(uploaded_pdf):
    reader = PdfReader(uploaded_pdf)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


# ---------------------------------------------------
# CLAUSE EXPLANATION LOGIC
# ---------------------------------------------------
EXPLANATION_TEMPLATES = {
    "Limitation of liability": {
        "meaning": "This clause appears to limit the company’s legal responsibility for certain losses or damages.",
        "why": "If something goes wrong, your ability to claim compensation may be reduced.",
        "questions": [
            "What kinds of damages are excluded?",
            "Does the limitation apply even if the company is negligent?",
            "Are there any exceptions to this limitation?"
        ]
    },
    "Unilateral termination": {
        "meaning": "This clause suggests the company may suspend or terminate your access under broad conditions.",
        "why": "Your account or service access may be ended with limited notice or appeal options.",
        "questions": [
            "Can the company terminate without warning?",
            "Will unused payments be refunded?",
            "Can you recover your data after termination?"
        ]
    },
    "Unilateral change": {
        "meaning": "This clause indicates the company may change the agreement terms without requiring your fresh approval.",
        "why": "Important rules, pricing, or conditions may change after you already accepted them.",
        "questions": [
            "Will you receive advance notice of changes?",
            "Can you reject major changes?",
            "Can you stop using the service if terms change?"
        ]
    },
    "Content removal": {
        "meaning": "This clause suggests the company can remove, restrict, or disable your uploaded or shared content.",
        "why": "Your content may be taken down, sometimes at the platform’s discretion.",
        "questions": [
            "Under what conditions can content be removed?",
            "Is there an appeal process?",
            "Will you be informed before or after removal?"
        ]
    },
    "Contract by using": {
        "meaning": "This clause means simply using the service may count as agreeing to the contract terms.",
        "why": "Users may become bound by the agreement even without a separate signature.",
        "questions": [
            "Exactly when does the contract take effect?",
            "Does continued use mean automatic acceptance of updates?",
            "Is there any way to opt out?"
        ]
    },
    "Choice of law": {
        "meaning": "This clause specifies which jurisdiction’s laws will govern the agreement.",
        "why": "If a dispute happens, the legal rules applied may be from another place or country.",
        "questions": [
            "Which law governs the agreement?",
            "Is that location convenient for the user?",
            "Does it disadvantage users in another country?"
        ]
    },
    "Jurisdiction": {
        "meaning": "This clause identifies where legal disputes must be filed or heard.",
        "why": "You may have to resolve disputes in a distant or unfamiliar court/location.",
        "questions": [
            "Where must disputes be filed?",
            "Is that location practical for users?",
            "Can users bring claims elsewhere?"
        ]
    },
    "Arbitration": {
        "meaning": "This clause suggests disputes may be resolved through arbitration rather than a regular court.",
        "why": "Arbitration can affect your options for formal legal action and appeals.",
        "questions": [
            "Is arbitration mandatory?",
            "Can users opt out?",
            "Are class actions waived?"
        ]
    }
}


def get_template_explanation(category):
    return EXPLANATION_TEMPLATES.get(category, {
        "meaning": "This clause may require attention.",
        "why": "It may affect the user’s rights or obligations.",
        "questions": ["What does this clause allow?", "How might it affect the user?"]
    })


def get_gemini_client():
    """
    Creates a Gemini client using the API key stored securely
    in Streamlit Secrets.

    Returns None if Gemini is unavailable or no API key exists.
    """
    if not GEMINI_AVAILABLE:
        return None

    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def explain_with_gemini(clause_text, detected_categories):
    """
    Uses Gemini to explain a FinePrint-detected clause
    in plain language.

    Falls back gracefully if the API is unavailable.
    """

    client = get_gemini_client()

    if client is None:
        return None

    categories_text = ", ".join(detected_categories)

    prompt = f"""
You are the Generative AI explanation layer of a contract-transparency
application called FinePrint.

FinePrint's trained machine-learning classifier detected the following
Terms-of-Service clause categories:

{categories_text}

ORIGINAL CLAUSE:
\"\"\"{clause_text}\"\"\"

Your job is NOT to decide whether the clause is illegal, enforceable,
valid, or legally unfair.

Instead, help an ordinary non-lawyer understand what the clause appears
to mean and what they may want to pay attention to.

Return a concise response using EXACTLY these headings:

### Simple Meaning
Explain the clause in 2-3 simple sentences.

### Why It Matters
Explain the practical significance to a user in 2-3 sentences.

### Questions to Consider
Provide exactly 3 useful questions the user may want to investigate
before accepting the agreement.

### Roman Urdu
Explain the clause naturally in simple Roman Urdu in 2-3 sentences.

Do not:
- provide legal advice
- say the clause is illegal
- say the user should or should not sign
- invent facts not contained in the clause
- make claims about Pakistani, US, EU, or other law unless explicitly
  stated in the clause

Keep the entire response concise.
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        if response and response.text:
            return response.text.strip()

        return None

    except Exception as e:
        return None


def roman_urdu_explanation(categories):
    lines = []
    for cat in categories:
        if cat == "Unilateral change":
            lines.append("Yeh clause kehta hai ke company terms ko baad mein bina aap ki nayi raza mandi ke change kar sakti hai.")
        elif cat == "Unilateral termination":
            lines.append("Yeh clause batata hai ke company kuch suratoun mein aap ka account band ya suspend kar sakti hai.")
        elif cat == "Limitation of liability":
            lines.append("Yeh clause company ki zimmedari ko limit karta hai agar kisi nuqsan ya maslay ki surat mein dispute ho.")
        elif cat == "Content removal":
            lines.append("Yeh clause kehta hai ke company aap ka content remove ya restrict kar sakti hai.")
        elif cat == "Contract by using":
            lines.append("Is ka matlab hai ke service use karna hi terms ko accept karna samjha ja sakta hai.")
        elif cat == "Choice of law":
            lines.append("Yeh clause batata hai ke is agreement par kis jagah ya state ka law apply hoga.")
        elif cat == "Jurisdiction":
            lines.append("Yeh clause batata hai ke dispute kis court ya jagah par suna jayega.")
        elif cat == "Arbitration":
            lines.append("Yeh clause kehta hai ke dispute court ke bajaye arbitration ke zariye solve ho sakta hai.")
    return " ".join(lines)


# ---------------------------------------------------
# APP SIDEBAR
# ---------------------------------------------------
with st.sidebar:
    st.markdown("## ⚖️ FinePrint")
    st.markdown("**Understand before you agree.**")
    st.markdown("---")
    st.markdown("### What FinePrint does")
    st.markdown("""
- Detects potentially concerning Terms-of-Service clauses  
- Classifies them into clause categories  
- Explains them in simple language  
- Provides helpful follow-up questions  
- Supports Roman Urdu guidance
""")
    st.markdown("---")
   st.markdown("### Hybrid AI Architecture")

st.markdown("""
**Clause Detection:**  
TF-IDF + One-vs-Rest Logistic Regression

**Generative AI:**  
Gemini 3.8 Flash

**Final ML Test Performance:**  
- **Micro F1:** 0.7313  
- **Macro F1:** 0.7442
""")

if get_gemini_client() is not None:
    st.success("Generative AI: Connected")
else:
    st.warning("Generative AI: Template fallback mode")
    st.markdown("---")
    st.info("FinePrint is an informational transparency tool and does not provide legal advice.")


# ---------------------------------------------------
# HERO SECTION
# ---------------------------------------------------
st.markdown("""
<div class="hero-card">
    <h1>⚖️ FinePrint</h1>
    <h3>AI-Powered Contract Transparency & Unfair Clause Detection</h3>
    <p>
        FinePrint helps users understand what they may be agreeing to in digital Terms & Conditions.
        It detects potentially concerning clauses and explains them in simple language.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# METRICS
# ---------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Test Micro F1</div>
        <div class="metric-value">0.7313</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Test Macro F1</div>
        <div class="metric-value">0.7442</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Clause Categories</div>
        <div class="metric-value">8</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Focus</div>
        <div class="metric-value">ToS</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------
tab1, tab2 = st.tabs(["📄 Analyse Full Agreement", "✍ Analyse Single Clause"])

with tab1:
    st.markdown("### Upload or paste Terms & Conditions")

    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

    text_input = st.text_area(
        "Or paste the agreement text here",
        height=250,
        placeholder="Paste Terms & Conditions here..."
    )

    if st.button("Analyse Agreement"):
        full_text = ""

        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                full_text = extract_pdf_text(uploaded_file)
            else:
                full_text = uploaded_file.read().decode("utf-8", errors="ignore")

        if text_input.strip():
            full_text = text_input.strip()

        if not full_text.strip():
            st.warning("Please upload a file or paste agreement text.")
        else:
            sentences = split_into_sentences(full_text)

            flagged_results = []
            category_counts = {}

            for sentence in sentences:
                results = analyze_clause(sentence)
                if results:
                    flagged_results.append({
                        "sentence": sentence,
                        "results": results
                    })

                    for r in results:
                        category_counts[r["category"]] = category_counts.get(r["category"], 0) + 1

            st.markdown("## Contract Transparency Report")

            c1, c2, c3 = st.columns(3)
            c1.metric("Sentences Analysed", len(sentences))
            c2.metric("Flagged Clauses", len(flagged_results))
            c3.metric("Unique Categories Detected", len(category_counts))

            if category_counts:
                summary_df = pd.DataFrame({
                    "Clause Category": list(category_counts.keys()),
                    "Count": list(category_counts.values())
                }).sort_values("Count", ascending=False)

                st.markdown("### Detected Clause Categories")
                st.dataframe(summary_df, use_container_width=True)

            if not flagged_results:
                st.success("No trained unfair-clause categories were detected in this text.")
            else:
                st.markdown("### Flagged Clauses")

                for idx, item in enumerate(flagged_results, start=1):
                    sentence = item["sentence"]
                    results = item["results"]

                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown(f"#### Clause {idx}")

                    for r in results:
                        st.markdown(
                            f'<span class="badge">{r["category"]} | {r["confidence"]:.2%}</span>',
                            unsafe_allow_html=True
                        )

                    st.markdown("**Original clause:**")
                    st.markdown(f'<div class="plain-box">{sentence}</div>', unsafe_allow_html=True)

                    top_categories = [r["category"] for r in results]

                    gemini_text = explain_with_gemini(sentence, top_categories)

                    if gemini_text:
                        st.markdown("**AI Explanation:**")
                        st.markdown(f'<div class="plain-box">{gemini_text}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown("**Simple explanation:**")
                        for cat in top_categories:
                            info = get_template_explanation(cat)
                            st.markdown(f"**{cat}**")
                            st.markdown(f"- **Meaning:** {info['meaning']}")
                            st.markdown(f"- **Why it matters:** {info['why']}")
                            st.markdown("- **Questions to consider:**")
                            for q in info["questions"]:
                                st.markdown(f"  - {q}")

                    st.markdown("**Roman Urdu summary:**")
                    st.markdown(f'<div class="plain-box">{roman_urdu_explanation(top_categories)}</div>', unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)


with tab2:
    st.markdown("### Analyse a single clause")

    single_clause = st.text_area(
        "Enter one clause",
        height=180,
        placeholder="Example: We reserve the right to modify these terms at any time without prior notice."
    )

    if st.button("Analyse Clause"):
        if not single_clause.strip():
            st.warning("Please enter a clause.")
        else:
            results = analyze_clause(single_clause)

            st.markdown("## Clause Analysis")

            if not results:
                st.success("No trained unfair-clause category was detected.")
            else:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)

                for r in results:
                    st.markdown(
                        f'<span class="badge">{r["category"]} | {r["confidence"]:.2%}</span>',
                        unsafe_allow_html=True
                    )

                st.markdown("**Original clause:**")
                st.markdown(f'<div class="plain-box">{single_clause}</div>', unsafe_allow_html=True)

                top_categories = [r["category"] for r in results]

                gemini_text = explain_with_gemini(single_clause, top_categories)

                if gemini_text:
                    st.markdown("**AI Explanation:**")
                    st.markdown(f'<div class="plain-box">{gemini_text}</div>', unsafe_allow_html=True)
                else:
                    st.markdown("**Simple explanation:**")
                    for cat in top_categories:
                        info = get_template_explanation(cat)
                        st.markdown(f"**{cat}**")
                        st.markdown(f"- **Meaning:** {info['meaning']}")
                        st.markdown(f"- **Why it matters:** {info['why']}")
                        st.markdown("- **Questions to consider:**")
                        for q in info["questions"]:
                            st.markdown(f"  - {q}")

                st.markdown("**Roman Urdu summary:**")
                st.markdown(f'<div class="plain-box">{roman_urdu_explanation(top_categories)}</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)


st.markdown("---")
st.caption("FinePrint is an informational AI system for contract transparency and does not provide legal advice.")
