import json
import re
import io
import docx
import PyPDF2
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np

# ---------------- Optional RAG Dependencies ----------------
try:
    import ahocorasick
    from sentence_transformers import SentenceTransformer
    import faiss
    from transformers import pipeline
    from rapidfuzz import fuzz
    HAS_RAG = True
except ImportError:
    HAS_RAG = False

# ---------------- FastAPI Init ----------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ---------------- Load JSON Skills ----------------
with open("skills.json", "r", encoding="utf-8") as f:
    data = json.load(f)

SKILL_SET = []
def collect_skills(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            collect_skills(v)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, str):
                SKILL_SET.append(item.strip())
            else:
                collect_skills(item)
    elif isinstance(obj, str):
        SKILL_SET.append(obj.strip())

collect_skills(data)
SKILL_SET = [s for s in SKILL_SET if len(s) > 1]
SKILL_SET = list(dict.fromkeys(SKILL_SET))
print(f"✅ Loaded {len(SKILL_SET)} unique skills.")

# ---------------- Build Aho-Corasick Automaton ----------------
if 'ahocorasick' in globals():
    automaton = ahocorasick.Automaton()
    for idx, skill in enumerate(SKILL_SET):
        automaton.add_word(skill.lower(), (idx, skill))
    automaton.make_automaton()
else:
    automaton = None

# ---------------- Load JSON Countries ----------------
with open("countries.json", "r", encoding="utf-8") as f:
    COUNTRY_LIST = json.load(f)
COUNTRY_LIST = [c.strip() for c in COUNTRY_LIST]

# ---------------- Helpers ----------------
def extract_text_from_file(file: UploadFile):
    file.file.seek(0)
    content = ""
    name = file.filename.lower()
    raw = file.file.read()
    if name.endswith(".txt"):
        content = raw.decode("utf-8", errors="ignore")
    elif name.endswith(".docx"):
        doc = docx.Document(io.BytesIO(raw))
        content = "\n".join([para.text for para in doc.paragraphs])
    elif name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(raw))
        for page in pdf_reader.pages:
            content += (page.extract_text() or "") + "\n"
    return content

def extract_skills_exact(jd_text: str):
    jd_lower = jd_text.lower()
    found_skills = set()
    if automaton:
        for end_idx, (idx, skill) in automaton.iter(jd_lower):
            start_idx = end_idx - len(skill) + 1
            before = jd_lower[start_idx - 1] if start_idx > 0 else " "
            after = jd_lower[end_idx + 1] if end_idx + 1 < len(jd_lower) else " "
            if not before.isalnum() and not after.isalnum():
                found_skills.add(skill)
    else:
        for skill in SKILL_SET:
            pattern = r"\b" + re.escape(skill.lower()) + r"\b"
            if re.search(pattern, jd_lower):
                found_skills.add(skill)
    return sorted(found_skills)

# ---------------- Semantic / RAG ----------------
embedder, skill_embeddings, faiss_index, generator = None, None, None, None

if HAS_RAG:
    print("Initializing RAG models...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    skill_embeddings = embedder.encode(SKILL_SET, convert_to_numpy=True, show_progress_bar=False)
    skill_embeddings = skill_embeddings / (np.linalg.norm(skill_embeddings, axis=1, keepdims=True) + 1e-12)
    skill_embeddings = skill_embeddings.astype("float32")

    d = skill_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(d)
    faiss_index.add(skill_embeddings)

    generator = pipeline("text2text-generation", model="google/flan-t5-small")

def filter_skills(jd_text, skills, threshold=70):
    filtered = []
    jd_lower = jd_text.lower()
    for skill in skills:
        if fuzz.partial_ratio(skill.lower(), jd_lower) >= threshold:
            filtered.append(skill)
    return filtered

def rag_extract(jd_text: str, top_k=10):
    if not HAS_RAG:
        return [], "RAG not available. Missing packages."
    q_emb = embedder.encode([jd_text], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
    q_emb = q_emb.astype("float32")

    top_k = min(top_k, len(SKILL_SET))
    D, I = faiss_index.search(q_emb, top_k)
    retrieved = [SKILL_SET[i] for i in I[0]]
    retrieved_filtered = filter_skills(jd_text, retrieved)
    if not retrieved_filtered:
        return [], "No relevant skills found after filtering."

    context = "Relevant skills: " + ", ".join(retrieved_filtered)
    prompt = f"Job Description:\n{jd_text}\n\n{context}\nAnswer: list only the relevant skills clearly."
    ans = generator(prompt, max_length=50, do_sample=False)[0]["generated_text"]
    return retrieved_filtered, ans

# ---------------- Extract Location, Language, Education, Experience, Role & Country ----------------
import spacy
import pycountry
import geonamescache

try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

gc = geonamescache.GeonamesCache()
cities = {city["name"] for city in gc.get_cities().values()}

def extract_location(text: str):
    doc = nlp(text)
    locations = []
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            loc = ent.text.strip()
            if loc in cities:
                locations.append(loc)

    regex_locs = re.findall(
        r"\b(?:remote|onsite|hybrid|bangalore|bengaluru|chennai|delhi|mumbai|pune|hyderabad|singapore)\b",
        text, re.IGNORECASE
    )
    for loc in regex_locs:
        locations.append(loc.title())
    return list(set(locations)) if locations else ["Not mentioned"]

def extract_experience(text: str):
    text = text.lower()
    match_range = re.search(r"(\d+)\s*(?:-|\sto\s)\s*(\d+)\s*(?:years?|yrs?)", text)
    if match_range:
        start, end = match_range.groups()
        return [f"{start}-{end} years"]
    match_single = re.search(r"(\d+)\+?\s*(?:years?|yrs?)", text)
    if match_single:
        return [f"{match_single.group(1)} years"]
    if re.search(r"\bfresher(s)?\b", text):
        return ["Fresher"]
    return ["Not mentioned"]

def extract_role(text: str):
    patterns = [
        r"(?:role|position|job title)[:\-]\s*([A-Za-z ]+)",
        r"we are hiring (?:for )?an? ([A-Za-z ]+)",
        r"looking for an? ([A-Za-z ]+)"
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return [match.group(1).strip()]
    return ["Not mentioned"]

def extract_country(text: str):
    text_lower = text.lower()
    found = []
    # Exact match first
    for country in COUNTRY_LIST:
        if re.search(rf"\b{re.escape(country.lower())}\b", text_lower):
            found.append(country)
    # RAG/LLM fallback
    if HAS_RAG and not found:
        prompt = f"Job Description:\n{text}\n\nQuestion: Which country is this job based in? Answer only the country name."
        ans = generator(prompt, max_length=20, do_sample=False)[0]["generated_text"].strip()
        if ans and ans.lower() in [c.lower() for c in COUNTRY_LIST]:
            found.append(ans)
    return list(set(found)) if found else ["Not mentioned"]

LANGUAGES = set()
EDUCATION = set()
def collect_lang_edu(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() == "languages" and isinstance(v, list):
                for item in v:
                    if isinstance(item, str):
                        LANGUAGES.add(item.strip())
            elif k.lower() == "education" and isinstance(v, list):
                for item in v:
                    if isinstance(item, str):
                        EDUCATION.add(item.strip())
            else:
                collect_lang_edu(v)
    elif isinstance(obj, list):
        for item in obj:
            collect_lang_edu(item)
collect_lang_edu(data)

def extract_languages(text: str):
    found = [lang for lang in LANGUAGES if re.search(rf"\b{re.escape(lang)}\b", text, re.IGNORECASE)]
    return found if found else ["Not mentioned"]

def extract_education(text: str):
    edu_patterns = [
        r"\bB\.?Tech\b", r"\bM\.?Tech\b", r"\bB\.?E\b", r"\bM\.?E\b",
        r"\bB\.?Sc\b", r"\bM\.?Sc\b", r"\bPh\.?D\b",
        r"\bMBA\b", r"\bMCA\b", r"\bBCA\b", r"\bDiploma\b"
    ]
    found = []
    for pat in edu_patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        found.extend(matches)
    found.extend([edu for edu in EDUCATION if re.search(rf"\b{re.escape(edu)}\b", text, re.IGNORECASE)])
    found = list(set([f.upper() for f in found]))
    return found if found else ["Not mentioned"]

# ---------------- Pydantic ----------------
class ChatRequest(BaseModel):
    message: str
    mode: str = "exact"
    top_k: int = 10

# ---------------- Endpoints ----------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/extract-skills/")
async def extract_skills_endpoint(file: UploadFile = File(...), mode: str = "exact", top_k: int = 10):
    text = extract_text_from_file(file)
    if mode == "rag" and HAS_RAG:
        skills_found, llm_ans = rag_extract(text, top_k)
    else:
        skills_found = extract_skills_exact(text)
        llm_ans = None

    result = {
        "skills": skills_found,
        "languages": extract_languages(text),
        "education": extract_education(text),
        "Work Mode / City": extract_location(text),
        "country": extract_country(text) ,
        "experience": extract_experience(text),
        "role": extract_role(text)
         
    }
    if llm_ans:
        result["llm_summary"] = llm_ans
    return result

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if req.mode == "rag" and HAS_RAG:
        skills_found, llm_ans = rag_extract(req.message, req.top_k)
    else:
        skills_found = extract_skills_exact(req.message)
        llm_ans = None

    result = {
        "skills": skills_found,
        "languages": extract_languages(req.message),
        "education": extract_education(req.message),
        "Work Mode / City": extract_location(req.message),
        "country": extract_country(req.message) ,
        "experience": extract_experience(req.message),
        "role": extract_role(req.message)
        
    }
    if llm_ans:
        result["llm_summary"] = llm_ans
    return result
