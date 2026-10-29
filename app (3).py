import io
import json
import os
import re
import time

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Nutrition Label Analyzer (ONQI)", page_icon="🥗", layout="wide")

# ------------------------------------------------------------------ config
def get_secret(name, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)

GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GEMINI_MODEL = get_secret("GEMINI_MODEL", "gemini-3.5-flash")

NUTRITION_FIELDS = ("serving_size", "servings_per_container", "calories", "total_fat",
                    "saturated_fat", "trans_fat", "cholesterol", "sodium",
                    "total_carbohydrate", "dietary_fiber", "total_sugars",
                    "added_sugars", "protein", "vitamin_d", "calcium", "iron", "potassium")

NUTRITION_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["nutrition_facts", "ingredients", "health_claims", "allergens",
                 "analysis_notes", "raw_text"],
    "properties": {
        "nutrition_facts": {
            "type": "object", "required": list(NUTRITION_FIELDS),
            "properties": {
                "serving_size": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                **{f: {"anyOf": [{"type": "number"}, {"type": "null"}]}
                   for f in NUTRITION_FIELDS if f != "serving_size"},
            },
        },
        "ingredients": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "health_claims": {"type": "array", "items": {"type": "string"}},
        "allergens": {"type": "array", "items": {"type": "string"}},
        "analysis_notes": {"type": "string"},
        "raw_text": {"type": "string"},
    },
}

NUTRITION_PROMPT = """Extract nutrition information from this label. Identify the
nutrition table first, then select ONE column. Prefer the Per Serving column for
ONQI; use Per 100 g only if no serving column exists and state that in analysis_notes.
Never mix columns. Preserve printed decimal values exactly, distinguish kcal from kJ,
and grams from milligrams. Never invent a value: return null when it cannot be read
confidently. Extract ingredients, health claims, allergens, and raw visible text when
available. The response schema is supplied separately; fill every schema field."""

# ------------------------------------------------------------------ extraction
@st.cache_resource
def get_client():
    if not GEMINI_API_KEY:
        return None
    from google import genai
    return genai.Client(api_key=GEMINI_API_KEY)

def prepare_image(img: Image.Image) -> bytes:
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > 1280:
        img.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()

def parse_gemini_json(text):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.I)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found in response")
    data = json.loads(cleaned[start:end + 1])
    if not isinstance(data.get("nutrition_facts"), dict):
        raise ValueError("Response has no nutrition_facts")
    return normalize(data)

def normalize(data):
    facts = data.get("nutrition_facts") or {}
    aliases = {"sugars": "total_sugars", "carbohydrates": "total_carbohydrate",
               "carbs": "total_carbohydrate", "fiber": "dietary_fiber"}
    clean = {}
    for k, v in facts.items():
        k = str(k).strip().lower().replace(" ", "_")
        k = aliases.get(k, k)
        if v in (None, "", "null", "N/A", "n/a"):
            continue
        if k != "serving_size" and isinstance(v, str):
            m = re.search(r"-?\d+(?:\.\d+)?", v.replace(",", ""))
            v = float(m.group()) if m else v
        clean[k] = v
    data["nutrition_facts"] = clean
    data["ingredients"] = data.get("ingredients") or ""
    data["health_claims"] = data.get("health_claims") or []
    data["allergens"] = data.get("allergens") or []
    data["raw_text"] = data.get("raw_text") or ""
    return data

def analyze_with_gemini(img: Image.Image):
    client = get_client()
    if client is None:
        raise RuntimeError("GEMINI_API_KEY is not set")
    from google import genai
    part = genai.types.Part.from_bytes(data=prepare_image(img), mime_type="image/jpeg")
    last_err = None
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[NUTRITION_PROMPT, part],
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=NUTRITION_RESPONSE_SCHEMA,
                    temperature=0,
                ),
            )
            data = parse_gemini_json(resp.text)
            data["analysis_notes"] = data.get("analysis_notes") or "Analyzed using Gemini Vision AI"
            data["extraction_reliable"] = True
            return data
        except Exception as e:
            last_err = e
            msg = str(e).upper()
            if ("503" in msg or "UNAVAILABLE" in msg or "RESOURCE_EXHAUSTED" in msg) and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            break
    raise RuntimeError(f"Gemini analysis failed: {last_err}")

def analyze_with_tesseract(img: Image.Image):
    """Optional OCR fallback (needs the tesseract-ocr system package)."""
    import pytesseract
    text = pytesseract.image_to_string(img.convert("L"), config="--psm 6")
    low = text.lower()

    def grab(*pats):
        for p in pats:
            m = re.search(p, low)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
        return None

    facts = {
        "calories": grab(r"calories[:\s]*(\d+)", r"energy[:\s]*(\d+)"),
        "total_fat": grab(r"total\s*fat[:\s]*(\d+\.?\d*)"),
        "saturated_fat": grab(r"saturated\s*fat[:\s]*(\d+\.?\d*)"),
        "trans_fat": grab(r"trans\s*fat[:\s]*(\d+\.?\d*)"),
        "cholesterol": grab(r"cholesterol[:\s]*(\d+)"),
        "sodium": grab(r"sodium[:\s]*(\d+)"),
        "total_carbohydrate": grab(r"total\s*carb\w*[:\s]*(\d+\.?\d*)"),
        "dietary_fiber": grab(r"fib(?:er|re)[:\s]*(\d+\.?\d*)"),
        "total_sugars": grab(r"sugars?[:\s]*(\d+\.?\d*)"),
        "protein": grab(r"protein[:\s]*(\d+\.?\d*)"),
    }
    data = normalize({"nutrition_facts": {k: v for k, v in facts.items() if v is not None},
                      "raw_text": text})
    data["extraction_reliable"] = validate_ocr(data)
    data["analysis_notes"] = ("Extracted using Tesseract OCR (fallback)" if data["extraction_reliable"]
                              else "Could not confidently read this image. Please upload a clearer label.")
    if not data["extraction_reliable"]:
        data["nutrition_facts"] = {}
    return data

def validate_ocr(data):
    f = data.get("nutrition_facts", {})
    if not f:
        return False
    n = lambda k: float(f.get(k, 0) or 0)
    for k in ("total_fat", "saturated_fat", "trans_fat", "total_carbohydrate",
              "dietary_fiber", "total_sugars", "protein"):
        if not 0 <= n(k) <= 100:
            return False
    if n("calories") > 3000 or n("sodium") > 10000 or n("cholesterol") > 3000:
        return False
    if n("saturated_fat") > n("total_fat") or n("dietary_fiber") > n("total_carbohydrate"):
        return False
    if n("total_fat") + n("total_carbohydrate") + n("protein") > 125:
        return False
    return any(n(k) > 0 for k in ("calories", "total_fat", "total_carbohydrate", "protein"))

# ------------------------------------------------------------------ ONQI scoring
def _f(v, d=0.0):
    try:
        return d if v in (None, "", "null") else float(v)
    except (TypeError, ValueError):
        return d

def calculate_onqi(facts):
    """Returns (score, breakdown rows). Score 1-100, higher is better; 0 = no data."""
    sat, trans = _f(facts.get("saturated_fat")), _f(facts.get("trans_fat"))
    sodium, chol = _f(facts.get("sodium")), _f(facts.get("cholesterol"))
    sugar = max(_f(facts.get("sugars")), _f(facts.get("total_sugars")))
    fiber, protein = _f(facts.get("dietary_fiber")), _f(facts.get("protein"))
    cal, fat = _f(facts.get("calories")), _f(facts.get("total_fat"))

    if not any([cal > 0, fat > 0, sat > 0, sodium > 0, sugar > 0, protein > 0, fiber > 0]):
        return 0, []

    rows = []
    def add(label, value, points):
        rows.append({"Factor": label, "Value": value, "Points": round(points, 1)})

    if sat > 0:     add("Saturated fat",  f"{sat} g",     -min(15, sat / 5 * 5))
    if trans > 0:   add("Trans fat",      f"{trans} g",   -min(15, trans * 10))
    if sodium > 0:  add("Sodium",         f"{sodium} mg", -min(10, sodium / 600 * 5))
    if sugar > 0:   add("Sugar",          f"{sugar} g",   -min(10, sugar / 12 * 5))
    if chol > 0:    add("Cholesterol",    f"{chol} mg",   -min(5, chol / 100 * 2.5))
    if cal > 200:   add("High calories",  f"{cal} kcal",  -min(5, (cal - 200) / 200 * 2.5))
    if fiber > 0:   add("Dietary fiber",  f"{fiber} g",   min(15, fiber / 3 * 5))
    if protein > 0: add("Protein",        f"{protein} g", min(15, protein / 10 * 5))
    if 0 < fat <= 5:     add("Low fat",    f"{fat} g",    max(0, 10 - fat * 2))
    if 0 < sugar <= 5:   add("Low sugar",  f"{sugar} g",  max(0, 5 - sugar))
    if 0 < sodium <= 200: add("Low sodium", f"{sodium} mg", max(0, 5 - sodium / 40))

    score = 50 + sum(r["Points"] for r in rows)
    return max(1, min(100, round(score))), rows

def rating_for(score):
    if score >= 80: return "Excellent 🏆", "A highly nutritious choice with an excellent balance of nutrients.", "success"
    if score >= 60: return "Good 👍", "A healthy choice with good nutritional value.", "success"
    if score >= 40: return "Fair ⚠️", "Moderate nutritional value. Consume in moderation.", "warning"
    if score >= 20: return "Poor 👎", "Limited nutritional value; may not be the healthiest choice.", "error"
    return "Very Poor ❌", "Not nutritionally beneficial. Consider healthier alternatives.", "error"

def health_analysis(facts, score):
    sat, trans = _f(facts.get("saturated_fat")), _f(facts.get("trans_fat"))
    sodium, chol = _f(facts.get("sodium")), _f(facts.get("cholesterol"))
    sugar = max(_f(facts.get("sugars")), _f(facts.get("total_sugars")))
    fiber, protein, cal = _f(facts.get("dietary_fiber")), _f(facts.get("protein")), _f(facts.get("calories"))
    pros, cons, recs, alts = [], [], [], []

    if fiber >= 5: pros.append(f"High in fiber ({fiber} g) – excellent for digestive health")
    elif fiber >= 3: pros.append(f"Good fiber content ({fiber} g) – supports healthy digestion")
    if protein >= 15: pros.append(f"Excellent protein source ({protein} g)")
    elif protein >= 10: pros.append(f"Good protein content ({protein} g) – helps with satiety")
    if sat <= 2: pros.append(f"Low in saturated fat ({sat} g)")
    if sugar <= 5: pros.append(f"Low in sugar ({sugar} g)")
    if 0 < sodium <= 200: pros.append(f"Low sodium ({sodium} mg)")
    if trans == 0: pros.append("No trans fat")
    if 0 < cal <= 150: pros.append(f"Low calorie ({cal} kcal)")

    if sat > 5: cons.append(f"High in saturated fat ({sat} g) – may increase heart disease risk")
    elif sat > 3: cons.append(f"Moderate saturated fat ({sat} g)")
    if trans > 0: cons.append(f"Contains trans fat ({trans} g) – best avoided")
    if sodium > 600: cons.append(f"High in sodium ({sodium} mg) – may affect blood pressure")
    elif sodium > 400: cons.append(f"Moderate sodium ({sodium} mg)")
    if sugar > 15: cons.append(f"High in sugar ({sugar} g)")
    elif sugar > 10: cons.append(f"Moderate sugar ({sugar} g)")
    if chol > 60: cons.append(f"Contains cholesterol ({chol} mg)")
    if cal > 300: cons.append(f"High calorie content ({cal} kcal) – consider portion size")

    if score < 60:
        recs += ["Consume in moderation", "Balance with healthier foods through the day"]
    if sodium > 400: recs.append("Drink plenty of water to offset the sodium")
    if sugar > 10: recs.append("Pair with protein or fiber to slow sugar absorption")
    if sat > 3: recs.append("Balance with unsaturated fats such as avocado or nuts")
    if fiber < 3: recs.append("Add fiber-rich foods to the meal")
    if protein < 5: recs.append("Pair with a protein source for better satiety")

    if score < 70:
        alts += ["Fresh fruits and vegetables", "Whole-grain options",
                 "Lean proteins (chicken, fish, beans)", "Nuts and seeds in moderation"]
    if sugar > 15: alts.append("Sugar-free or naturally sweetened alternatives")
    if sodium > 500: alts.append("Low-sodium versions of similar products")
    return pros, cons, recs, alts

# ------------------------------------------------------------------ UI helpers
def show_result(data, source_img=None):
    facts = data.get("nutrition_facts", {})
    if data.get("extraction_reliable") is False or not facts:
        st.error(data.get("analysis_notes") or "Unable to read nutrition data. Upload a clearer, straight-on image.")
        return

    score, rows = calculate_onqi(facts)
    if score == 0:
        st.error("No reliable nutrition values found, so no ONQI score was calculated.")
        return
    rating, summary, kind = rating_for(score)

    left, right = st.columns([1, 2])
    with left:
        if source_img is not None:
            st.image(source_img, width="stretch")
        st.metric("ONQI score", f"{score} / 100")
        st.progress(score / 100)
        getattr(st, kind)(f"**{rating}**\n\n{summary}")
    with right:
        st.subheader("Extracted nutrition facts")
        table = pd.DataFrame({"Nutrient": [k.replace("_", " ").title() for k in facts],
                              "Value": [str(v) for v in facts.values()]})
        st.dataframe(table, hide_index=True, width="stretch")
        if data.get("analysis_notes"):
            st.caption(data["analysis_notes"])

    st.subheader("How the score was calculated")
    st.caption("Start at 50, subtract penalties, add bonuses, clamp to 1-100.")
    br = pd.DataFrame(rows)
    st.bar_chart(br.set_index("Factor")["Points"])
    st.dataframe(br, hide_index=True, width="stretch")

    pros, cons, recs, alts = health_analysis(facts, score)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("✅ Pros")
        for p in pros or ["None identified"]: st.write(f"- {p}")
        st.subheader("💡 Recommendations")
        for r in recs or ["No special recommendations"]: st.write(f"- {r}")
    with c2:
        st.subheader("⚠️ Cons")
        for c in cons or ["None identified"]: st.write(f"- {c}")
        if alts:
            st.subheader("🔄 Healthier alternatives")
            for a in alts: st.write(f"- {a}")

    if data.get("ingredients"):
        with st.expander("Ingredients"): st.write(data["ingredients"])
    if data.get("allergens"):
        st.info("Allergens detected: " + ", ".join(data["allergens"]))
    if data.get("health_claims"):
        st.info("Label claims: " + ", ".join(data["health_claims"]))
    if data.get("raw_text"):
        with st.expander("Raw text read from the label"): st.text(data["raw_text"])

# ------------------------------------------------------------------ pages
st.sidebar.title("🥗 ONQI Analyzer")
page = st.sidebar.radio("Go to", ["Analyze a label", "Manual entry", "How it works"])
st.sidebar.caption("AI extraction: " + ("✅ Gemini connected" if GEMINI_API_KEY else "⚠️ no API key – OCR / manual only"))

if page == "Analyze a label":
    st.title("🥗 Nutrition Label Analyzer")
    st.write("Upload a photo of a nutrition facts label to get an **Overall Nutritional Quality Index (ONQI)**, "
             "pros and cons, recommendations and healthier alternatives.")
    method = st.radio("Extraction method", ["Gemini AI (best)", "Tesseract OCR (fallback)"],
                      horizontal=True, index=0 if GEMINI_API_KEY else 1)
    file = st.file_uploader("Nutrition label image", type=["jpg", "jpeg", "png", "webp"])
    if file and st.button("Analyze", type="primary"):
        img = Image.open(file)
        with st.spinner("Reading the label…"):
            try:
                if method.startswith("Gemini"):
                    try:
                        data = analyze_with_gemini(img)
                    except Exception as e:
                        st.warning(f"{e}. Trying OCR fallback…")
                        data = analyze_with_tesseract(img)
                else:
                    data = analyze_with_tesseract(img)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()
        show_result(data, img)

elif page == "Manual entry":
    st.title("✍️ Manual entry")
    st.write("No image? Type the values from the label (per serving).")
    c1, c2, c3 = st.columns(3)
    with c1:
        cal = st.number_input("Calories (kcal)", 0.0, 3000.0, 0.0)
        fat = st.number_input("Total fat (g)", 0.0, 100.0, 0.0)
        sat = st.number_input("Saturated fat (g)", 0.0, 100.0, 0.0)
        trans = st.number_input("Trans fat (g)", 0.0, 100.0, 0.0)
    with c2:
        chol = st.number_input("Cholesterol (mg)", 0.0, 3000.0, 0.0)
        sodium = st.number_input("Sodium (mg)", 0.0, 10000.0, 0.0)
        carbs = st.number_input("Total carbohydrate (g)", 0.0, 100.0, 0.0)
    with c3:
        fiber = st.number_input("Dietary fiber (g)", 0.0, 100.0, 0.0)
        sugar = st.number_input("Total sugars (g)", 0.0, 100.0, 0.0)
        protein = st.number_input("Protein (g)", 0.0, 100.0, 0.0)
    if st.button("Calculate", type="primary"):
        facts = {"calories": cal, "total_fat": fat, "saturated_fat": sat, "trans_fat": trans,
                 "cholesterol": chol, "sodium": sodium, "total_carbohydrate": carbs,
                 "dietary_fiber": fiber, "total_sugars": sugar, "protein": protein}
        show_result({"nutrition_facts": facts, "extraction_reliable": True,
                     "analysis_notes": "Entered manually"})

else:
    st.title("📘 How it works")
    st.markdown("""
**Pipeline:** image → Gemini Vision (structured JSON, temperature 0, retries on 503/quota errors) →
Tesseract OCR fallback with plausibility checks → ONQI score → health analysis.

**Extraction rules:** one column only (Per Serving preferred, Per 100 g if that is all there is),
printed decimals preserved, kcal vs kJ and g vs mg distinguished, `null` instead of guessing.

**OCR safety checks:** OCR output is rejected (no score) if a gram value is outside 0-100, calories exceed 3000,
sodium exceeds 10,000 mg, saturated fat exceeds total fat, fiber exceeds carbohydrates, or macros sum above 125 g.

### ONQI (1-100, higher is better)
The score starts at **50**, then penalties are subtracted and bonuses added.
""")
    st.table(pd.DataFrame({
        "Factor": ["Saturated fat", "Trans fat", "Sodium", "Sugar", "Cholesterol", "Calories > 200",
                   "Dietary fiber", "Protein", "Low fat (≤5 g)", "Low sugar (≤5 g)", "Low sodium (≤200 mg)"],
        "Effect": ["−5 per 5 g (max −15)", "−10 per g (max −15)", "−5 per 600 mg (max −10)",
                   "−5 per 12 g (max −10)", "−2.5 per 100 mg (max −5)", "−2.5 per 200 kcal over 200 (max −5)",
                   "+5 per 3 g (max +15)", "+5 per 10 g (max +15)", "+(10 − 2×fat g)",
                   "+(5 − sugar g)", "+(5 − sodium/40)"],
    }).set_index("Factor"))
    st.table(pd.DataFrame({
        "Score": ["80-100", "60-79", "40-59", "20-39", "1-19"],
        "Rating": ["Excellent", "Good", "Fair", "Poor", "Very Poor"],
    }).set_index("Score"))
    st.caption("ONQI is a simplified, NuVal-inspired index for education, not medical or dietary advice. "
               "Scores use per-serving values, so serving size affects comparisons between products.")
