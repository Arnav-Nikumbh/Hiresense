import os, re, json, uuid, io
from pathlib import Path
from typing import Optional, List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import aiofiles

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from nlp_engine import ResumeParser
from resume_scorer import ResumeScorer
import gemini_service as gemini

app = FastAPI(title="HireSense AI", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR   = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

parser = ResumeParser()
scorer = ResumeScorer()

static_path = BASE_DIR / "frontend" / "static"
static_path.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    resume_context: Optional[Dict] = None
    job_description: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = BASE_DIR / "frontend" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

@app.post("/api/parse-resume")
async def parse_resume(
    file: UploadFile = File(...),
    job_description: Optional[str] = Form(None)
):
    allowed = {"application/pdf","image/png","image/jpeg","image/jpg","image/tiff","text/plain"}
    content_type = file.content_type or ""
    filename     = file.filename or ""

    if content_type not in allowed and not filename.endswith(('.pdf','.png','.jpg','.jpeg','.txt')):
        raise HTTPException(400, "Unsupported file type.")

    file_id   = str(uuid.uuid4())
    ext       = Path(filename).suffix or ".tmp"
    save_path = UPLOAD_DIR / f"{file_id}{ext}"
    contents  = await file.read()

    async with aiofiles.open(save_path, "wb") as f:
        await f.write(contents)

    try:
        text = await extract_text(save_path, contents, content_type, filename)
        if not text or len(text.strip()) < 50:
            raise HTTPException(422, "Could not extract sufficient text from the file.")

        parsed = parser.parse(text)

        gemini_skills = None
        gemini_summary = None
        if gemini.is_available():
            try:
                gemini_skills  = gemini.extract_skills_with_gemini(text)
                gemini_summary = gemini.generate_resume_summary(text)
                if gemini_skills and "technical_skills" in gemini_skills:
                    _merge_gemini_skills(parsed, gemini_skills)
                if gemini_summary and not parsed.get("summary"):
                    parsed["summary"] = gemini_summary
                if gemini_skills:
                    parsed["experience_level"] = gemini_skills.get("experience_level","")
                    parsed["primary_role"]      = gemini_skills.get("primary_role","")
                    parsed["domain_expertise"]  = gemini_skills.get("domain_expertise",[])
            except Exception as e:
                gemini_skills = {"error": str(e)}

        score_data   = None
        gemini_match = None
        if job_description and job_description.strip():
            score_data = scorer.score(parsed, job_description)
            if gemini.is_available():
                try:
                    flat = parsed.get("skills",{}).get("all_technical_flat",[])
                    gemini_match = gemini.match_skills_with_gemini(text, job_description, flat)
                    if gemini_match and "match_percentage" in gemini_match:
                        blended = round(score_data["overall_score"]*0.4 + gemini_match["match_percentage"]*0.6, 1)
                        score_data["overall_score"]          = blended
                        score_data["gemini_match"]           = gemini_match
                        score_data["grade"]                  = scorer._get_grade(blended)
                        score_data["matched_skills"]         = gemini_match.get("matched_skills", score_data["matched_skills"])
                        score_data["missing_skills"]         = gemini_match.get("missing_critical_skills", score_data["missing_skills"])
                        score_data["hiring_recommendation"]  = gemini_match.get("hiring_recommendation","")
                        score_data["recommendation_reason"]  = gemini_match.get("recommendation_reason","")
                        score_data["transferable_skills"]    = gemini_match.get("transferable_skills",[])
                        score_data["strengths"]              = gemini_match.get("strengths",[])
                        score_data["concerns"]               = gemini_match.get("concerns",[])
                except Exception as e:
                    score_data["gemini_error"] = str(e)

        gemini_recs = None
        if gemini.is_available():
            try:
                gemini_recs = gemini.generate_recommendations_with_gemini(text, job_description or "", parsed)
            except Exception as e:
                gemini_recs = [{"error": str(e)}]

        return JSONResponse({
            "success": True,
            "file_id": file_id,
            "filename": filename,
            "extracted_text": text,
            "extracted_text_preview": text[:500],
            "parsed": parsed,
            "score": score_data,
            "gemini_skills": gemini_skills,
            "gemini_recommendations": gemini_recs,
            "gemini_available": gemini.is_available(),
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")
    finally:
        try: save_path.unlink(missing_ok=True)
        except: pass

def _merge_gemini_skills(parsed, gemini_skills):
    tech = parsed.setdefault("skills",{}).setdefault("technical",{})
    flat = set(s.lower() for s in parsed["skills"].get("all_technical_flat",[]))
    for cat, lst in gemini_skills.get("technical_skills",{}).items():
        if isinstance(lst, list):
            existing = tech.get(cat,[])
            for s in lst:
                if s.lower() not in flat:
                    existing.append(s.lower())
                    flat.add(s.lower())
            tech[cat] = existing
    parsed["skills"]["all_technical_flat"] = list(flat)
    parsed["skills"]["total_count"]        = len(flat)
    existing_soft = set(parsed["skills"].get("soft",[]))
    for s in gemini_skills.get("soft_skills",[]):
        existing_soft.add(s.lower())
    parsed["skills"]["soft"] = list(existing_soft)[:20]

async def extract_text(save_path, contents, content_type, filename):
    if content_type == "text/plain" or filename.endswith(".txt"):
        return contents.decode("utf-8", errors="ignore")
    if content_type.startswith("image/") or filename.endswith(('.png','.jpg','.jpeg')):
        try:
            import pytesseract; from PIL import Image
            return pytesseract.image_to_string(Image.open(io.BytesIO(contents)))
        except: pass
    if content_type == "application/pdf" or filename.endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(contents)) as pdf:
                t = "\n".join(p.extract_text() or "" for p in pdf.pages)
                if t.strip(): return t
        except: pass
        try:
            import PyPDF2
            r = PyPDF2.PdfReader(io.BytesIO(contents))
            t = "\n".join(p.extract_text() or "" for p in r.pages)
            if t.strip(): return t
        except: pass
        try:
            import pytesseract; from PIL import Image; from pdf2image import convert_from_bytes
            imgs = convert_from_bytes(contents, dpi=200)
            return "\n".join(pytesseract.image_to_string(i) for i in imgs)
        except: pass
    return ""

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not gemini.is_available():
        return JSONResponse({"reply": "⚠️ Gemini API key not configured. Add your key to the `.env` file to enable the AI assistant.", "error": True})
    try:
        history = [{"role": m.role, "content": m.content} for m in req.history]
        reply   = gemini.chat_with_resume_assistant(
            user_message=req.message,
            conversation_history=history,
            resume_context=req.resume_context,
            job_description=req.job_description,
        )
        return JSONResponse({"reply": reply, "error": False})
    except Exception as e:
        return JSONResponse({"reply": f"Error: {str(e)}", "error": True})

@app.get("/api/health")
async def health():
    return {"status":"ok","service":"HireSense AI v2","gemini_enabled":gemini.is_available(),"model":os.getenv("GEMINI_MODEL","gemini-2.0-flash")}


# ── JSON Resume Download ─────────────────────────────────────────
from fastapi.responses import Response as FastAPIResponse

@app.post("/api/export-json")
async def export_json(data: dict):
    """Return a clean structured JSON export of the parsed resume."""
    parsed = data.get("parsed", {})
    score  = data.get("score", None)
    filename = data.get("filename", "resume")

    export = {
        "meta": {
            "exported_by": "HireSense AI v2",
            "model":       os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20"),
            "filename":    filename,
        },
        "contact":             parsed.get("contact", {}),
        "summary":             parsed.get("summary", ""),
        "experience_level":    parsed.get("experience_level", ""),
        "primary_role":        parsed.get("primary_role", ""),
        "total_experience_years": parsed.get("total_experience_years", 0),
        "skills": {
            "technical":        parsed.get("skills", {}).get("technical", {}),
            "soft":             parsed.get("skills", {}).get("soft", []),
            "all_flat":         parsed.get("skills", {}).get("all_technical_flat", []),
            "domain_expertise": parsed.get("domain_expertise", []),
            "total_count":      parsed.get("skills", {}).get("total_count", 0),
        },
        "experience":     parsed.get("experience", []),
        "education":      parsed.get("education", []),
        "projects":       parsed.get("projects", []),
        "certifications": parsed.get("certifications", []),
        "match_score": {
            "overall":          score.get("overall_score") if score else None,
            "ats_score":        score.get("ats_score") if score else None,
            "grade":            score.get("grade") if score else None,
            "breakdown":        score.get("breakdown") if score else {},
            "matched_skills":   score.get("matched_skills", []) if score else [],
            "missing_skills":   score.get("missing_skills", []) if score else [],
            "hiring_recommendation": score.get("hiring_recommendation") if score else None,
            "recommendation_reason": score.get("recommendation_reason") if score else None,
        } if score else None,
    }

    safe_name = re.sub(r'[^\w\-.]', '_', filename.replace('.pdf','').replace('.txt',''))
    return FastAPIResponse(
        content     = json.dumps(export, indent=2, ensure_ascii=False),
        media_type  = "application/json",
        headers     = {"Content-Disposition": f'attachment; filename="{safe_name}_hiresense.json"'}
    )
