"""
gemini_service.py  —  HireSense AI  v3.2
Uses google-genai SDK (google.genai).

KEY FIXES:
- API_KEY read fresh from os.environ on every call (no stale module constants)
- Chat uses generate_content() NOT chats.create() — avoids v1beta model mismatch
- Model: gemini-2.5-flash-preview-05-20 works on v1 (generate_content) but NOT v1beta (chats)
"""

import os, json, re
from typing import Optional, Dict, List, Any
from pathlib import Path
from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH, override=False)  # override=False: dont stomp main.py load


# ── Always read live from environment, never cache as constants ──
def _api_key() -> str:
    return os.environ.get("GEMINI_API_KEY", "").strip()

def _model() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()

def _enabled() -> bool:
    return os.environ.get("GEMINI_ENABLED", "true").lower() == "true"

def is_available() -> bool:
    key = _api_key()
    return _enabled() and bool(key) and key != "your_gemini_api_key_here"


_client = None

def _get_client():
    global _client
    key = _api_key()
    if not key or key == "your_gemini_api_key_here":
        raise ValueError(
            f"GEMINI_API_KEY missing or placeholder. Edit {_ENV_PATH}"
        )
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=key)
    return _client


def _call(prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
    """Single-turn call using generate_content (v1 API — works with all models)."""
    from google.genai import types
    client = _get_client()
    response = client.models.generate_content(
        model    = _model(),
        contents = prompt,
        config   = types.GenerateContentConfig(
            max_output_tokens = max_tokens,
            temperature       = temperature,
        ),
    )
    return response.text.strip()


def _parse_json(raw: str):
    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*',     '', raw)
    return json.loads(raw.strip())


# ─────────────────────────────────────────────────────────────────
# 1.  SKILL EXTRACTION
# ─────────────────────────────────────────────────────────────────
def extract_skills_with_gemini(resume_text: str) -> Dict[str, Any]:
    prompt = f"""You are a technical recruiter AI. Analyze this resume and extract ALL skills.

RESUME TEXT:
\"\"\"
{resume_text[:3000]}
\"\"\"

Return ONLY a valid JSON object (no markdown, no explanation):
{{
  "technical_skills": {{
    "languages":    ["programming languages"],
    "frameworks":   ["frameworks and libraries"],
    "databases":    ["databases"],
    "cloud_devops": ["cloud/devops tools"],
    "ai_ml":        ["AI/ML skills"],
    "tools":        ["dev tools"]
  }},
  "soft_skills":             ["soft skills"],
  "domain_expertise":        ["domain areas e.g. fintech, healthcare"],
  "certifications_detected": ["certifications"],
  "experience_level":        "fresher|junior|mid|senior|lead",
  "primary_role":            "most likely job role"
}}"""
    try:
        return _parse_json(_call(prompt, 800))
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────
# 2.  SKILL MATCHING
# ─────────────────────────────────────────────────────────────────
def match_skills_with_gemini(resume_text: str, job_description: str, parsed_skills: List[str]) -> Dict[str, Any]:
    prompt = f"""You are an expert technical recruiter. Compare this candidate to the job description.

JOB DESCRIPTION:
\"\"\"
{job_description[:2000]}
\"\"\"

CANDIDATE SKILLS: {', '.join(parsed_skills[:40])}

RESUME:
\"\"\"
{resume_text[:2000]}
\"\"\"

Return ONLY valid JSON (no markdown):
{{
  "match_percentage":        <0-100>,
  "matched_skills":          ["skills from JD the candidate has"],
  "missing_critical_skills": ["must-have skills missing"],
  "missing_nice_to_have":    ["nice-to-have skills missing"],
  "transferable_skills":     ["equivalent/related skills candidate has"],
  "strengths":               ["2-3 specific strengths for this role"],
  "concerns":                ["1-2 genuine concerns"],
  "hiring_recommendation":   "strong_yes|yes|maybe|no",
  "recommendation_reason":   "one sentence"
}}"""
    try:
        return _parse_json(_call(prompt, 700))
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────
# 3.  RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────
def generate_recommendations_with_gemini(resume_text: str, job_description: str = "", parsed: Dict = None) -> List[Dict]:
    jd_section = f"\nTARGET JOB:\n\"\"\"\n{job_description[:1200]}\n\"\"\"\n" if job_description else ""
    prompt = f"""You are a professional resume coach for students.

RESUME:
\"\"\"
{resume_text[:3000]}
\"\"\"
{jd_section}
Return ONLY a valid JSON array (no markdown):
[{{
  "category": "impact|skills|formatting|keywords|experience|education|projects|summary|linkedin",
  "priority": "high|medium|low",
  "title":    "short title max 8 words",
  "problem":  "what is wrong (1-2 sentences)",
  "fix":      "exactly what to do (2-3 sentences)",
  "example":  "before/after example or empty string",
  "impact":   "why this matters"
}}]

Generate 6-8 specific recommendations for THIS resume."""
    try:
        result = _parse_json(_call(prompt, 1200))
        return result if isinstance(result, list) else []
    except Exception as e:
        return [{"error": str(e)}]


# ─────────────────────────────────────────────────────────────────
# 4.  CHATBOT
#
#  WHY NOT chats.create():
#    chats.create() internally uses the v1beta API endpoint.
#    Preview model names like gemini-2.5-flash-preview-05-20 return 404
#    on v1beta. generate_content() uses v1 and works fine.
#
#  SOLUTION:
#    Build the full conversation as a structured Contents list and call
#    generate_content() directly. Identical behaviour, no API version issues.
# ─────────────────────────────────────────────────────────────────
def chat_with_resume_assistant(
    user_message:         str,
    conversation_history: List[Dict],
    resume_context:       Optional[Dict] = None,
    job_description:      Optional[str]  = None,
) -> str:
    from google.genai import types

    client = _get_client()

    # ── System context string ─────────────────────────────────────
    ctx_parts = []
    if resume_context:
        contact    = resume_context.get("contact", {})
        skills     = resume_context.get("skills", {})
        exp        = resume_context.get("experience", [])
        edu        = resume_context.get("education", [])
        skill_list = skills.get("all_technical_flat", [])[:30]
        exp_years  = resume_context.get("total_experience_years", 0)

        ctx_parts.append(
            "CANDIDATE PROFILE (from uploaded resume):\n"
            f"- Name: {contact.get('name', 'Unknown')}\n"
            f"- Experience: ~{exp_years} years\n"
            f"- Level: {resume_context.get('experience_level') or 'Not specified'}\n"
            f"- Primary role: {resume_context.get('primary_role') or 'Not detected'}\n"
            f"- Skills: {', '.join(skill_list) or 'None detected'}\n"
            f"- Education: {', '.join(e.get('degree','') for e in edu[:2]) or 'Not detected'}\n"
            f"- Recent roles: {', '.join(e.get('title','') for e in exp[:3]) or 'Not detected'}"
        )
    if job_description:
        ctx_parts.append(f"TARGET JOB (excerpt):\n{job_description[:500]}")

    system_text = (
        "You are HireSense AI Assistant — a friendly expert resume coach for students "
        "and early-career professionals.\n\n"
        "Personality: Encouraging but honest. Specific not vague. Concise (under 200 words "
        "unless asked for detail). Use bullet points for lists.\n\n"
        "Expertise: Resume writing, ATS optimization, skill gap analysis, interview prep, "
        "LinkedIn, cover letters, career guidance.\n\n"
        + ("\n\n".join(ctx_parts) if ctx_parts else
           "No resume uploaded yet — encourage the user to upload one for specific advice.")
        + "\n\nIMPORTANT: Always refer to the candidate's actual profile data above when answering."
    )

    # ── Build Contents list: system → history → new user message ─
    # Using generate_content with a Contents list gives us full multi-turn control
    contents: List[types.Content] = []

    # Inject system as a first user/model pair (standard technique for generate_content)
    contents.append(types.Content(
        role  = "user",
        parts = [types.Part(text=f"[SYSTEM INSTRUCTIONS]\n{system_text}")]
    ))
    contents.append(types.Content(
        role  = "model",
        parts = [types.Part(text="Understood. I'm ready to help as HireSense AI Assistant.")]
    ))

    # Add conversation history (last 12 messages)
    for msg in conversation_history[-12:]:
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append(types.Content(
            role  = role,
            parts = [types.Part(text=msg["content"])]
        ))

    # Add current user message
    contents.append(types.Content(
        role  = "user",
        parts = [types.Part(text=user_message)]
    ))

    # ── Call generate_content (v1 API — no model version issues) ─
    from google.genai import types as gtypes
    response = client.models.generate_content(
        model    = _model(),
        contents = contents,
        config   = gtypes.GenerateContentConfig(
            max_output_tokens = 1000,
            temperature       = 0.7,
        ),
    )
    return response.text


# ─────────────────────────────────────────────────────────────────
# 5.  SUMMARY
# ─────────────────────────────────────────────────────────────────
def generate_resume_summary(resume_text: str) -> str:
    prompt = f"""Write a compelling 2-3 sentence professional summary for this candidate.
Third person, specific, no fluff.

RESUME:
\"\"\"
{resume_text[:2500]}
\"\"\"

Return ONLY the summary text, no labels, no markdown."""
    try:
        return _call(prompt, 150)
    except Exception:
        return ""