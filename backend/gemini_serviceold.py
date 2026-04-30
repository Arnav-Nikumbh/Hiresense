"""
gemini_service.py  —  HireSense AI  v3
Uses the NEW google-genai SDK (google.genai) with gemini-2.5-flash-preview-05-20.
The old google.generativeai library is deprecated — this file does NOT use it.
"""

import os, json, re
from typing import Optional, Dict, List, Any
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GEMINI_ENABLED = os.getenv("GEMINI_ENABLED", "true").lower() == "true"
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20")
API_KEY        = os.getenv("GEMINI_API_KEY", "")

_client = None


def _get_client():
    global _client
    if _client is None:
        if not API_KEY or API_KEY == "your_gemini_api_key_here":
            raise ValueError("GEMINI_API_KEY not set in .env file")
        from google import genai
        _client = genai.Client(api_key=API_KEY)
    return _client


def _call(prompt: str, max_tokens: int = 1024) -> str:
    """Single-turn Gemini call using new SDK."""
    from google import genai
    from google.genai import types

    client   = _get_client()
    response = client.models.generate_content(
        model   = GEMINI_MODEL,
        contents= prompt,
        config  = types.GenerateContentConfig(
            max_output_tokens = max_tokens,
            temperature       = 0.3,
        ),
    )
    return response.text.strip()


def _parse_json(raw: str):
    """Strip markdown fences and parse JSON."""
    raw = re.sub(r'^```json\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'^```\s*',     '', raw, flags=re.MULTILINE)
    return json.loads(raw.strip())


def is_available() -> bool:
    return GEMINI_ENABLED and bool(API_KEY) and API_KEY != "your_gemini_api_key_here"


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
    "languages":   ["programming languages"],
    "frameworks":  ["frameworks and libraries"],
    "databases":   ["databases"],
    "cloud_devops":["cloud/devops tools"],
    "ai_ml":       ["AI/ML skills"],
    "tools":       ["dev tools"]
  }},
  "soft_skills":             ["soft skills mentioned"],
  "domain_expertise":        ["domain areas like fintech, healthcare etc"],
  "certifications_detected": ["certifications mentioned"],
  "experience_level":        "fresher|junior|mid|senior|lead",
  "primary_role":            "most likely job role"
}}

Be thorough. Include skills from project descriptions too."""

    try:
        return _parse_json(_call(prompt, 800))
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────
# 2.  SKILL MATCHING
# ─────────────────────────────────────────────────────────────────
def match_skills_with_gemini(resume_text: str, job_description: str, parsed_skills: List[str]) -> Dict[str, Any]:
    prompt = f"""You are an expert technical recruiter. Compare the candidate's resume to the job description.

JOB DESCRIPTION:
\"\"\"
{job_description[:2000]}
\"\"\"

CANDIDATE'S DETECTED SKILLS: {', '.join(parsed_skills[:40])}

RESUME EXCERPT:
\"\"\"
{resume_text[:2000]}
\"\"\"

Return ONLY a valid JSON object (no markdown):
{{
  "match_percentage":        <integer 0-100>,
  "matched_skills":          ["skills from JD the candidate has"],
  "missing_critical_skills": ["must-have skills missing"],
  "missing_nice_to_have":    ["nice-to-have skills missing"],
  "transferable_skills":     ["candidate skills equivalent to JD requirements"],
  "strengths":               ["2-3 specific strengths for this role"],
  "concerns":                ["1-2 genuine concerns or gaps"],
  "hiring_recommendation":   "strong_yes|yes|maybe|no",
  "recommendation_reason":   "one sentence explanation"
}}"""

    try:
        return _parse_json(_call(prompt, 700))
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────
# 3.  RESUME RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────
def generate_recommendations_with_gemini(resume_text: str, job_description: str = "", parsed: Dict = None) -> List[Dict]:
    jd_section = f"\nTARGET JOB DESCRIPTION:\n\"\"\"\n{job_description[:1500]}\n\"\"\"\n" if job_description else ""

    prompt = f"""You are a professional resume coach helping a student improve their resume.

RESUME TEXT:
\"\"\"
{resume_text[:3000]}
\"\"\"
{jd_section}
Give specific, actionable resume improvement recommendations.

Return ONLY a valid JSON array (no markdown):
[
  {{
    "category": "impact|skills|formatting|keywords|experience|education|projects|summary|linkedin",
    "priority": "high|medium|low",
    "title":    "short title (max 8 words)",
    "problem":  "what is wrong or missing (1-2 sentences)",
    "fix":      "exactly what to do (2-3 sentences)",
    "example":  "concrete before/after example or empty string",
    "impact":   "why this matters for getting hired"
  }}
]

Generate 6-8 recommendations specific to THIS resume, not generic advice."""

    try:
        result = _parse_json(_call(prompt, 1200))
        return result if isinstance(result, list) else []
    except Exception as e:
        return [{"error": str(e)}]


# ─────────────────────────────────────────────────────────────────
# 4.  CHATBOT  (multi-turn, context-aware)
# ─────────────────────────────────────────────────────────────────
def chat_with_resume_assistant(
    user_message:         str,
    conversation_history: List[Dict],
    resume_context:       Optional[Dict] = None,
    job_description:      Optional[str]  = None,
) -> str:
    """
    Multi-turn resume coaching chatbot.
    Uses google.genai ChatSession for proper conversation management.
    """
    from google import genai
    from google.genai import types

    client = _get_client()

    # Build system instruction
    context_parts = []

    if resume_context:
        contact    = resume_context.get("contact", {})
        skills     = resume_context.get("skills", {})
        exp        = resume_context.get("experience", [])
        edu        = resume_context.get("education", [])
        skill_list = skills.get("all_technical_flat", [])[:30]
        exp_years  = resume_context.get("total_experience_years", 0)
        primary    = resume_context.get("primary_role", "")
        level      = resume_context.get("experience_level", "")

        context_parts.append(
            f"CANDIDATE PROFILE (from uploaded resume):\n"
            f"- Name: {contact.get('name', 'Unknown')}\n"
            f"- Experience: ~{exp_years} years\n"
            f"- Level: {level or 'Not specified'}\n"
            f"- Primary role: {primary or 'Not detected'}\n"
            f"- Top skills: {', '.join(skill_list) or 'None detected'}\n"
            f"- Education: {', '.join(e.get('degree','') for e in edu[:2]) or 'Not detected'}\n"
            f"- Recent roles: {', '.join(e.get('title','') for e in exp[:3]) or 'Not detected'}"
        )

    if job_description:
        context_parts.append(f"TARGET JOB DESCRIPTION (first 800 chars):\n{job_description[:800]}")

    context_block = "\n\n".join(context_parts)

    system_instruction = f"""You are HireSense AI Assistant — a friendly, expert resume coach and career advisor for students and early-career professionals.

Personality: Encouraging but honest. Specific not vague. Use examples. Keep responses concise (under 200 words unless asked for detail). Use bullet points for lists.

Your expertise:
- Resume writing and ATS optimization
- Skill gap analysis and learning roadmaps
- Interview preparation and career guidance
- LinkedIn profile optimization
- Cover letter writing

{context_block}

Rules:
- Always refer to the candidate's actual resume data when it is available above
- Give specific, actionable advice — not generic tips
- If asked about something unrelated to career/resume, politely redirect
- Be encouraging but realistic about gaps"""

    # Build conversation history for the SDK
    # google.genai uses Content objects with role "user" / "model"
    history_contents = []
    for msg in conversation_history[-10:]:  # last 10 msgs to stay in token budget
        sdk_role = "model" if msg["role"] == "assistant" else "user"
        history_contents.append(
            types.Content(role=sdk_role, parts=[types.Part(text=msg["content"])])
        )

    # Start a chat session with history
    chat_session = client.chats.create(
        model   = GEMINI_MODEL,
        config  = types.GenerateContentConfig(
            system_instruction = system_instruction,
            max_output_tokens  = 512,
            temperature        = 0.7,
        ),
        history = history_contents,
    )

    response = chat_session.send_message(user_message)
    return response.text.strip()


# ─────────────────────────────────────────────────────────────────
# 5.  RESUME SUMMARY
# ─────────────────────────────────────────────────────────────────
def generate_resume_summary(resume_text: str) -> str:
    prompt = f"""Read this resume and write a compelling 2-3 sentence professional summary for the candidate.
Write in third person. Be specific about their skills and experience level. No fluff.

RESUME:
\"\"\"
{resume_text[:2500]}
\"\"\"

Return ONLY the summary text, no labels, no markdown."""
    try:
        return _call(prompt, 150)
    except Exception:
        return ""
