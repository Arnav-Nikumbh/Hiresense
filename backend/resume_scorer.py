import re
import math
from typing import Dict, List, Any, Optional
from collections import Counter


class ResumeScorer:
    """Score resume against job description using TF-IDF and skill matching."""
    
    def score(self, parsed: Dict, job_description: str) -> Dict[str, Any]:
        jd_skills = self._extract_jd_skills(job_description)
        jd_keywords = self._extract_keywords(job_description)
        resume_skills_flat = set(s.lower() for s in parsed.get("skills", {}).get("all_technical_flat", []))
        resume_soft = set(s.lower() for s in parsed.get("skills", {}).get("soft", []))
        
        # Skill match score
        matched_skills = [s for s in jd_skills if s.lower() in resume_skills_flat]
        missing_skills = [s for s in jd_skills if s.lower() not in resume_skills_flat]
        skill_score = (len(matched_skills) / max(len(jd_skills), 1)) * 100
        
        # Keyword relevance (TF-IDF cosine similarity)
        resume_text = self._parsed_to_text(parsed)
        keyword_score = self._tfidf_similarity(resume_text, job_description) * 100
        
        # Experience relevance
        exp_years = parsed.get("total_experience_years", 0)
        req_years = self._extract_required_years(job_description)
        exp_score = min(100, (exp_years / max(req_years, 1)) * 100) if req_years else 70
        
        # Education relevance
        edu_score = self._score_education(parsed.get("education", []), job_description)
        
        # Overall weighted score
        overall = (
            skill_score * 0.40 +
            keyword_score * 0.30 +
            exp_score * 0.20 +
            edu_score * 0.10
        )
        
        # ATS compatibility score
        ats_score = self._calculate_ats_score(parsed, job_description)
        
        return {
            "overall_score": round(overall, 1),
            "ats_score": round(ats_score, 1),
            "breakdown": {
                "skills_match": round(skill_score, 1),
                "keyword_relevance": round(keyword_score, 1),
                "experience_match": round(exp_score, 1),
                "education_match": round(edu_score, 1),
            },
            "matched_skills": matched_skills[:20],
            "missing_skills": missing_skills[:15],
            "required_years": req_years,
            "grade": self._get_grade(overall),
        }
    
    def get_optimization_suggestions(self, parsed: Dict, job_description: str) -> List[Dict]:
        suggestions = []
        
        # Skill gaps
        if job_description:
            jd_skills = self._extract_jd_skills(job_description)
            resume_skills = set(s.lower() for s in parsed.get("skills", {}).get("all_technical_flat", []))
            missing = [s for s in jd_skills if s.lower() not in resume_skills]
            
            if missing:
                suggestions.append({
                    "type": "skills",
                    "priority": "high",
                    "title": "Add Missing Skills",
                    "detail": f"Include these skills if you have them: {', '.join(missing[:8])}",
                    "impact": "+15–25 points"
                })
        
        # Contact completeness
        contact = parsed.get("contact", {})
        missing_contact = [f for f in ["email", "phone", "linkedin"] if not contact.get(f)]
        if missing_contact:
            suggestions.append({
                "type": "contact",
                "priority": "high",
                "title": "Complete Contact Information",
                "detail": f"Missing: {', '.join(missing_contact)}",
                "impact": "+5–10 points"
            })
        
        # Summary
        if not parsed.get("summary"):
            suggestions.append({
                "type": "summary",
                "priority": "medium",
                "title": "Add Professional Summary",
                "detail": "A 3–4 sentence summary greatly improves ATS scoring and recruiter attention.",
                "impact": "+8–12 points"
            })
        
        # Experience bullets
        experiences = parsed.get("experience", [])
        if experiences:
            low_bullet_jobs = [e for e in experiences if len(e.get("responsibilities", [])) < 3]
            if low_bullet_jobs:
                suggestions.append({
                    "type": "experience",
                    "priority": "medium",
                    "title": "Strengthen Experience Bullets",
                    "detail": "Add 3–5 quantified achievement bullets per role (numbers, %, $)",
                    "impact": "+10–18 points"
                })
        
        # Projects
        if not parsed.get("projects"):
            suggestions.append({
                "type": "projects",
                "priority": "medium",
                "title": "Add Project Section",
                "detail": "List 2–4 relevant projects with tech stack and outcomes",
                "impact": "+6–10 points"
            })
        
        # Certifications
        if not parsed.get("certifications"):
            suggestions.append({
                "type": "certifications",
                "priority": "low",
                "title": "Add Certifications",
                "detail": "Industry certifications (AWS, Google, Coursera) validate skills",
                "impact": "+4–8 points"
            })
        
        # Keyword density
        if job_description:
            jd_keywords = self._extract_keywords(job_description)
            resume_text = self._parsed_to_text(parsed).lower()
            low_density_kw = [k for k in jd_keywords[:10] if resume_text.count(k.lower()) < 2]
            if len(low_density_kw) > 3:
                suggestions.append({
                    "type": "keywords",
                    "priority": "high",
                    "title": "Improve Keyword Density",
                    "detail": f"Increase usage of: {', '.join(low_density_kw[:6])}",
                    "impact": "+8–15 points"
                })
        
        return suggestions
    
    def _extract_jd_skills(self, jd: str) -> List[str]:
        from nlp_engine import ALL_SKILLS
        found = []
        jd_lower = jd.lower()
        for skill in ALL_SKILLS:
            if re.search(r'\b' + re.escape(skill) + r'\b', jd_lower):
                found.append(skill)
        return found
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords using TF-IDF-like scoring."""
        # Remove stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'not', 'no', 'nor', 'so',
            'yet', 'both', 'either', 'neither', 'each', 'other', 'such', 'what',
            'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'its', 'our', 'their', 'about', 'above', 'across',
            'after', 'against', 'along', 'among', 'around', 'as', 'before', 'behind',
            'below', 'between', 'during', 'into', 'through', 'under', 'up', 'while',
            'work', 'role', 'job', 'candidate', 'experience', 'skill', 'ability',
            'must', 'required', 'position', 'team', 'strong', 'good', 'well',
            'plus', 'also', 'more', 'than', 'how', 'when', 'where', 'why',
        }
        
        words = re.findall(r'\b[a-zA-Z][a-zA-Z\+\#\.]*\b', text.lower())
        word_freq = Counter(w for w in words if w not in stop_words and len(w) > 2)
        return [w for w, _ in word_freq.most_common(50)]
    
    def _tfidf_similarity(self, text1: str, text2: str) -> float:
        """Simple TF-IDF cosine similarity."""
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 
            'with', 'by', 'is', 'are', 'was', 'were', 'be', 'have', 'has', 'had',
            'do', 'does', 'will', 'would', 'could', 'should', 'may', 'can', 'not'
        }
        
        def tokenize(text):
            tokens = re.findall(r'\b[a-zA-Z\+\#\.]{2,}\b', text.lower())
            return [t for t in tokens if t not in stop_words]
        
        def tf(tokens):
            freq = Counter(tokens)
            total = max(len(tokens), 1)
            return {t: c / total for t, c in freq.items()}
        
        t1 = tokenize(text1)
        t2 = tokenize(text2)
        
        tf1 = tf(t1)
        tf2 = tf(t2)
        
        vocab = set(tf1.keys()) | set(tf2.keys())
        
        v1 = [tf1.get(w, 0) for w in vocab]
        v2 = [tf2.get(w, 0) for w in vocab]
        
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a ** 2 for a in v1))
        mag2 = math.sqrt(sum(b ** 2 for b in v2))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return min(dot / (mag1 * mag2), 1.0)
    
    def _extract_required_years(self, jd: str) -> Optional[float]:
        pattern = re.search(
            r'(\d+)\+?\s*(?:to\s*\d+)?\s*years?\s*(?:of\s*)?(?:experience|exp)',
            jd, re.IGNORECASE
        )
        if pattern:
            return float(pattern.group(1))
        return None
    
    def _score_education(self, education: List[Dict], jd: str) -> float:
        if not education:
            return 50.0
        
        # Check degree requirements
        degree_scores = {"PhD": 100, "Master's": 90, "MBA": 85, "B.Tech/BE": 80,
                        "B.Sc": 75, "BCA": 70, "BA": 65, "Associate": 60,
                        "Diploma": 55, "Certification": 50, "High School": 40}
        
        max_score = 0
        for edu in education:
            degree_type = edu.get("degree", "")
            score = degree_scores.get(degree_type, 50)
            max_score = max(max_score, score)
        
        return max_score
    
    def _calculate_ats_score(self, parsed: Dict, jd: str) -> float:
        score = 0
        
        # Contact info completeness (20 pts)
        contact = parsed.get("contact", {})
        if contact.get("email"): score += 7
        if contact.get("phone"): score += 5
        if contact.get("linkedin"): score += 4
        if contact.get("name"): score += 4
        
        # Has summary (15 pts)
        if parsed.get("summary"): score += 15
        
        # Skills section (20 pts)
        skill_count = parsed.get("skills", {}).get("total_count", 0)
        score += min(20, skill_count * 2)
        
        # Experience (25 pts)
        exp = parsed.get("experience", [])
        if exp:
            score += min(15, len(exp) * 5)
            # Bullet points
            total_bullets = sum(len(e.get("responsibilities", [])) for e in exp)
            score += min(10, total_bullets)
        
        # Education (10 pts)
        if parsed.get("education"): score += 10
        
        # Keywords from JD (10 pts)
        if jd:
            jd_keywords = self._extract_keywords(jd)
            resume_text = self._parsed_to_text(parsed).lower()
            matched = sum(1 for k in jd_keywords[:20] if k in resume_text)
            score += min(10, matched * 0.5)
        
        return min(100, score)
    
    def _parsed_to_text(self, parsed: Dict) -> str:
        parts = []
        if parsed.get("summary"):
            parts.append(parsed["summary"])
        
        for skill_list in parsed.get("skills", {}).get("technical", {}).values():
            parts.extend(skill_list)
        
        for exp in parsed.get("experience", []):
            parts.append(exp.get("title", ""))
            parts.extend(exp.get("responsibilities", []))
        
        for proj in parsed.get("projects", []):
            parts.append(proj.get("name", ""))
            parts.append(proj.get("description", ""))
        
        return ' '.join(parts)
    
    def _get_grade(self, score: float) -> str:
        if score >= 85: return "Excellent"
        if score >= 70: return "Good"
        if score >= 55: return "Fair"
        if score >= 40: return "Below Average"
        return "Needs Work"
