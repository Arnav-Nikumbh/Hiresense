import re
from typing import Dict, List, Any, Optional
from datetime import datetime


# Comprehensive skill taxonomy
SKILLS_DB = {
    "languages": [
        "python", "javascript", "typescript", "java", "c++", "c#", "c", "go", "rust",
        "kotlin", "swift", "ruby", "php", "scala", "r", "matlab", "perl", "dart",
        "elixir", "haskell", "lua", "bash", "shell", "powershell", "sql", "html", "css",
        "sass", "less", "solidity", "assembly"
    ],
    "frameworks": [
        "react", "vue", "angular", "nextjs", "nuxtjs", "svelte", "django", "flask",
        "fastapi", "spring", "springboot", "express", "nodejs", "rails", "laravel",
        "asp.net", "dotnet", ".net", "tensorflow", "pytorch", "keras", "scikit-learn",
        "sklearn", "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
        "fastapi", "starlette", "celery", "redis", "rabbitmq", "kafka", "graphql",
        "rest", "restful", "grpc", "websocket", "oauth", "jwt", "bootstrap", "tailwind",
        "material-ui", "chakra", "redux", "mobx", "vuex", "pinia", "rxjs"
    ],
    "databases": [
        "mysql", "postgresql", "postgres", "mongodb", "sqlite", "oracle", "mssql",
        "redis", "cassandra", "dynamodb", "firebase", "supabase", "elasticsearch",
        "neo4j", "influxdb", "clickhouse", "snowflake", "bigquery", "redshift",
        "cockroachdb", "mariadb", "couchdb", "realm"
    ],
    "cloud_devops": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
        "terraform", "ansible", "jenkins", "github actions", "gitlab ci", "circleci",
        "travis ci", "heroku", "vercel", "netlify", "nginx", "apache", "linux",
        "ubuntu", "debian", "centos", "helm", "istio", "prometheus", "grafana",
        "datadog", "splunk", "elk stack", "logstash", "kibana", "puppet", "chef",
        "vagrant", "packer", "cloudformation", "pulumi"
    ],
    "ai_ml": [
        "machine learning", "deep learning", "nlp", "natural language processing",
        "computer vision", "reinforcement learning", "neural networks", "cnn", "rnn",
        "lstm", "transformer", "bert", "gpt", "llm", "rag", "fine-tuning",
        "feature engineering", "data preprocessing", "model deployment", "mlops",
        "a/b testing", "regression", "classification", "clustering", "time series",
        "recommendation systems", "opencv", "spacy", "nltk", "huggingface",
        "langchain", "openai", "anthropic", "stable diffusion", "generative ai",
        "vector database", "embeddings", "tesseract", "ocr"
    ],
    "tools": [
        "git", "github", "gitlab", "bitbucket", "jira", "confluence", "notion",
        "figma", "sketch", "adobe xd", "postman", "swagger", "insomnia", "vscode",
        "intellij", "pycharm", "vim", "emacs", "webpack", "vite", "babel", "eslint",
        "prettier", "junit", "pytest", "jest", "mocha", "cypress", "selenium",
        "playwright", "jupyter", "colab", "tableau", "power bi", "excel"
    ]
}

# Flat skill list for quick matching
ALL_SKILLS = []
for category, skills in SKILLS_DB.items():
    ALL_SKILLS.extend(skills)
ALL_SKILLS = sorted(set(ALL_SKILLS), key=len, reverse=True)  # Longer first for better matching

# Education keywords
DEGREE_PATTERNS = [
    (r'\b(ph\.?d|doctor of philosophy)\b', 'PhD'),
    (r'\b(m\.?s\.?|master of science|msc|m\.tech|master\'?s?)\b', "Master's"),
    (r'\b(m\.?b\.?a|master of business)\b', 'MBA'),
    (r'\b(b\.?e\.?|b\.?tech|bachelor of (engineering|technology))\b', 'B.Tech/BE'),
    (r'\b(b\.?s\.?|b\.?sc|bachelor of science)\b', 'B.Sc'),
    (r'\b(b\.?c\.?a|bachelor of computer applications)\b', 'BCA'),
    (r'\b(b\.?a\.?|bachelor of arts)\b', 'BA'),
    (r'\b(associate\'?s?|a\.?s\.?)\b', 'Associate'),
    (r'\b(diploma)\b', 'Diploma'),
    (r'\b(certification|certified|certificate)\b', 'Certification'),
    (r'\b(10\+2|hsc|secondary|high school|ssc|matriculation)\b', 'High School'),
]

# Soft skills
SOFT_SKILLS = [
    "leadership", "communication", "teamwork", "problem solving", "critical thinking",
    "time management", "adaptability", "creativity", "collaboration", "mentoring",
    "project management", "agile", "scrum", "kanban", "presentations", "public speaking",
    "negotiation", "conflict resolution", "analytical", "detail-oriented", "self-motivated",
    "multitasking", "research", "documentation", "client management", "stakeholder management"
]

# Section header patterns
SECTION_PATTERNS = {
    "experience": [
        r"(work\s*experience|professional\s*experience|employment(\s*history)?|"
        r"career\s*history|work\s*history|positions?\s*held|job\s*history|internship)"
    ],
    "education": [
        r"(education(al)?\s*(background|history|qualification)?|"
        r"academic\s*(background|qualification|history)?|qualification)"
    ],
    "skills": [
        r"(technical\s*skills?|skills?\s*(set|summary)?|competenc(ies|e)|"
        r"technologies|proficiencies|expertise|core\s*competencies)"
    ],
    "projects": [
        r"(projects?(\s*&\s*achievements?)?|personal\s*projects?|"
        r"academic\s*projects?|notable\s*projects?)"
    ],
    "summary": [
        r"(professional\s*summary|career\s*(objective|summary)|"
        r"about\s*me|profile|objective|summary|overview)"
    ],
    "certifications": [
        r"(certifications?|licenses?|accreditations?|credentials?|courses?)"
    ],
    "achievements": [
        r"(achievements?|accomplishments?|awards?|honors?|recognition)"
    ],
    "contact": [
        r"(contact(\s*information)?|personal\s*(information|details))"
    ]
}


class ResumeParser:
    def __init__(self):
        self.skill_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(s) for s in ALL_SKILLS) + r')\b',
            re.IGNORECASE
        )
        self.soft_skill_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(s) for s in SOFT_SKILLS) + r')\b',
            re.IGNORECASE
        )

    def parse(self, text: str) -> Dict[str, Any]:
        """Full resume parsing pipeline."""
        cleaned = self._clean_text(text)
        sections = self._extract_sections(cleaned)
        
        return {
            "contact": self._extract_contact(cleaned),
            "summary": self._extract_summary(sections.get("summary", ""), cleaned),
            "skills": self._extract_skills(cleaned),
            "experience": self._extract_experience(sections.get("experience", "")),
            "education": self._extract_education(sections.get("education", ""), cleaned),
            "projects": self._extract_projects(sections.get("projects", "")),
            "certifications": self._extract_certifications(
                sections.get("certifications", ""), cleaned
            ),
            "total_experience_years": self._estimate_experience_years(cleaned),
            "seniority_level": "",  # filled below
        }

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Split resume into named sections."""
        lines = text.split('\n')
        sections = {}
        current_section = "header"
        current_lines = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                current_lines.append(line)
                continue
            
            matched_section = None
            for section_name, patterns in SECTION_PATTERNS.items():
                for pattern in patterns:
                    if re.match(r'^\s*' + pattern + r'\s*:?\s*$', stripped, re.IGNORECASE):
                        matched_section = section_name
                        break
                if matched_section:
                    break
            
            if matched_section:
                sections[current_section] = '\n'.join(current_lines)
                current_section = matched_section
                current_lines = []
            else:
                current_lines.append(line)
        
        sections[current_section] = '\n'.join(current_lines)
        return sections

    def _extract_contact(self, text: str) -> Dict[str, str]:
        contact = {}
        
        # Email
        email_match = re.search(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}', text)
        if email_match:
            contact["email"] = email_match.group()
        
        # Phone (Indian + international)
        phone_match = re.search(
            r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3,5}\)?[-.\s]?)(\d{3,4}[-.\s]?\d{3,4})', text
        )
        if phone_match:
            contact["phone"] = ''.join(filter(None, phone_match.groups())).strip()
        
        # LinkedIn
        linkedin_match = re.search(r'linkedin\.com/in/[\w-]+', text, re.IGNORECASE)
        if linkedin_match:
            contact["linkedin"] = linkedin_match.group()
        
        # GitHub
        github_match = re.search(r'github\.com/[\w-]+', text, re.IGNORECASE)
        if github_match:
            contact["github"] = github_match.group()
        
        # Name heuristic: first non-empty line that looks like a name
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        for line in lines[:5]:
            if (re.match(r'^[A-Z][a-zA-Z]+([\s][A-Z][a-zA-Z]+){0,3}$', line) and
                    len(line) < 60 and not any(c in line for c in '@+/')):
                contact["name"] = line
                break
        
        # Location
        location_match = re.search(
            r'(?:location|address|city)?\s*:?\s*'
            r'([A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z\s]+)', text
        )
        if location_match:
            contact["location"] = location_match.group(1).strip()
        
        return contact

    def _extract_summary(self, section_text: str, full_text: str) -> str:
        if section_text.strip():
            lines = [l.strip() for l in section_text.split('\n') if l.strip()]
            return ' '.join(lines[:8])
        return ""

    def _extract_skills(self, text: str) -> Dict[str, Any]:
        # Technical skills by category
        found_skills = {}
        text_lower = text.lower()
        
        for category, skills in SKILLS_DB.items():
            matched = []
            for skill in skills:
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text_lower):
                    matched.append(skill)
            if matched:
                found_skills[category] = matched
        
        # Soft skills
        soft_matches = list(set(
            m.group().lower() for m in self.soft_skill_pattern.finditer(text)
        ))
        
        # All skills flat
        all_technical = []
        for skills in found_skills.values():
            all_technical.extend(skills)
        
        return {
            "technical": found_skills,
            "soft": soft_matches[:15],
            "all_technical_flat": list(set(all_technical)),
            "total_count": len(set(all_technical))
        }

    def _extract_experience(self, section_text: str) -> List[Dict]:
        if not section_text.strip():
            return []
        
        experiences = []
        
        # Split by common job entry patterns
        # Look for date patterns as entry boundaries
        date_pattern = re.compile(
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}|'
            r'\d{1,2}/\d{4}|\d{4})\s*[-–—to]+\s*'
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}|'
            r'\d{1,2}/\d{4}|\d{4}|present|current|now)',
            re.IGNORECASE
        )
        
        lines = [l.strip() for l in section_text.split('\n') if l.strip()]
        
        current_entry = {}
        current_bullets = []
        
        for i, line in enumerate(lines):
            date_match = date_pattern.search(line)
            
            if date_match:
                # Save previous entry
                if current_entry:
                    current_entry["responsibilities"] = current_bullets
                    experiences.append(current_entry)
                
                current_entry = {
                    "duration": date_match.group(),
                    "start": date_match.group(1),
                    "end": date_match.group(2),
                    "title": "",
                    "company": "",
                    "responsibilities": []
                }
                current_bullets = []
                
                # Title/company usually on same or adjacent lines
                remaining = line[:date_match.start()].strip()
                if remaining:
                    parts = re.split(r'\s*[|@,·•]\s*', remaining)
                    if parts:
                        current_entry["title"] = parts[0].strip()
                    if len(parts) > 1:
                        current_entry["company"] = parts[1].strip()
            
            elif current_entry and (line.startswith(('•', '-', '·', '*', '▪')) or 
                                      line.startswith(tuple('0123456789'))):
                bullet = re.sub(r'^[•\-·*▪\d\.]+\s*', '', line).strip()
                if bullet:
                    current_bullets.append(bullet)
            
            elif current_entry and not current_entry.get("title") and i < 3:
                current_entry["title"] = line
            
            elif current_entry and not current_entry.get("company") and i < 5:
                if re.search(r'(inc\.|ltd\.|llc|pvt\.|technologies|solutions|systems|'
                             r'consulting|services|group|corp)', line, re.IGNORECASE):
                    current_entry["company"] = line
        
        if current_entry:
            current_entry["responsibilities"] = current_bullets
            experiences.append(current_entry)
        
        return experiences[:10]

    def _extract_education(self, section_text: str, full_text: str) -> List[Dict]:
        educations = []
        text_to_search = section_text if section_text.strip() else full_text
        
        lines = [l.strip() for l in text_to_search.split('\n') if l.strip()]
        
        for line in lines:
            for pattern, degree_type in DEGREE_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    edu = {"degree": degree_type, "raw": line}
                    
                    # Year
                    year_match = re.search(r'(19|20)\d{2}', line)
                    if year_match:
                        edu["year"] = year_match.group()
                    
                    # Institution keywords
                    if re.search(r'(university|college|institute|school|iit|nit|bits)',
                                 line, re.IGNORECASE):
                        edu["institution"] = line
                    
                    # GPA/percentage
                    grade_match = re.search(
                        r'(\d+\.?\d*\s*(?:cgpa|gpa|%|percent|grade))', line, re.IGNORECASE
                    )
                    if grade_match:
                        edu["grade"] = grade_match.group(1)
                    
                    educations.append(edu)
                    break
        
        return educations[:6]

    def _extract_projects(self, section_text: str) -> List[Dict]:
        if not section_text.strip():
            return []
        
        projects = []
        lines = [l.strip() for l in section_text.split('\n') if l.strip()]
        
        current_project = {}
        current_desc = []
        
        for line in lines:
            # New project if line looks like a title (short, possibly bold, no bullet)
            is_title = (
                not line.startswith(('•', '-', '*', '▪')) and
                len(line) < 80 and
                len(line.split()) < 10 and
                not re.match(r'^https?://', line)
            )
            
            if is_title and current_desc:
                if current_project:
                    current_project["description"] = ' '.join(current_desc)
                    projects.append(current_project)
                current_project = {"name": line, "technologies": [], "description": ""}
                current_desc = []
            elif is_title and not current_project:
                current_project = {"name": line, "technologies": [], "description": ""}
            else:
                # Extract technologies from description
                tech_match = re.search(
                    r'(?:tech(?:nologies)?|tools?|built with|using|stack)\s*:?\s*([^\n]+)',
                    line, re.IGNORECASE
                )
                if tech_match:
                    techs = [t.strip() for t in re.split(r'[,|/]', tech_match.group(1))]
                    if current_project:
                        current_project["technologies"].extend(techs[:8])
                
                bullet = re.sub(r'^[•\-·*▪]\s*', '', line)
                if bullet:
                    current_desc.append(bullet)
        
        if current_project:
            current_project["description"] = ' '.join(current_desc)
            projects.append(current_project)
        
        return projects[:8]

    def _extract_certifications(self, section_text: str, full_text: str) -> List[str]:
        certs = []
        cert_pattern = re.compile(
            r'(aws\s+\w+|azure\s+\w+|google\s+\w+|certified\s+[\w\s]+|'
            r'oracle\s+\w+|pmp|cissp|cpa|cfa|comptia[\s\+\w]+|'
            r'coursera|udemy|edx|linkedin learning)[\s\w]*(?:certification|certificate|credential)?',
            re.IGNORECASE
        )
        
        text_to_search = section_text if section_text.strip() else full_text
        for match in cert_pattern.finditer(text_to_search):
            cert = match.group().strip()
            if len(cert) > 5 and cert not in certs:
                certs.append(cert)
        
        return certs[:10]

    def _estimate_experience_years(self, text: str) -> float:
        """Estimate total years of experience from dates."""
        date_ranges = re.findall(
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*)?'
            r'((?:19|20)\d{2})\s*[-–—to]+\s*'
            r'(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*)?'
            r'((?:19|20)\d{2}|present|current|now)',
            text, re.IGNORECASE
        )
        
        current_year = datetime.now().year
        total_months = 0
        
        for _, start_year, end_year in date_ranges:
            try:
                start = int(start_year)
                end = current_year if end_year.lower() in ('present', 'current', 'now') else int(end_year)
                if 1990 <= start <= current_year and start <= end:
                    total_months += (end - start) * 12
            except (ValueError, AttributeError):
                continue
        
        years = round(total_months / 12, 1)
        
        # Also check for explicit mentions
        explicit = re.search(r'(\d+)\+?\s*years?\s*(of\s*)?(?:experience|exp)', text, re.IGNORECASE)
        if explicit:
            exp_years = float(explicit.group(1))
            years = max(years, exp_years)
        
        return min(years, 40)  # cap at 40
