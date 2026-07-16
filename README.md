# HireSense AI 🧠

An intelligent resume parsing and screening platform built with FastAPI, OCR, and NLP.

## Live link - https://hiresense-2-p6l3.onrender.com
## Features
- 📄 **Resume Parsing** — PDF, image (PNG/JPEG), and plain text support
- 🔍 **OCR Extraction** — Tesseract OCR for scanned PDFs and images
- 🤖 **NLP Analysis** — Extracts skills, experience, education, contact, projects
- 🎯 **Job Matching** — TF-IDF cosine similarity against job descriptions
- 📊 **Scoring Engine** — Overall match, ATS compatibility, skill gap analysis
- 💡 **Optimization Tips** — Actionable resume improvement suggestions
- 🎨 **Beautiful UI** — Dark mode, animated score ring, tabbed results

## Tech Stack
- **Backend**: Python, FastAPI, Uvicorn
- **OCR**: Tesseract, pdf2image, pdfplumber
- **NLP**: Custom TF-IDF, regex-based parsing (50+ skill taxonomy)
- **Frontend**: Vanilla HTML/CSS/JS (no framework needed)

## Setup

```bash
# 1. Install system dependencies (Ubuntu/Debian)
sudo apt-get install tesseract-ocr poppler-utils -y

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run the server
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 4. Open http://localhost:8000
```

## Project Structure
```
hiresense/
├── backend/
│   ├── main.py           # FastAPI app, routes, file handling
│   ├── nlp_engine.py     # Resume parser (sections, contact, skills, experience)
│   └── resume_scorer.py  # TF-IDF scoring, ATS, suggestions
├── frontend/
│   └── index.html        # Complete single-file UI
├── requirements.txt
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/parse-resume` | Upload and analyze resume |
| POST | `/api/optimize-resume` | Get optimization suggestions |
| GET | `/api/health` | Health check |
| GET | `/docs` | Auto-generated Swagger UI |

## API Usage (curl)
```bash
curl -X POST http://localhost:8000/api/parse-resume \
  -F "file=@resume.pdf" \
  -F "job_description=Python developer with FastAPI and Docker"
```
