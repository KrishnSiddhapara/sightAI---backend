# 🖼️ AI Image Object Identifier & Multi-Version Editor

A modern, full-stack AI Vision application featuring a **React + Vite Frontend**, a **FastAPI REST API Backend**, and **Google Gemini 2.5 Flash VLM**.

Understand visual content, verify physical object counts, extract independent person attributes, edit images in natural language with non-destructive version history, and interact via visual Q&A.

---

## ✨ Features & Capabilities

- **⚡ Grounded Object Analysis & Counting**: Zero-hallucination physical entity detection with exact confirmed counts and unconfirmed/occluded counts.
- **👤 Person-Specific Attributes**: Independent analysis per detected person (clothing, clothing color, pose, action, accessories).
- **📍 Scene Understanding**: Grounded scene classification (category, environment, activity, and summary).
- **🎨 AI Image Editor & Multi-Version History**: Edit images using natural language prompts while preserving unrelated content. Creates immutable version history (`Version 0` -> `Version 1` -> `Version 2`).
- **🛡️ Multi-Stage AI Safety Gate**: Pre-screens input images and generated edit outputs for safety before displaying or saving them.
- **💬 Visual Q&A ("Ask AI")**: Ask questions grounded strictly in visual evidence with chat message history.
- **📄 Multi-Format Export**: Download any generated version as high-quality `JPG` or `PDF`.
- **📱 Responsive SaaS UI**: Built with React, Vite, Lucide React icons, and a custom dark design system (`#0B0F19`).

---

## 🏗️ Architecture Overview

```text
┌─────────────────────────────────────────────────────────┐
│              React 19 + Vite Frontend                  │
│  (Navbar, Dropzone, Progress, Objects, Editor, Q&A)     │
└───────────────────────────┬─────────────────────────────┘
                            │ REST API (JSON / FormData)
                            ▼
┌─────────────────────────────────────────────────────────┐
│                 FastAPI REST Backend                    │
│                 (backend/server.py)                     │
└───────────────────────────┬─────────────────────────────┘
                            │ Core Python Services
                            ▼
┌─────────────────────────────────────────────────────────┐
│     safety.py • vision.py • image_editor.py             │
│     user_query.py • version_manager.py                  │
└───────────────────────────┬─────────────────────────────┘
                            │ Secure API Key
                            ▼
┌─────────────────────────────────────────────────────────┐
│               Google Gemini API (VLM)                   │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Project Structure

```text
image_identifier/
├── backend/
│   ├── server.py               # FastAPI REST API endpoints
│   └── __init__.py
├── frontend/
│   ├── src/
│   │   ├── components/         # Modular React UI components
│   │   │   ├── Navbar.jsx
│   │   │   ├── HeroSection.jsx
│   │   │   ├── ImageUploader.jsx
│   │   │   ├── AnalysisProgress.jsx
│   │   │   ├── SafetyStatus.jsx
│   │   │   ├── StatsCards.jsx
│   │   │   ├── ObjectList.jsx
│   │   │   ├── ObjectInstanceCard.jsx
│   │   │   ├── SceneCard.jsx
│   │   │   ├── SummaryCard.jsx
│   │   │   ├── ImageEditor.jsx
│   │   │   ├── VersionHistory.jsx
│   │   │   ├── ImageComparison.jsx
│   │   │   ├── AskAI.jsx
│   │   │   └── EmptyState.jsx
│   │   ├── services/
│   │   │   └── api.js          # Axios API client
│   │   ├── index.css           # Dark theme design system & tokens
│   │   ├── App.jsx             # Main Application shell
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── services/                   # Core Python AI business logic
│   ├── safety.py               # Gemini safety classifier
│   ├── vision.py               # Grounded vision & counting service
│   ├── image_editor.py         # Generative AI image editing
│   ├── user_query.py           # Grounded visual Q&A
│   ├── version_manager.py      # Immutable version manager & byte exporter
│   └── schemas.py              # Pydantic data schemas
├── utils/
│   └── image_validation.py     # Image format & size validation
├── tests/
│   ├── test_api.py             # FastAPI unit tests
│   ├── test_accuracy.py
│   ├── test_e2e_editor.py
│   ├── test_image_editor.py
│   ├── test_qna.py
│   ├── test_safety.py
│   └── test_version_manager.py
├── requirements.txt            # Python dependencies
└── .env                        # Environment variables
```

---

## 🔑 Environment Setup

1. Clone repository:
   ```bash
   git clone https://github.com/KrishnSiddhapara/ai-image-object-identifier.git
   cd ai-image-object-identifier
   ```

2. Create a `.env` file in the project root:
   ```env
   VLM_API_KEY=your_actual_gemini_api_key_here
   ```

---

## 🏃 Running the Application

### 1. Start the FastAPI Backend Server
```bash
# Install Python packages
pip install -r requirements.txt

# Run FastAPI server on port 8000
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
```
Backend API will be available at: `http://localhost:8000`

### 2. Start the React Frontend Dev Server
```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Start Vite dev server
npm run dev
```
Open `http://localhost:5173` in your browser!

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check & API key configuration status |
| `POST` | `/api/safety-check` | Performs pre-screening safety check on uploaded image |
| `POST` | `/api/analyze` | Performs grounded visual analysis, object counting & scene understanding |
| `POST` | `/api/edit` | Generates AI image edit & screens generated output for safety |
| `POST` | `/api/ask` | Answers grounded questions about the uploaded image |
| `POST` | `/api/export` | Exports image bytes as JPG, PNG, or PDF downloads |

---

## 🧪 Running Tests

Run the complete Python test suite:
```bash
python -m pytest
```