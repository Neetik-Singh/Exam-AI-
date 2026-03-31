# ExamAI — Question Predictor

Upload past model set PDFs → Get predicted questions ranked by importance (VVI / Important / Normal)

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000 in your browser.

## How it works

1. Upload multiple model set PDFs
2. App extracts all questions from each PDF
3. Similar questions are clustered together
4. Questions ranked by how many sets they appeared in
5. Download a prediction PDF with VVI / Important / Normal labels

## Features

- Drag & drop PDF upload
- Multi-file support
- Frequency-based importance scoring
- Downloadable PDF report with Devanagari support
- Works with Nepali (Devanagari) text

## Importance Levels

- 🔴 VVI — appeared in 60%+ of sets
- 🟠 Important — appeared in 30%+ of sets  
- ⚫ Normal — appeared less frequently
