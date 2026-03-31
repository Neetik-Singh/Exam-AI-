from flask import Flask, request, jsonify, send_file, render_template_string
import pdfplumber
import re
import os
import json
from collections import defaultdict, Counter
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from werkzeug.utils import secure_filename
import tempfile
import uuid

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Register Devanagari font if available
DEVANAGARI_FONT = False
try:
    pdfmetrics.registerFont(TTFont('Noto', '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf'))
    pdfmetrics.registerFont(TTFont('NotoBold', '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf'))
    DEVANAGARI_FONT = True
except:
    pass

FONT = 'NotoBold' if DEVANAGARI_FONT else 'Helvetica-Bold'
FONT_REG = 'Noto' if DEVANAGARI_FONT else 'Helvetica'

UPLOAD_FOLDER = tempfile.mkdtemp()
OUTPUT_FOLDER = tempfile.mkdtemp()

def extract_text_from_pdf(filepath):
    text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception as e:
        print(f"Error extracting {filepath}: {e}")
    return text

def extract_questions(text):
    """Extract questions from text using multiple patterns"""
    questions = []
    
    # Pattern 1: Numbered questions like १. २. or 1. 2.
    patterns = [
        r'(?:^|\n)\s*[१२३४५६७८९१०\d]+[.।]\s*(.+?)(?=\n\s*[१२३४५६७८९१०\d]+[.।]|\n\n|$)',
        r'(?:प्र\.?\s*[१२३४५६७८९\d]+\.?\s*)(.+?)(?=प्र\.?\s*[१२३४५६७८९\d]+|\n\n|$)',
        r'(.+?\?)',  # anything ending with ?
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
        for m in matches:
            clean = m.strip()
            if len(clean) > 15 and len(clean) < 500:
                questions.append(clean)
    
    # Deduplicate
    seen = set()
    unique = []
    for q in questions:
        key = q[:50].strip()
        if key not in seen:
            seen.add(key)
            unique.append(q)
    
    return unique

def normalize_question(q):
    """Normalize question for comparison"""
    # Remove punctuation, extra spaces, normalize
    q = q.lower().strip()
    q = re.sub(r'[।?!,;:\-\(\)\'\"]+', ' ', q)
    q = re.sub(r'\s+', ' ', q)
    return q

def simple_similarity(q1, q2):
    """Word overlap similarity"""
    words1 = set(normalize_question(q1).split())
    words2 = set(normalize_question(q2).split())
    if not words1 or not words2:
        return 0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)

def cluster_questions(all_questions_by_set):
    """
    Cluster similar questions across sets and count frequency
    Returns list of (representative_question, frequency, all_variants, sets_appeared)
    """
    # Flatten with set tracking
    flat = []
    for set_idx, questions in enumerate(all_questions_by_set):
        for q in questions:
            flat.append((q, set_idx))
    
    if not flat:
        return []
    
    # Greedy clustering by similarity
    clusters = []  # list of {'questions': [...], 'sets': set()}
    used = [False] * len(flat)
    
    for i, (qi, si) in enumerate(flat):
        if used[i]:
            continue
        cluster = {'questions': [(qi, si)], 'sets': {si}}
        used[i] = True
        
        for j, (qj, sj) in enumerate(flat):
            if used[j] or i == j:
                continue
            sim = simple_similarity(qi, qj)
            if sim > 0.35:  # threshold
                cluster['questions'].append((qj, sj))
                cluster['sets'].add(sj)
                used[j] = True
        
        clusters.append(cluster)
    
    # Build result
    result = []
    for cluster in clusters:
        questions = cluster['questions']
        sets_appeared = cluster['sets']
        frequency = len(sets_appeared)
        
        # Pick longest as representative (usually most complete)
        rep = max(questions, key=lambda x: len(x[0]))[0]
        all_variants = list(set(q for q, s in questions))
        
        result.append({
            'question': rep,
            'frequency': frequency,
            'total_appearances': len(questions),
            'sets_appeared': sorted(list(sets_appeared)),
            'variants': all_variants[:3]  # top 3 variants
        })
    
    # Sort by frequency descending
    result.sort(key=lambda x: (-x['frequency'], -x['total_appearances']))
    return result

def assign_importance(frequency, total_sets):
    """Assign VVI / Important / Normal based on frequency"""
    ratio = frequency / max(total_sets, 1)
    if ratio >= 0.6 or frequency >= 6:
        return 'VVI', colors.HexColor('#b71c1c')
    elif ratio >= 0.3 or frequency >= 3:
        return 'Important', colors.HexColor('#e65100')
    else:
        return 'Normal', colors.HexColor('#212121')

def generate_prediction_pdf(clusters, total_sets, output_path, subject=""):
    doc = SimpleDocTemplate(output_path, pagesize=A4,
        rightMargin=1.8*cm, leftMargin=1.8*cm,
        topMargin=1.8*cm, bottomMargin=1.8*cm)

    title_s = ParagraphStyle('T', fontName=FONT, fontSize=16,
        textColor=colors.HexColor('#1a237e'), spaceAfter=4, alignment=1)
    sub_s = ParagraphStyle('S', fontName=FONT_REG, fontSize=10,
        textColor=colors.grey, spaceAfter=10, alignment=1)
    sec_s = ParagraphStyle('SEC', fontName=FONT, fontSize=11,
        textColor=colors.white, spaceAfter=5, spaceBefore=10,
        backColor=colors.HexColor('#1a237e'), leading=18)
    vvi_s = ParagraphStyle('VVI', fontName=FONT, fontSize=11,
        textColor=colors.HexColor('#b71c1c'), spaceAfter=3, spaceBefore=8)
    imp_s = ParagraphStyle('IMP', fontName=FONT, fontSize=11,
        textColor=colors.HexColor('#e65100'), spaceAfter=3, spaceBefore=8)
    norm_s = ParagraphStyle('NORM', fontName=FONT, fontSize=10.5,
        textColor=colors.HexColor('#212121'), spaceAfter=3, spaceBefore=6)
    detail_s = ParagraphStyle('D', fontName=FONT_REG, fontSize=9.5,
        textColor=colors.HexColor('#555555'), spaceAfter=2, leftIndent=12)
    legend_s = ParagraphStyle('L', fontName=FONT_REG, fontSize=9,
        textColor=colors.HexColor('#333333'), spaceAfter=3)

    story = []

    # Title
    title = f"Exam Question Predictor"
    if subject:
        title += f" — {subject}"
    story.append(Paragraph(title, title_s))
    story.append(Paragraph(
        f"Analyzed {total_sets} model sets | {len(clusters)} unique question clusters detected",
        sub_s))
    story.append(HRFlowable(width='100%', thickness=2, color=colors.HexColor('#1a237e')))
    story.append(Spacer(1, 6))

    # Legend
    story.append(Paragraph("🔴 VVI = appeared in 60%+ sets  |  🟠 Important = 30%+ sets  |  ⚫ Normal = less frequent", legend_s))
    story.append(Spacer(1, 8))

    # VVI Questions
    vvi = [c for c in clusters if assign_importance(c['frequency'], total_sets)[0] == 'VVI']
    imp = [c for c in clusters if assign_importance(c['frequency'], total_sets)[0] == 'Important']
    norm = [c for c in clusters if assign_importance(c['frequency'], total_sets)[0] == 'Normal']

    if vvi:
        story.append(Paragraph(f"  🔴 VVI QUESTIONS — Must Prepare ({len(vvi)} questions)", sec_s))
        story.append(Spacer(1, 4))
        for i, c in enumerate(vvi, 1):
            items = []
            items.append(Paragraph(
                f"🔴 Q{i}. {c['question'][:200]}",
                vvi_s))
            items.append(Paragraph(
                f"Appeared in {c['frequency']}/{total_sets} sets (Sets: {', '.join(str(s+1) for s in c['sets_appeared'])})",
                detail_s))
            story.append(KeepTogether(items))

    story.append(Spacer(1, 6))

    if imp:
        story.append(Paragraph(f"  🟠 IMPORTANT QUESTIONS ({len(imp)} questions)", sec_s))
        story.append(Spacer(1, 4))
        for i, c in enumerate(imp, 1):
            items = []
            items.append(Paragraph(
                f"🟠 Q{i}. {c['question'][:200]}",
                imp_s))
            items.append(Paragraph(
                f"Appeared in {c['frequency']}/{total_sets} sets (Sets: {', '.join(str(s+1) for s in c['sets_appeared'])})",
                detail_s))
            story.append(KeepTogether(items))

    story.append(Spacer(1, 6))

    if norm:
        story.append(Paragraph(f"  ⚫ OTHER QUESTIONS ({len(norm)} questions)", sec_s))
        story.append(Spacer(1, 4))
        for i, c in enumerate(norm[:30], 1):  # limit to top 30
            items = []
            items.append(Paragraph(
                f"Q{i}. {c['question'][:200]}",
                norm_s))
            items.append(Paragraph(
                f"Appeared in {c['frequency']}/{total_sets} sets",
                detail_s))
            story.append(KeepTogether(items))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#1a237e')))
    story.append(Paragraph("Generated by ExamAI — Upload more sets for better predictions!", sub_s))

    doc.build(story)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ExamAI — Question Predictor</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface2: #1a1a26;
    --border: #2a2a3d;
    --accent: #6c63ff;
    --accent2: #ff6b6b;
    --accent3: #ffd93d;
    --text: #e8e8f0;
    --muted: #6b6b8a;
    --vvi: #ff6b6b;
    --imp: #ffd93d;
    --norm: #6c63ff;
  }

  body {
    font-family: 'Space Grotesk', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Animated background grid */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: 
      linear-gradient(rgba(108,99,255,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(108,99,255,0.03) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: 0;
  }

  .container {
    max-width: 860px;
    margin: 0 auto;
    padding: 2rem 1.5rem;
    position: relative;
    z-index: 1;
  }

  /* Header */
  header {
    text-align: center;
    padding: 3rem 0 2rem;
  }

  .logo {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -2px;
    margin-bottom: 0.5rem;
  }

  .tagline {
    color: var(--muted);
    font-size: 1rem;
    font-weight: 400;
    letter-spacing: 0.05em;
  }

  /* Upload Zone */
  .upload-zone {
    background: var(--surface);
    border: 2px dashed var(--border);
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    margin: 2rem 0;
    position: relative;
    overflow: hidden;
  }

  .upload-zone::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(108,99,255,0.05), rgba(255,107,107,0.05));
    opacity: 0;
    transition: opacity 0.3s;
  }

  .upload-zone:hover, .upload-zone.drag-over {
    border-color: var(--accent);
    transform: translateY(-2px);
    box-shadow: 0 20px 60px rgba(108,99,255,0.15);
  }

  .upload-zone:hover::before, .upload-zone.drag-over::before {
    opacity: 1;
  }

  .upload-icon {
    font-size: 3.5rem;
    margin-bottom: 1rem;
    display: block;
  }

  .upload-zone h2 {
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: var(--text);
  }

  .upload-zone p {
    color: var(--muted);
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
  }

  #fileInput { display: none; }

  .btn {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1.75rem;
    border-radius: 12px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    cursor: pointer;
    border: none;
    transition: all 0.2s ease;
  }

  .btn-primary {
    background: linear-gradient(135deg, var(--accent), #8b83ff);
    color: white;
    box-shadow: 0 4px 20px rgba(108,99,255,0.3);
  }

  .btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(108,99,255,0.4);
  }

  .btn-danger {
    background: linear-gradient(135deg, var(--accent2), #ff8e8e);
    color: white;
    box-shadow: 0 4px 20px rgba(255,107,107,0.3);
  }

  .btn-danger:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(255,107,107,0.4);
  }

  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none !important;
  }

  /* File list */
  .file-list {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    margin: 1rem 0;
  }

  .file-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    animation: slideIn 0.3s ease;
  }

  @keyframes slideIn {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .file-icon { font-size: 1.2rem; }
  .file-name { flex: 1; font-size: 0.9rem; color: var(--text); }
  .file-size { font-size: 0.8rem; color: var(--muted); }
  .file-remove { 
    background: none; border: none; color: var(--muted); 
    cursor: pointer; font-size: 1rem; padding: 0.2rem;
    transition: color 0.2s;
  }
  .file-remove:hover { color: var(--accent2); }

  /* Subject input */
  .input-group {
    margin: 1.5rem 0;
  }

  .input-group label {
    display: block;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
  }

  .input-group input {
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: var(--text);
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    transition: border-color 0.2s;
    outline: none;
  }

  .input-group input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(108,99,255,0.1);
  }

  /* Progress */
  .progress-bar {
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
    margin: 1rem 0;
    display: none;
  }

  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    border-radius: 2px;
    width: 0%;
    transition: width 0.4s ease;
    animation: shimmer 1.5s infinite;
  }

  @keyframes shimmer {
    0% { opacity: 1; }
    50% { opacity: 0.7; }
    100% { opacity: 1; }
  }

  /* Status */
  .status {
    display: none;
    padding: 1rem 1.25rem;
    border-radius: 12px;
    font-size: 0.95rem;
    margin: 1rem 0;
    border: 1px solid transparent;
  }

  .status.processing {
    display: block;
    background: rgba(108,99,255,0.1);
    border-color: rgba(108,99,255,0.3);
    color: var(--accent);
  }

  .status.success {
    display: block;
    background: rgba(100,200,100,0.1);
    border-color: rgba(100,200,100,0.3);
    color: #6dc96d;
  }

  .status.error {
    display: block;
    background: rgba(255,107,107,0.1);
    border-color: rgba(255,107,107,0.3);
    color: var(--accent2);
  }

  /* Results */
  .results {
    display: none;
    margin-top: 2rem;
    animation: fadeIn 0.5s ease;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .results-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
    gap: 1rem;
  }

  .results-header h2 {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
  }

  /* Stats */
  .stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
  }

  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem;
    text-align: center;
  }

  .stat-number {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .stat-label {
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  /* Question cards */
  .section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    margin: 1.5rem 0 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .q-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
  }

  .q-card::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 3px;
    border-radius: 2px 0 0 2px;
  }

  .q-card.vvi { border-color: rgba(255,107,107,0.3); }
  .q-card.vvi::before { background: var(--vvi); }
  .q-card.imp { border-color: rgba(255,217,61,0.3); }
  .q-card.imp::before { background: var(--imp); }
  .q-card.norm::before { background: var(--norm); }

  .q-card:hover {
    transform: translateX(4px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
  }

  .q-text {
    font-size: 0.95rem;
    line-height: 1.5;
    margin-bottom: 0.4rem;
  }

  .q-meta {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
  }

  .badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
  }

  .badge-vvi { background: rgba(255,107,107,0.15); color: var(--vvi); }
  .badge-imp { background: rgba(255,217,61,0.15); color: var(--imp); }
  .badge-norm { background: rgba(108,99,255,0.15); color: var(--norm); }

  .freq-text {
    font-size: 0.8rem;
    color: var(--muted);
  }

  .sets-text {
    font-size: 0.75rem;
    color: var(--muted);
    font-family: monospace;
  }

  /* Actions */
  .actions {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: 2rem;
    padding-top: 2rem;
    border-top: 1px solid var(--border);
  }

  /* Footer */
  footer {
    text-align: center;
    padding: 3rem 0 2rem;
    color: var(--muted);
    font-size: 0.85rem;
  }

  @media (max-width: 600px) {
    .stats { grid-template-columns: repeat(2, 1fr); }
    .logo { font-size: 2rem; }
  }
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="logo">ExamAI</div>
    <p class="tagline">Upload past model sets → Get predicted questions ranked by importance</p>
  </header>

  <div class="upload-zone" id="uploadZone">
    <span class="upload-icon">📚</span>
    <h2>Drop your model set PDFs here</h2>
    <p>Upload multiple sets for better predictions. More sets = more accurate ranking.</p>
    <label class="btn btn-primary" for="fileInput">
      📂 Choose PDFs
    </label>
    <input type="file" id="fileInput" accept=".pdf" multiple>
  </div>

  <div class="file-list" id="fileList"></div>

  <div class="input-group">
    <label>Subject Name (optional)</label>
    <input type="text" id="subjectInput" placeholder="e.g. Social Studies, Mathematics, Science...">
  </div>

  <div style="display:flex; gap:1rem; flex-wrap:wrap;">
    <button class="btn btn-danger" id="analyzeBtn" disabled onclick="analyzeFiles()">
      🔍 Analyze & Predict
    </button>
  </div>

  <div class="progress-bar" id="progressBar">
    <div class="progress-fill" id="progressFill"></div>
  </div>

  <div class="status" id="status"></div>

  <div class="results" id="results">
    <div class="results-header">
      <h2>📊 Prediction Results</h2>
      <button class="btn btn-primary" onclick="downloadPDF()">⬇️ Download PDF</button>
    </div>

    <div class="stats" id="statsGrid"></div>

    <div id="questionsContainer"></div>

    <div class="actions">
      <button class="btn btn-primary" onclick="downloadPDF()">⬇️ Download Full PDF Report</button>
      <button class="btn" style="background:var(--surface2);border:1px solid var(--border);color:var(--text);"
        onclick="resetApp()">🔄 Analyze New Files</button>
    </div>
  </div>

  <footer>ExamAI — Built for students 💪</footer>
</div>

<script>
let selectedFiles = [];
let resultData = null;
let pdfUrl = null;

const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const analyzeBtn = document.getElementById('analyzeBtn');

// Drag and drop
uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const files = Array.from(e.dataTransfer.files).filter(f => f.type === 'application/pdf');
  addFiles(files);
});

fileInput.addEventListener('change', () => {
  addFiles(Array.from(fileInput.files));
  fileInput.value = '';
});

function addFiles(files) {
  files.forEach(f => {
    if (!selectedFiles.find(sf => sf.name === f.name)) {
      selectedFiles.push(f);
    }
  });
  renderFileList();
}

function removeFile(name) {
  selectedFiles = selectedFiles.filter(f => f.name !== name);
  renderFileList();
}

function renderFileList() {
  fileList.innerHTML = '';
  selectedFiles.forEach(f => {
    const size = (f.size / 1024).toFixed(0);
    const div = document.createElement('div');
    div.className = 'file-item';
    div.innerHTML = `
      <span class="file-icon">📄</span>
      <span class="file-name">${f.name}</span>
      <span class="file-size">${size} KB</span>
      <button class="file-remove" onclick="removeFile('${f.name}')" title="Remove">✕</button>
    `;
    fileList.appendChild(div);
  });
  analyzeBtn.disabled = selectedFiles.length < 1;
}

function setStatus(msg, type) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = `status ${type}`;
}

function setProgress(pct) {
  const bar = document.getElementById('progressBar');
  const fill = document.getElementById('progressFill');
  bar.style.display = pct > 0 ? 'block' : 'none';
  fill.style.width = pct + '%';
}

async function analyzeFiles() {
  if (!selectedFiles.length) return;

  analyzeBtn.disabled = true;
  document.getElementById('results').style.display = 'none';
  setProgress(10);
  setStatus('⏳ Uploading and analyzing files...', 'processing');

  const formData = new FormData();
  selectedFiles.forEach(f => formData.append('files', f));
  formData.append('subject', document.getElementById('subjectInput').value);

  try {
    setProgress(40);
    const res = await fetch('/analyze', { method: 'POST', body: formData });
    setProgress(80);
    const data = await res.json();

    if (!res.ok) throw new Error(data.error || 'Analysis failed');

    resultData = data;
    setProgress(100);
    setStatus(`✅ Done! Found ${data.clusters.length} unique question patterns across ${data.total_sets} sets.`, 'success');
    setTimeout(() => setProgress(0), 500);
    renderResults(data);
  } catch (err) {
    setStatus(`❌ Error: ${err.message}`, 'error');
    setProgress(0);
    analyzeBtn.disabled = false;
  }
}

function renderResults(data) {
  const { clusters, total_sets } = data;

  // Stats
  const vvi = clusters.filter(c => c.importance === 'VVI');
  const imp = clusters.filter(c => c.importance === 'Important');
  const norm = clusters.filter(c => c.importance === 'Normal');

  document.getElementById('statsGrid').innerHTML = `
    <div class="stat-card">
      <div class="stat-number">${total_sets}</div>
      <div class="stat-label">Sets Analyzed</div>
    </div>
    <div class="stat-card">
      <div class="stat-number">${clusters.length}</div>
      <div class="stat-label">Unique Questions</div>
    </div>
    <div class="stat-card">
      <div class="stat-number">${vvi.length}</div>
      <div class="stat-label">VVI Questions</div>
    </div>
  `;

  const container = document.getElementById('questionsContainer');
  container.innerHTML = '';

  const sections = [
    { label: '🔴 VVI — Must Prepare', items: vvi, cls: 'vvi', badge: 'badge-vvi', badgeText: '🔴 VVI' },
    { label: '🟠 Important', items: imp, cls: 'imp', badge: 'badge-imp', badgeText: '🟠 Important' },
    { label: '⚫ Other Questions', items: norm.slice(0, 30), cls: 'norm', badge: 'badge-norm', badgeText: '⚫ Normal' },
  ];

  sections.forEach(sec => {
    if (!sec.items.length) return;
    const title = document.createElement('div');
    title.className = 'section-title';
    title.textContent = `${sec.label} (${sec.items.length})`;
    container.appendChild(title);

    sec.items.forEach((c, i) => {
      const card = document.createElement('div');
      card.className = `q-card ${sec.cls}`;
      card.innerHTML = `
        <div class="q-text">${i+1}. ${c.question.substring(0, 250)}${c.question.length > 250 ? '...' : ''}</div>
        <div class="q-meta">
          <span class="badge ${sec.badge}">${sec.badgeText}</span>
          <span class="freq-text">Appeared in ${c.frequency}/${total_sets} sets</span>
          <span class="sets-text">Sets: ${c.sets_appeared.map(s => s+1).join(', ')}</span>
        </div>
      `;
      container.appendChild(card);
    });
  });

  document.getElementById('results').style.display = 'block';
  document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}

async function downloadPDF() {
  if (!resultData) return;
  const btn = event.target;
  btn.disabled = true;
  btn.textContent = '⏳ Generating PDF...';

  try {
    const res = await fetch('/generate-pdf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(resultData)
    });
    if (!res.ok) throw new Error('PDF generation failed');
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'exam_predictions.pdf';
    a.click();
    URL.revokeObjectURL(url);
  } catch(err) {
    alert('PDF generation failed: ' + err.message);
  }

  btn.disabled = false;
  btn.textContent = '⬇️ Download PDF';
}

function resetApp() {
  selectedFiles = [];
  renderFileList();
  document.getElementById('results').style.display = 'none';
  setStatus('', '');
  setProgress(0);
  resultData = null;
  analyzeBtn.disabled = true;
}
</script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        files = request.files.getlist('files')
        subject = request.form.get('subject', '')

        if not files:
            return jsonify({'error': 'No files uploaded'}), 400

        all_questions = []
        processed = 0

        for f in files:
            if not f.filename.endswith('.pdf'):
                continue
            tmp_path = os.path.join(UPLOAD_FOLDER, f'{uuid.uuid4()}.pdf')
            f.save(tmp_path)
            text = extract_text_from_pdf(tmp_path)
            questions = extract_questions(text)
            all_questions.append(questions)
            os.remove(tmp_path)
            processed += 1

        if not processed:
            return jsonify({'error': 'No valid PDFs processed'}), 400

        total_sets = processed
        clusters = cluster_questions(all_questions)

        # Add importance to each cluster
        for c in clusters:
            imp, _ = assign_importance(c['frequency'], total_sets)
            c['importance'] = imp

        return jsonify({
            'clusters': clusters,
            'total_sets': total_sets,
            'subject': subject
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    try:
        data = request.get_json()
        clusters = data.get('clusters', [])
        total_sets = data.get('total_sets', 1)
        subject = data.get('subject', '')

        output_path = os.path.join(OUTPUT_FOLDER, f'{uuid.uuid4()}.pdf')
        generate_prediction_pdf(clusters, total_sets, output_path, subject)

        return send_file(output_path, as_attachment=True,
                        download_name='exam_predictions.pdf',
                        mimetype='application/pdf')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("ExamAI running at http://localhost:5000")
    app.run(debug=True, port=5000)
