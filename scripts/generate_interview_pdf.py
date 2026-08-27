import os
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "ScriptureRAG — Complete 8-Phase Interview & Viva Master Guide")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)
        
        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_text)
        self.drawString(54, 36, "Major Project Viva Guide • All 8 Phases + Full UI Reference")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        self.restoreState()


def build_all_8_phases_pdf(output_path="ScriptureRAG_Technical_Interview_Guide.pdf"):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Clean Professional Color Palette
    c_primary = colors.HexColor("#1E3A8A")   # Deep Blue
    c_secondary = colors.HexColor("#D97706") # Warm Gold
    c_dark = colors.HexColor("#0F172A")      # Dark Slate
    c_body = colors.HexColor("#334155")      # Slate Text
    c_bg_light = colors.HexColor("#F8FAFC")  # Light Card Background
    c_border = colors.HexColor("#E2E8F0")    # Border

    style_title = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=c_primary,
        spaceAfter=4
    )
    style_subtitle = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=14,
        textColor=c_secondary,
        spaceAfter=10
    )
    style_h1 = ParagraphStyle(
        'Header1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12.5,
        leading=16,
        textColor=c_primary,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    style_h2 = ParagraphStyle(
        'Header2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=c_secondary,
        spaceBefore=6,
        spaceAfter=2,
        keepWithNext=True
    )
    style_body = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=c_body,
        spaceAfter=4
    )
    style_bullet = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11.5,
        textColor=c_body,
        leftIndent=10,
        spaceAfter=2.5
    )
    style_table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10.5,
        textColor=c_dark
    )
    style_table_head = ParagraphStyle(
        'TableHead',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=11,
        textColor=colors.white
    )

    story = []

    # Title
    story.append(Paragraph("ScriptureRAG: Complete 8-Phase Interview Guide", style_title))
    story.append(Paragraph("A Clear, Step-by-Step Breakdown of All 8 Phases + Web UI (Simple English)", style_subtitle))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_secondary, spaceBefore=0, spaceAfter=8))

    # SECTION 1: WHAT IS THIS PROJECT?
    story.append(Paragraph("1. Project Summary in Simple Words", style_h1))
    story.append(Paragraph(
        "<b>ScriptureRAG</b> is an authentic AI counselor for the <b>18 Hindu Mahapuranas</b> (Vishnu, Bhagavata, Shiva, Garuda, Padma, Agni, etc.). Unlike normal ChatGPT which invents fake verses, ScriptureRAG uses <b>RAG (Retrieval-Augmented Generation)</b> across 8 specialized phases to find the real book page and cite exact references (e.g., <i>[Bhagavata Purana, Canto 1, Page 85]</i>).",
        style_body
    ))

    # SECTION 2: ALL 8 PHASES TABLE
    story.append(Paragraph("2. Master Table of All 8 Project Phases (+ Frontend UI)", style_h1))
    phases_table_data = [
        [Paragraph("Phase", style_table_head), Paragraph("Phase Name & File", style_table_head), Paragraph("Algorithms & Models Used", style_table_head), Paragraph("What it Does & Why it is Best", style_table_head)],
        [Paragraph("<b>Phase 1</b>", style_table_cell), Paragraph("<b>Environment & Data Setup</b><br/>(<code>requirements.txt</code>, <code>.env</code>)", style_table_cell), Paragraph("Python 3.10+, PyPDF2, Kaggle Dataset", style_table_cell), Paragraph("Sets up 123 Motilal Banarsidass (MLBD) Mahapurana PDFs and 1,404 Kaggle structured stories.", style_table_cell)],
        [Paragraph("<b>Phase 2</b>", style_table_cell), Paragraph("<b>Master Preprocessing</b><br/>(<code>src/preprocessing.py</code>)", style_table_cell), Paragraph("Sliding Window Chunking (800 chars / 150 overlap)", style_table_cell), Paragraph("Cuts books into 10,582 clean scripture chunks with unique IDs (<code>MLBD_{FILE}_C{INDEX}</code>) to stop data loss.", style_table_cell)],
        [Paragraph("<b>Phase 3</b>", style_table_cell), Paragraph("<b>Two-Layer Input Guardrail</b><br/>(<code>src/guardrails.py</code>)", style_table_cell), Paragraph("Fast Regex (&lt;1ms) + Gemini Flash Classifier", style_table_cell), Paragraph("Blocks hacking, coding questions, and stocks in 1ms. Layer 2 allows real spiritual questions like stress or grief.", style_table_cell)],
        [Paragraph("<b>Phase 4</b>", style_table_cell), Paragraph("<b>Pre-Retrieval Expansion</b><br/>(<code>src/query_processor.py</code>)", style_table_cell), Paragraph("Word-Boundary Regex Matcher (<code>\\b{word}\\b</code>)", style_table_cell), Paragraph("Translates everyday user words into Sanskrit scripture names without false matches (e.g. 'sin' in 'business').", style_table_cell)],
        [Paragraph("<b>Phase 5</b>", style_table_cell), Paragraph("<b>Semantic Vector Search</b><br/>(<code>src/vector_db.py</code>)", style_table_cell), Paragraph("<code>BAAI/bge-base-en-v1.5</code> (768-d) + ChromaDB", style_table_cell), Paragraph("Searches 10,582 book paragraphs in 0.01 seconds using local Cosine Similarity. 100% free forever.", style_table_cell)],
        [Paragraph("<b>Phase 6</b>", style_table_cell), Paragraph("<b>Cross-Encoder Re-Ranking</b><br/>(<code>src/reranker.py</code>)", style_table_cell), Paragraph("<code>cross-encoder/ms-marco-MiniLM-L-6-v2</code>", style_table_cell), Paragraph("Evaluates full token-level cross-attention to re-rank Top-6 candidates down to the Top-3 most accurate passages.", style_table_cell)],
        [Paragraph("<b>Phase 7</b>", style_table_cell), Paragraph("<b>Grounded LLM Generation</b><br/>(<code>src/llm_engine.py</code>)", style_table_cell), Paragraph("Google Gemini 3.6 Flash (T = 0.2, 2048 tokens)", style_table_cell), Paragraph("Synthesizes structured wisdom (Counsel, Evidence, Dharmic Principle) with strict [Purana, Page] citations.", style_table_cell)],
        [Paragraph("<b>Phase 8</b>", style_table_cell), Paragraph("<b>Output RAGAS Guardrail</b><br/>(<code>src/validation.py</code>)", style_table_cell), Paragraph("RAGAS Mathematical Entailment Formula", style_table_cell), Paragraph("Audits Faithfulness (F &ge; 0.85), Answer Relevance & Citations. Proves mathematically that the answer is true.", style_table_cell)],
        [Paragraph("<b>UI / App</b>", style_table_cell), Paragraph("<b>ChatGPT-Style Frontend</b><br/>(<code>frontend/app.py</code>)", style_table_cell), Paragraph("Streamlit + HTML5 Web Speech API (Mic)", style_table_cell), Paragraph("Minimalist dark-mode chatbot where users can type or click the microphone to speak questions directly.", style_table_cell)],
    ]
    t_phases = Table(phases_table_data, colWidths=[40, 115, 130, 219])
    t_phases.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [c_bg_light, colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_phases)
    story.append(Spacer(1, 6))

    # SECTION 3: DETAILED EXPLANATION OF ALL 8 PHASES
    story.append(Paragraph("3. Detailed Explanation of All 8 Phases: Why & How Each Works", style_h1))

    # Phase 1 & 2
    story.append(Paragraph("Phase 1 & 2: Canonical Datasets & Master Preprocessing", style_h2))
    story.append(Paragraph("• <b>What we did:</b> Gathered 123 Motilal Banarsidass (MLBD) translation volumes of the 18 Mahapuranas + 1,404 Kaggle story rows.", style_bullet))
    story.append(Paragraph("• <b>Chunking Logic:</b> 800 characters per chunk with 150 characters overlap. This keeps complete Sanskrit verses and English commentary intact.", style_bullet))
    story.append(Paragraph("• <b>Total Saved:</b> 10,582 chunks saved in <code>data/preprocessed_master_chunks.json</code>.", style_bullet))

    # Phase 3
    story.append(Paragraph("Phase 3: Two-Layer Hybrid Input Guardrail (<code>src/guardrails.py</code>)", style_h2))
    story.append(Paragraph("• <b>Layer 1 (Regex <1ms):</b> Strips hidden unicode, decodes Base64, and stops coding/finance prompts without spending API credits.", style_bullet))
    story.append(Paragraph("• <b>Layer 2 (Semantic Gemini):</b> Understands metaphors (e.g. <i>'my life has no database of joy'</i> is allowed for Puranic guidance, but <i>'write SQL database code'</i> is blocked).", style_bullet))

    # Phase 4
    story.append(Paragraph("Phase 4: Pre-Retrieval & Canonical Query Expansion (<code>src/query_processor.py</code>)", style_h2))
    story.append(Paragraph("• <b>What it does:</b> Maps everyday English questions to canonical scripture concepts (e.g. <i>'Krishna'</i> &rarr; <i>'Lord Vishnu'</i>, <i>'anxiety'</i> &rarr; <i>'Moha / Shoka'</i>).", style_bullet))
    story.append(Paragraph("• <b>Word-Boundary Regex:</b> Uses <code>\\b{word}\\b</code> so it never matches letters inside other words (e.g. avoids matching 'sin' in 'business').", style_bullet))

    # Phase 5
    story.append(Paragraph("Phase 5: Dense Vector Retrieval & ChromaDB (<code>src/vector_db.py</code>)", style_h2))
    story.append(Paragraph("• <b>Embedding Model:</b> <code>BAAI/bge-base-en-v1.5</code> (768 dimensions). One of the top open-source embedding models on Hugging Face.", style_bullet))
    story.append(Paragraph("• <b>Database:</b> Local <b>ChromaDB</b> (Cosine distance). Fast search across 10,582 chunks in 0.01 seconds without cloud hosting costs.", style_bullet))

    story.append(PageBreak()) # Clean page break for readability

    # Phase 6
    story.append(Paragraph("Phase 6: Cross-Encoder Precision Re-Ranking (<code>src/reranker.py</code>)", style_h2))
    story.append(Paragraph("• <b>Model:</b> <code>cross-encoder/ms-marco-MiniLM-L-6-v2</code>.", style_bullet))
    story.append(Paragraph("• <b>Why it is best:</b> Vector search finds 6 general matches. The Cross-Encoder reads both the question and paragraph together to score exact word-by-word relevance, picking the <b>Top 3 most accurate passages</b>. This removes noise and prevents the AI from getting confused.", style_bullet))

    # Phase 7
    story.append(Paragraph("Phase 7: Grounded LLM Generation Engine (<code>src/llm_engine.py</code>)", style_h2))
    story.append(Paragraph("• <b>Model:</b> <b>Google Gemini 3.6 Flash</b> with <b>Temperature = 0.2</b> and <b>2,048 max output tokens</b>.", style_bullet))
    story.append(Paragraph("• <b>Structure:</b> Generates in 3 sections: (1) Direct Counsel, (2) Holy Scripture Evidence with citations <code>[Purana, Page X]</code>, (3) Moral Duty / Dharmic principle.", style_bullet))

    # Phase 8
    story.append(Paragraph("Phase 8: Output Guardrail & RAGAS Verification (<code>src/validation.py</code>)", style_h2))
    story.append(Paragraph("• <b>What it does:</b> Evaluates the generated answer against the retrieved passages to ensure <b>Faithfulness &ge; 0.85 (85%)</b>.", style_bullet))
    story.append(Paragraph("• <b>Verification Badge:</b> Appends <code>*Verified Authentic Scripture Counsel | RAGAS Faithfulness: 1.00 (PASS)*</code> to certify authenticity.", style_bullet))
    story.append(Spacer(1, 6))

    # SECTION 4: WHAT WAS USED FOR THE UI (STREAMLIT & VOICE)
    story.append(Paragraph("4. Frontend Web Interface (UI Tools & Technologies)", style_h1))
    story.append(Paragraph(
        "The web user interface was built to provide an ultra-clean **ChatGPT dark-mode experience** with voice interaction:",
        style_body
    ))

    ui_table_data = [
        [Paragraph("UI Component", style_table_head), Paragraph("Technology Used", style_table_head), Paragraph("How it Works in Simple Words", style_table_head)],
        [Paragraph("Web Framework", style_table_cell), Paragraph("<b>Streamlit (Python)</b>", style_table_cell), Paragraph("Pure Python web framework that runs all 8 RAG phases locally with high speed and zero complex JS build steps.", style_table_cell)],
        [Paragraph("ChatGPT Dark Theme", style_table_cell), Paragraph("<b>Custom CSS Styling</b>", style_table_cell), Paragraph("Matches ChatGPT's official dark interface (<code>#212121</code> background, <code>#2f2f2f</code> rounded chat bubbles, clean typography).", style_table_cell)],
        [Paragraph("Voice Input (Mic)", style_table_cell), Paragraph("<b>HTML5 Web Speech API</b> (<code>webkitSpeechRecognition</code>)", style_table_cell), Paragraph("When the user clicks <b>'🎙️ Speak Question'</b>, the browser listens in Chrome/Edge, converts voice to text, and sends it with <b>zero API cost</b>.", style_table_cell)],
        [Paragraph("Conversation State", style_table_cell), Paragraph("<b>Streamlit Session State</b> (<code>st.session_state</code>)", style_table_cell), Paragraph("Stores conversation history in memory and provides a <b>'➕ New Chat'</b> button to start a new inquiry anytime.", style_table_cell)],
    ]
    t_ui = Table(ui_table_data, colWidths=[90, 120, 294])
    t_ui.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [c_bg_light, colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_ui)
    story.append(Spacer(1, 8))

    # SECTION 5: TOP INTERVIEW / VIVA QUESTIONS & SIMPLE ANSWERS
    story.append(Paragraph("5. Top Viva & Interview Questions (Ready-to-Speak Answers)", style_h1))

    simple_qa = [
        ("Q1: Explain the flow of your project in 1 minute.",
         "Answer: 'ScriptureRAG has 8 phases: Phase 1 & 2 preprocessed 123 Puranic volumes into 10,582 chunks. Phase 3 filters invalid queries via a 2-layer guardrail. Phase 4 expands queries with Sanskrit entities. Phase 5 uses BGE embeddings and ChromaDB for fast vector search. Phase 6 re-ranks candidates with a MiniLM Cross-Encoder. Phase 7 synthesizes answers with Gemini 3.6 Flash at T=0.2 with citations. Phase 8 validates Faithfulness (F >= 0.85) using RAGAS. The UI is built with Streamlit and HTML5 Voice input.'"),
        
        ("Q2: Why use RAG instead of Fine-Tuning a Large Language Model?",
         "Answer: 'Fine-tuning modifies model weights, which is expensive and still hallucinates fake verses. RAG keeps the 18 Mahapuranas in a separate database (ChromaDB), allowing the AI to retrieve exact passages and quote real page citations with 100% accuracy.'"),
        
        ("Q3: Which embedding model did you choose and why?",
         "Answer: 'I chose BAAI/bge-base-en-v1.5 (768 dimensions). It is a top-ranked open-source model on the Hugging Face MTEB leaderboard, runs 100% locally on CPU without paid API costs, and searches 10,582 chunks in 10 milliseconds.'"),
        
        ("Q4: Why is Cross-Encoder re-ranking (Phase 6) necessary after Vector search?",
         "Answer: 'Vector Search is fast for finding 6 candidate passages. The Cross-Encoder performs deep word-by-word cross-attention between the query and passage to pick the Top 3 most accurate passages, cutting out noise and preventing hallucinations.'"),
        
        ("Q5: What technologies did you use for the Frontend UI and Voice input?",
         "Answer: 'I used Streamlit with custom CSS to create a minimalist ChatGPT dark-mode interface. For voice input, I integrated the HTML5 Web Speech API directly in the browser so users can click the microphone button and speak their queries for free.'")
    ]

    for q, a in simple_qa:
        q_card = [
            [Paragraph(f"<b>{q}</b>", style_h2)],
            [Paragraph(a, style_body)]
        ]
        t_qcard = Table(q_card, colWidths=[504])
        t_qcard.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), c_bg_light),
            ('BOX', (0, 0), (-1, -1), 0.5, c_border),
            ('TOPPADDING', (0, 0), (-1, -1), 2.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(t_qcard)
        story.append(Spacer(1, 3.5))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated 8-Phase Interview Guide PDF at: {output_path}")

if __name__ == "__main__":
    out_file = "ScriptureRAG_Technical_Interview_Guide.pdf"
    build_all_8_phases_pdf(out_file)

