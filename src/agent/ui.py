import gradio as gr
import re  # --- NEW: Added regex library for smart text splitting ---
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from src.rag.vector_store import get_retriever
# --- NEW: Import our Deterministic Grader ---
from src.utils.grader import verify_math

# ---------------------------------------------------------
# 1. INITIALIZE THE OFFLINE AI ENGINE
# ---------------------------------------------------------
print("🧠 Booting up local LLM (Ollama: Llama 3.2 1B)...")
llm = ChatOllama(model="llama3.2:1b")
retriever = get_retriever()

system_prompt = (
    "You are an expert, patient Cambridge IGCSE Math Tutor. "
    "Use the provided syllabus context to answer the student's question accurately. "
    "If the answer is not contained within the context, do not guess; simply state that you do not know based on the syllabus.\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ---------------------------------------------------------
# 2. DEFINE THE AI ACTION FUNCTIONS
# ---------------------------------------------------------
def tutor_chat(message, history):
    try:
        response = rag_chain.invoke({"input": message})
        return response['answer']
    except Exception as e:
        print(f"❌ Error: {e}")
        return "⚠️ I'm having trouble connecting to my knowledge base. Please try again in a moment."

# --- UPDATED: Notes Generation without Asterisks ---
def generate_notes(topic, year):
    try:
        # 1. PROMPT ENGINEERING: Explicitly tell the AI to stop using Markdown
        if not year:
            prompt = (
                f"Generate concise, structured study notes for the IGCSE Math topic: {topic}. "
                f"Include key definitions, rules, and an example. "
                f"CRITICAL: Do NOT use asterisks (*) or markdown formatting. Use standard text, dashes (-), and numbered lists."
            )
        else:
            prompt = (
                f"Generate concise, structured study notes for the IGCSE Math topic: {topic} "
                f"specifically contextualized around the {year} examination syllabus standards. "
                f"Include key definitions, rules, and typical exam-style examples from {year}. "
                f"CRITICAL: Do NOT use asterisks (*) or markdown formatting. Use standard text, dashes (-), and numbered lists."
            )
        
        response = rag_chain.invoke({"input": prompt})
        
        # 2. PYTHON STRING STRIP: Forcefully remove any rogue asterisks just in case the 1B model disobeys
        clean_text = response['answer'].replace('*', '')
        
        return clean_text
    except Exception as e:
        return f"⚠️ Error generating notes: {str(e)}"

# --- INTERACTIVE PAST PAPER FUNCTIONS ---
def generate_exam_question(topic):
    try:
        if not topic:
            return "Please select a topic to pull a past paper question.", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value="", visible=False)
        
        prompt = (
            f"You are a strict Cambridge Examiner. Provide exactly ONE typical exam-style Cambridge IGCSE Math question "
            f"from past papers for the topic: {topic}. "
            f"CRITICAL: ONLY output the question text. Do NOT provide the answer, solution, or mark scheme. Stop typing immediately after the question ends."
        )
        response = rag_chain.invoke({"input": prompt})
        return response['answer'], gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(value="", visible=False)
    except Exception as e:
        return f"⚠️ Error loading question: {str(e)}", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value="", visible=False)

def validate_exam_answer(topic, question, user_answer):
    try:
        if not user_answer.strip():
            return gr.update(value="⚠️ Please type an answer before submitting.", visible=True)
            
        prompt = f"Context Topic: {topic}\nPast Paper Question: {question}\nStudent's Proposed Solution/Answer: {user_answer}\n\nAct as an official Cambridge Examiner. Grade this response against standard IGCSE mark scheme rules. State clearly if it gets full marks, partial marks, or zero marks, and provide a constructive breakdown of the official mark scheme steps."
        response = rag_chain.invoke({"input": prompt})
        
        return gr.update(value=response['answer'], visible=True)
    except Exception as e:
        return gr.update(value=f"⚠️ Error validating answer: {str(e)}", visible=True)

# --- INTERACTIVE QUICK QUIZ FUNCTIONS (STATE + SYMPY LOGIC) ---
def generate_single_question(topic, difficulty):
    try:
        if not topic or not difficulty:
            return "Please select a topic/difficulty.", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value="", visible=False), 0, ""
        
        prompt = (
            f"Create exactly ONE clear IGCSE Math exam question on the topic '{topic}' with difficulty '{difficulty}'. "
            f"You MUST format your output exactly like this:\n\n"
            f"QUESTION: [Write the question here]\n"
            f"ANSWER: [Write ONLY the final numerical/algebraic answer here, nothing else]"
        )
        response = rag_chain.invoke({"input": prompt})
        raw_text = response['answer']
        
        parts = re.split(r'(?i)answer\s*:', raw_text)
        
        if len(parts) > 1:
            ui_question = re.sub(r'(?i)question\s*:', '', parts[0]).strip()
            hidden_answer = parts[1].strip()
        else:
            ui_question = "⚠️ Error generating question format cleanly. Please click 'Generate Question' again to get a fresh one."
            hidden_answer = ""

        return ui_question, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(value="", visible=False), 0, hidden_answer
    except Exception as e:
        return f"⚠️ Error loading quiz: {str(e)}", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value="", visible=False), 0, ""

def validate_quiz_answer(topic, difficulty, question, user_answer, attempt_count, hidden_true_answer):
    try:
        if not user_answer.strip():
            return gr.update(value="⚠️ Please type an answer before submitting.", visible=True), attempt_count
            
        is_math_correct = verify_math(user_answer, hidden_true_answer)
        
        if is_math_correct:
            success_msg = f"✅ **CORRECT!** Excellent work. You successfully solved it."
            return gr.update(value=success_msg, visible=True), attempt_count

        if attempt_count == 0:
            prompt = (
                f"Context Topic: {topic}\nDifficulty: {difficulty}\n"
                f"Question Given: {question}\nStudent's Answer: {user_answer}\n"
                f"The correct numerical answer is {hidden_true_answer}, but the student got it wrong. "
                f"Start with '❌ INCORRECT - ATTEMPT 1'. Analyze their answer and provide a helpful, conceptual HINT about what their first step should be. Do NOT give them the final answer!"
            )
            response = rag_chain.invoke({"input": prompt})
            return gr.update(value=response['answer'], visible=True), 1
            
        else:
            prompt = (
                f"Context Topic: {topic}\nDifficulty: {difficulty}\n"
                f"Question Given: {question}\nStudent's Answer: {user_answer}\n"
                f"The correct numerical answer is {hidden_true_answer}. The student has failed twice. "
                f"Start with '❌ INCORRECT - FINAL ATTEMPT'. Provide the COMPLETE step-by-step mathematical solution to reach {hidden_true_answer}, explaining it like a Cambridge marking scheme."
            )
            response = rag_chain.invoke({"input": prompt})
            return gr.update(value=response['answer'], visible=True), attempt_count
            
    except Exception as e:
        return gr.update(value=f"⚠️ Error validating answer: {str(e)}", visible=True), attempt_count

igcse_topics = [
    "Algebra & Equations", "Geometry & Shapes", "Trigonometry", 
    "Probability", "Statistics", "Graphs & Functions", 
    "Numbers & Percentages", "Matrices & Vectors"
]
examination_years = ["2024", "2023", "2022", "2021", "2020", "2019"]

# ---------------------------------------------------------
# 3. ADVANCED FRONTEND STYLING (CSS & THEME)
# ---------------------------------------------------------
custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    button_primary_background_fill="linear-gradient(90deg, #4f46e5, #0ea5e9)",
    button_primary_background_fill_hover="linear-gradient(90deg, #4338ca, #0284c7)",
    button_primary_text_color="white",
    block_title_text_weight="600",
    block_border_width="0px",
    block_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)"
)
custom_css = """
body { background-color: #f8fafc; }
.hero-text { text-align: center; font-family: 'Inter', sans-serif; margin-bottom: 20px; }
.hero-title { background: linear-gradient(90deg, #4f46e5, #0ea5e9); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3em; font-weight: 800; margin-bottom: 0px; }
.hero-subtitle { color: #64748b; font-size: 1.1em; font-weight: 400; }
.gradio-container { border-radius: 16px; }
"""

# ---------------------------------------------------------
# 4. BUILD THE AESTHETIC DASHBOARD
# ---------------------------------------------------------
with gr.Blocks(title="IGCSE Math Pro", theme=custom_theme, css=custom_css) as demo:
    gr.HTML("<div class='hero-text'><h1 class='hero-title'>📐 IGCSE Math Pro</h1><p class='hero-subtitle'>An AI tutor for IGCSE Mathematics (0580), running on an open-source model. Not affiliated with Cambridge.</p></div>")
    
    with gr.Tabs():
        with gr.TabItem("💬 AI Tutor"):
            gr.ChatInterface(fn=tutor_chat, description="Ask any math question and get step-by-step help from Llama 3.2 1B.")
            
        with gr.TabItem("📚 Study Materials"):
            gr.Markdown("### 📖 Generate Custom Notebooks")
            with gr.Row():
                with gr.Column(scale=2): topic_dropdown_notes = gr.Dropdown(choices=igcse_topics, label="Select Syllabus Topic", container=True)
                with gr.Column(scale=1): year_dropdown_notes = gr.Dropdown(choices=examination_years, label="Select Syllabus Year", value="2024", container=True)
                with gr.Column(scale=1): btn_notes = gr.Button("✨ Generate Notes", variant="primary")
            output_notes = gr.Textbox(label="Your Study Notebook", lines=12)
            btn_notes.click(fn=generate_notes, inputs=[topic_dropdown_notes, year_dropdown_notes], outputs=output_notes)

        with gr.TabItem("📝 Past Papers"):
            gr.Markdown("### 📄 Interactive Exam Question Practice")
            with gr.Row():
                with gr.Column(scale=3): topic_dropdown_exam = gr.Dropdown(choices=igcse_topics, label="Select Exam Topic", container=True)
                with gr.Column(scale=1): btn_exam = gr.Button("🧠 Pull Past Question", variant="primary")
            exam_question_output = gr.Markdown("Select a topic above and click to pull a genuine exam-style challenge.")
            with gr.Column() as exam_workspace:
                user_exam_answer_input = gr.Textbox(label="Write your working steps and final answer here:", placeholder="e.g., Working: 3x = 12 => x = 4", visible=False)
                btn_submit_exam = gr.Button("Submit Solution for Grading", variant="primary", visible=False)
                exam_feedback_output = gr.Markdown("", visible=False)
            btn_exam.click(fn=generate_exam_question, inputs=topic_dropdown_exam, outputs=[exam_question_output, user_exam_answer_input, btn_submit_exam, exam_workspace, exam_feedback_output])
            btn_submit_exam.click(fn=validate_exam_answer, inputs=[topic_dropdown_exam, exam_question_output, user_exam_answer_input], outputs=exam_feedback_output)

        with gr.TabItem("🎯 Quick Quiz"):
            gr.Markdown("### ⏱️ Dynamic Evaluation Challenge")
            
            quiz_attempt_state = gr.State(0)
            quiz_hidden_answer = gr.State("") 
            
            with gr.Row():
                with gr.Column(scale=2): topic_dropdown_quiz = gr.Dropdown(choices=igcse_topics, label="1. Choose Topic")
                with gr.Column(scale=2): difficulty_dropdown_quiz = gr.Dropdown(choices=["Easy", "Medium", "Hard"], label="2. Choose Difficulty")
                with gr.Column(scale=1): btn_quiz = gr.Button("🚀 Generate Question", variant="primary")
            
            quiz_question_output = gr.Markdown("Select your parameters above and click generate to load your challenge.")
            
            with gr.Column() as quiz_workspace:
                user_answer_input = gr.Textbox(label="Write your calculation or final answer here:", placeholder="e.g., x = 5 or 24cm...", visible=False)
                btn_submit_answer = gr.Button("Submit Answer for Verification", variant="primary", visible=False)
                quiz_feedback_output = gr.Markdown("", visible=False)
            
            btn_quiz.click(
                fn=generate_single_question, 
                inputs=[topic_dropdown_quiz, difficulty_dropdown_quiz], 
                outputs=[quiz_question_output, user_answer_input, btn_submit_answer, quiz_workspace, quiz_feedback_output, quiz_attempt_state, quiz_hidden_answer]
            )
            
            btn_submit_answer.click(
                fn=validate_quiz_answer,
                inputs=[topic_dropdown_quiz, difficulty_dropdown_quiz, quiz_question_output, user_answer_input, quiz_attempt_state, quiz_hidden_answer],
                outputs=[quiz_feedback_output, quiz_attempt_state]
            )

        with gr.TabItem("🧮 Formula Bank"):
            gr.Markdown(r"""
            ### 📌 Essential IGCSE Formulas
            **Quadratic Formula:** $$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$
            **Pythagoras Theorem:** $$a^2 + b^2 = c^2$$
            **Area of a Circle:** $$A = \pi r^2$$
            **Speed:** $$\text{Speed} = \frac{\text{Distance}}{\text{Time}}$$
            **Trigonometry (SOH CAH TOA):** $$\sin(\theta) = \frac{\text{Opposite}}{\text{Hypotenuse}}$$
            """)

if __name__ == "__main__":
    print("\n✅ Launching Web Dashboard on Hugging Face...")
    demo.launch(server_name="0.0.0.0", server_port=7860)