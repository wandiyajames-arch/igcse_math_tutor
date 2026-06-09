import gradio as gr
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from src.rag.vector_store import get_retriever

# ---------------------------------------------------------
# 1. INITIALIZE THE OFFLINE AI ENGINE
# ---------------------------------------------------------
print("🧠 Booting up local LLM (Ollama: Llama 3.2)...")
llm = ChatOllama(model="llama3.2")
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
    response = rag_chain.invoke({"input": message})
    return response['answer']

def generate_notes(topic):
    prompt = f"Generate concise, structured study notes for the IGCSE Math topic: {topic}. Include key definitions, rules, and an example."
    response = rag_chain.invoke({"input": prompt})
    return response['answer']

def get_past_question(topic):
    prompt = f"Provide one typical IGCSE past exam-style question for the topic: {topic}. Underneath the question, provide a step-by-step mark scheme/solution."
    response = rag_chain.invoke({"input": prompt})
    return response['answer']

def generate_quiz(topic):
    prompt = f"Create a short, 3-question quiz on {topic} for IGCSE Math. Make question 1 easy, question 2 medium, and question 3 hard. Do NOT provide the answers yet."
    response = rag_chain.invoke({"input": prompt})
    return response['answer']

igcse_topics = [
    "Algebra & Equations", "Geometry & Shapes", "Trigonometry", 
    "Probability", "Statistics", "Graphs & Functions", 
    "Numbers & Percentages", "Matrices & Vectors"
]

# ---------------------------------------------------------
# 3. ADVANCED FRONTEND STYLING (CSS & THEME)
# ---------------------------------------------------------
# We define a custom color palette and pull a clean Google Font
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

# Custom CSS for glowing text, rounded corners, and sleek spacing
custom_css = """
body { background-color: #f8fafc; }
.hero-text {
    text-align: center;
    font-family: 'Inter', sans-serif;
    margin-bottom: 20px;
}
.hero-title {
    background: linear-gradient(90deg, #4f46e5, #0ea5e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3em;
    font-weight: 800;
    margin-bottom: 0px;
}
.hero-subtitle {
    color: #64748b;
    font-size: 1.1em;
    font-weight: 400;
}
.gradio-container {
    border-radius: 16px;
}
"""

# ---------------------------------------------------------
# 4. BUILD THE AESTHETIC DASHBOARD (Gradio 6.0 Compliant)
# ---------------------------------------------------------
# REMOVED theme and css from here
with gr.Blocks(title="IGCSE Math Pro") as demo:
    
    # Custom HTML Hero Section
    gr.HTML("""
    <div class="hero-text">
        <h1 class="hero-title">📐 IGCSE Math Pro</h1>
        <p class="hero-subtitle">Your 100% offline, private Cambridge AI engine.</p>
    </div>
    """)
    
    with gr.Tabs():
        
        # TAB 1: Main Chat
        with gr.TabItem("💬 AI Tutor"):
            gr.ChatInterface(
                fn=tutor_chat,
                description="Ask any math question and get step-by-step help from Llama 3.2.",
            )
            
        # TAB 2: Study Materials
        with gr.TabItem("📚 Study Materials"):
            gr.Markdown("### 📖 Generate Custom Notebooks")
            with gr.Row():
                with gr.Column(scale=3):
                    topic_dropdown_notes = gr.Dropdown(choices=igcse_topics, label="Select Syllabus Topic", show_label=False, container=False)
                with gr.Column(scale=1):
                    btn_notes = gr.Button("✨ Generate Notes", variant="primary")
            # REMOVED the show_copy_button parameter
            output_notes = gr.Textbox(label="Your Study Notebook", lines=12)
            
            btn_notes.click(fn=generate_notes, inputs=topic_dropdown_notes, outputs=output_notes)

        # TAB 3: Past Questions
        with gr.TabItem("📝 Past Papers"):
            gr.Markdown("### 📄 Exam Question Generator")
            with gr.Row():
                with gr.Column(scale=3):
                    topic_dropdown_exam = gr.Dropdown(choices=igcse_topics, label="Select Exam Topic", show_label=False, container=False)
                with gr.Column(scale=1):
                    btn_exam = gr.Button("🧠 Get Practice Question", variant="primary")
            # REMOVED the show_copy_button parameter
            output_exam = gr.Textbox(label="Exam Question & Mark Scheme", lines=12)
            
            btn_exam.click(fn=get_past_question, inputs=topic_dropdown_exam, outputs=output_exam)

        # TAB 4: Quick Quiz
        with gr.TabItem("🎯 Quick Quiz"):
            gr.Markdown("### ⏱️ Rapid Fire Testing")
            with gr.Row():
                with gr.Column(scale=3):
                    topic_dropdown_quiz = gr.Dropdown(choices=igcse_topics, label="Select Quiz Topic", show_label=False, container=False)
                with gr.Column(scale=1):
                    btn_quiz = gr.Button("🚀 Start Quiz", variant="primary")
            output_quiz = gr.Textbox(label="Your Quiz Questions", lines=10)
            
            btn_quiz.click(fn=generate_quiz, inputs=topic_dropdown_quiz, outputs=output_quiz)

        # TAB 5: Formula Bank
        with gr.TabItem("🧮 Formula Bank"):
            gr.Markdown(r"""
            ### 📌 Essential IGCSE Formulas
            
            **Quadratic Formula:** $$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$
            
            **Pythagoras Theorem:** $$a^2 + b^2 = c^2$$
            
            **Area of a Circle:** $$A = \pi r^2$$
            
            **Circumference of a Circle:** $$C = 2\pi r$$
            
            **Speed:** $$\text{Speed} = \frac{\text{Distance}}{\text{Time}}$$
            
            **Density:** $$\text{Density} = \frac{\text{Mass}}{\text{Volume}}$$
            
            **Trigonometry (SOH CAH TOA):** $$\sin(\theta) = \frac{\text{Opposite}}{\text{Hypotenuse}}$$
            $$\cos(\theta) = \frac{\text{Adjacent}}{\text{Hypotenuse}}$$
            $$\tan(\theta) = \frac{\text{Opposite}}{\text{Adjacent}}$$
            """)
if __name__ == "__main__":
    print("\n✅ Launching Web Dashboard on Hugging Face...")
    # 0.0.0.0 tells the app to accept connections from the public internet
    demo.launch(server_name="0.0.0.0", server_port=7860)