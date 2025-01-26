import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
from PyPDF2 import PdfReader
import datetime

# Load models
@st.cache_resource
def load_models():
    # Load T5 for question generation
    qg_model_name = "t5-small"
    qg_model = T5ForConditionalGeneration.from_pretrained(qg_model_name)
    qg_tokenizer = T5Tokenizer.from_pretrained(qg_model_name)

    # Load a QA pipeline for answer evaluation
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

    return qg_model, qg_tokenizer, qa_pipeline

# Function to generate questions
def generate_questions(context, num_questions=3, max_length=50):
    input_text = "generate questions: " + context
    input_ids = qg_tokenizer.encode(input_text, return_tensors="pt")
    outputs = qg_model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_questions,
        num_beams=5,
        early_stopping=True,
    )
    questions = [qg_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return questions

# Function to evaluate answers
def evaluate_answer(context, question, user_answer):
    result = qa_pipeline(question=question, context=context)
    correct_answer = result["answer"]
    is_correct = user_answer.strip().lower() == correct_answer.strip().lower()
    return is_correct, correct_answer, result["start"], result["end"]

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Streamlit App
def main():
    st.set_page_config(page_title="Question Generator & Evaluator", page_icon="❓", layout="wide")
    st.title("📚 Question Generation and Answer Evaluation")
    st.write("Generate questions from a context and evaluate your answers!")

    # Load models
    qg_model, qg_tokenizer, qa_pipeline = load_models()

    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload a text or PDF file", type=["txt", "pdf"])
        num_questions = st.slider("Number of questions to generate", 1, 10, 3)
        st.markdown("---")
        st.write("**Instructions:**")
        st.write("1. Upload a file or enter context manually.")
        st.write("2. Generate questions.")
        st.write("3. Provide answers and evaluate them.")

    # Input context
    context = ""
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            context = extract_text_from_pdf(uploaded_file)
        else:
            context = uploaded_file.read().decode("utf-8")
    else:
        context = st.text_area("Or enter the context manually:", height=150)

    if context:
        st.subheader("📝 Context")
        st.write(context)

        # Generate questions
        if st.button("Generate Questions"):
            st.subheader("❓ Generated Questions")
            questions = generate_questions(context, num_questions)
            for i, question in enumerate(questions):
                st.write(f"{i+1}. {question}")

            # Store questions in session state
            st.session_state.questions = questions

        # Answer evaluation
        if "questions" in st.session_state:
            st.subheader("✅ Answer Evaluation")
            for i, question in enumerate(st.session_state.questions):
                st.write(f"**Question {i+1}:** {question}")
                user_answer = st.text_input(f"Your answer for Question {i+1}:", key=f"answer_{i}")

                if user_answer:
                    is_correct, correct_answer, start, end = evaluate_answer(context, question, user_answer)
                    if is_correct:
                        st.success("✅ Correct!")
                    else:
                        st.error(f"❌ Incorrect. The correct answer is: **{correct_answer}**")

                    # Highlight correct answer in context
                    st.write("**Correct Answer in Context:**")
                    highlighted_context = (
                        context[:start] + "**" + context[start:end] + "**" + context[end:]
                    )
                    st.markdown(highlighted_context)

        # Session history
        if "history" not in st.session_state:
            st.session_state.history = []

        if st.button("Save Session"):
            session_data = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "context": context,
                "questions": st.session_state.questions if "questions" in st.session_state else [],
                "user_answers": [st.session_state.get(f"answer_{i}", "") for i in range(num_questions)],
            }
            st.session_state.history.append(session_data)
            st.success("Session saved!")

        if st.session_state.history:
            st.subheader("📜 Session History")
            for session in st.session_state.history:
                st.write(f"**Timestamp:** {session['timestamp']}")
                st.write(f"**Context:** {session['context'][:100]}...")  # Show a preview
                st.write("---")

# Run the app
if __name__ == "__main__":
    main()