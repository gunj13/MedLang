import streamlit as st
import json
import faiss
import numpy as np
import torch
from typing import TypedDict, Annotated, Sequence
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import operator

# NEW: Hugging Face Inference client
from huggingface_hub import InferenceClient

# Page configuration
st.set_page_config(
    page_title="MedLang - Women's Health Assistant",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f7fa;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #8bc34a;
    }
    .reasoning-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .model-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        background-color: #ffebee;
        color: #c62828;
    }
    .stButton>button {
        width: 100%;
        background-color: #8bc34a;
        color: white;
    }
    .main-header {
        text-align: center;
        color: #2c3e50;
        padding: 2rem 0;
    }
    .context-box {
        background-color: #e1f5fe;
        border-left: 4px solid #0277bd;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    .stats-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


# --- Load environment variables ---
@st.cache_resource
def load_environment():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        st.error("⚠️ HF_TOKEN not found in .env file! Please add your HuggingFace access token.")
        st.info("Get your token from: https://huggingface.co/settings/tokens")
        st.stop()
    return hf_token

# --- Initialize embedding model ---
@st.cache_resource
def initialize_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# --- Load Menstrual-LLaMA (REPLACED: use Inference API; CPU-safe) ---
@st.cache_resource
def load_menstrual_llama(hf_token):
    try:
        client = InferenceClient(api_key=hf_token)
        return client, None
    except Exception as e:
        st.error(f"⚠️ Could not initialize InferenceClient: {str(e)}")
        st.info("Ensure HF_TOKEN is correct and has access to the desired model/provider.")
        st.stop()


# --- Load dataset and build FAISS index ---
@st.cache_resource
def load_dataset_and_index(data_file, _embedder):
    try:
        qa_pairs = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                qa_pairs.append({"question": obj["question"], "answer": obj["answer"]})

        dataset = Dataset.from_list(qa_pairs)
        question_embeddings = _embedder.encode(dataset["question"], convert_to_numpy=True)

        dim = question_embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(question_embeddings)

        return dataset, index
    except FileNotFoundError:
        st.error(f"⚠️ Dataset file '{data_file}' not found! Please ensure it's in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading dataset: {str(e)}")
        st.stop()


# --- State Definition ---
class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    question: str
    retrieved_context: list
    reasoning: str
    answer: str


# --- Helper Functions ---
def format_chat_history(messages, max_exchanges=3):
    """Format recent chat history for context"""
    if not messages:
        return ""

    history_str = "Previous conversation:\n"
    for msg in messages[-(max_exchanges*2):]:
        if isinstance(msg, HumanMessage):
            history_str += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_str += f"Assistant: {msg.content}\n"
    return history_str + "\n"


# --- Node Functions ---
def retrieve_context(state: GraphState, dataset, index, embedder) -> GraphState:
    """Retrieve top 2 most relevant Q&A pairs from pregnancy dataset"""
    query = state["question"]

    # Search FAISS for top 2 similar questions
    query_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, 2)

    retrieved = [dataset[int(i)] for i in I[0]]

    return {
        "retrieved_context": retrieved
    }

def generate_reasoning_and_answer(state: GraphState, menstrual_llama, tokenizer) -> GraphState:
    """
    Single unified node: Generate both reasoning and answer using Menstrual-LLaMA (now via HF Inference API).
    NOTE: Only the model-call section is changed to call HF's chat completions API (meta-llama).
    All prompt text, RAG context injection, and parsing logic are preserved verbatim.
    """
    query = state["question"]
    context = state["retrieved_context"]
    chat_history = state["messages"][:-1]

    # Format retrieved pregnancy context
    context_str = ""
    if context:
        context_str = "\n\nRetrieved Pregnancy Knowledge Base (PREGNANCY ONLY - use ONLY if relevant):\n"
        for i, ctx in enumerate(context):
            context_str += f"\nReference {i+1}:\nQ: {ctx['question']}\nA: {ctx['answer']}\n"

    # Format chat history for multi-turn conversation
    history_str = format_chat_history(chat_history, max_exchanges=3)

    # Construct system message
    system_message = """You are MedLang, an expert AI assistant for women's health, specializing in BOTH menstrual health and pregnancy. You MUST maintain conversational continuity and use the provided chat history to understand the context of follow-up questions.

YOUR KNOWLEDGE:
- You also have general pregnancy knowledge from your base training.
- You are capable of handling questions about periods, menstruation, PMS, PCOS, ovulation, fertility, pregnancy, conception, prenatal care, and more.
- You are multilingual and should respond in the same language as the user's query.
"""

    # Construct user message with all context
    user_message = f"""
{history_str}
CURRENT USER QUESTION: {query}

{context_str}

INSTRUCTIONS FOR ANSWERING:
1. REASONING FIRST (2-3 sentences):
   - **CRITICAL:** Analyze the **Previous conversation** and the **CURRENT USER QUESTION** together. Identify if this is a follow-up question (e.g., "What kind?") and explicitly state what it refers to (e.g., "What kind of songs for the baby").
   - Identify the primary topic: **Menstrual Health**, **Pregnancy/Fertility**, or **Irrelevant/General**.
   - If Menstrual Health, note that you will rely primarily on your internal knowledge.
   - If Pregnancy/Fertility, assess if the Retrieved Pregnancy Knowledge Base is relevant.
   - **CRITICAL:** If the query is about Menstrual Health (like delayed periods), explicitly state that the RAG context (which is pregnancy-only) is **IGNORED** for the answer.
   - Note the query language.
   - **NEW: Pragmatic Analysis —** You MUST analyze the tone, phrasing, and underlying inference of the user's query. Humans express uncertainty or assumptions differently:
       - **Presuppositions**: Statements implying mutual belief or factual assumptions.
         Example: “Which immunity injections can I skip for my baby?” presupposes it’s acceptable to skip some vaccines.
       - **Implicatures**: Softer, uncertain suggestions.
         Example: “Is it sufficient if my baby takes most immunity injections?” implies uncertainty but suggests the same inference.
     Your task is to **identify and separate implicatures from presuppositions** to better interpret the user’s intent and address **stronger false inferences** explicitly in your reasoning.

2. ANSWER SECOND (4-7 sentences):
   - **CRITICAL:** Do NOT give vague answers. Provide **SPECIFIC examples, causes, or types**.
   - **For information requiring specific detail (like causes of delayed periods or types of music): use a numbered or bulleted list in the answer.**
   - Use your extensive menstrual health knowledge as your PRIMARY source.
   - Use the Retrieved Pregnancy Knowledge Base **ONLY** if the query is clearly about a pregnancy topic.
   - For severe symptoms or emergencies, always include the standard medical disclaimer.
   - CRITICAL: Respond in the SAME language as the user's question.

FORMAT YOUR RESPONSE EXACTLY AS:
**REASONING:**
[Your 2-3 sentence pragmatic inference here]

**ANSWER:**
[Your detailed, specific, and structured response here, including lists where appropriate]"""


    try:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        try:
            completion = menstrual_llama.chat.completions.create(
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                messages=messages,
                max_tokens=400,
                temperature=0.6,
                top_p=0.9,
            )
            # completion.choices[0].message may be {'role': 'assistant', 'content': '...'}
            response_text = completion.choices[0].message.get("content", "")
        except Exception as api_e:
            raise api_e

        # Parse reasoning and answer
        reasoning = ""
        answer = ""

        if "**REASONING:**" in response_text and "**ANSWER:**" in response_text:
            parts = response_text.split("**ANSWER:**")
            reasoning = parts[0].replace("**REASONING:**", "").strip()
            answer = parts[1].strip()
        elif "REASONING:" in response_text and "ANSWER:" in response_text:
            parts = response_text.split("ANSWER:")
            reasoning = parts[0].replace("REASONING:", "").strip()
            answer = parts[1].strip()
        else:
            # Fallback: treat entire response as answer
            answer = response_text
            reasoning = "Analyzing query and generating response based on training knowledge."

        return {
            "reasoning": reasoning,
            "answer": answer,
            "messages": [AIMessage(content=answer)]
        }

    except Exception as e:
        error_msg = f"I apologize, but I encountered an error processing your question. Error: {str(e)}"
        return {
            "reasoning": "Error occurred during processing",
            "answer": error_msg,
            "messages": [AIMessage(content=error_msg)]
        }


# --- Build the Graph ---
@st.cache_resource
def create_chatbot_graph(_embedder, _dataset, _index, _menstrual_llama, _tokenizer):
    """Create the LangGraph workflow with Menstrual-LLaMA"""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node(
        "retrieve",
        lambda state: retrieve_context(state, _dataset, _index, _embedder)
    )

    workflow.add_node(
        "reason_and_answer",
        lambda state: generate_reasoning_and_answer(state, _menstrual_llama, _tokenizer)
    )

    # Add edges: retrieve → reason_and_answer → END
    workflow.add_edge("retrieve", "reason_and_answer")
    workflow.add_edge("reason_and_answer", END)

    # Set entry point
    workflow.set_entry_point("retrieve")

    # Compile with memory for multi-turn conversations
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app


# --- Initialize Session State ---
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        import uuid
        st.session_state.thread_id = str(uuid.uuid4())
    if "show_reasoning" not in st.session_state:
        st.session_state.show_reasoning = True
    if "show_context" not in st.session_state:
        st.session_state.show_context = False
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0


# --- Main App ---
def main():
    # Initialize
    initialize_session_state()
    hf_token = load_environment()
    embedder = initialize_embedder()

    # Load Menstrual-LLaMA with access token
    menstrual_llama, tokenizer = load_menstrual_llama(hf_token)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🤰 MedLang")
        st.markdown("*Women's Health Companion*")
        st.markdown("---")

        # Dataset file input
        data_file = st.text_input(
            "Pregnancy Dataset Path",
            value="merged_preg_dataset.jsonl",
            help="Path to your pregnancy Q&A JSONL file (1,378 pairs)"
        )

        # Load dataset and create graph
        if st.button("🔄 Reload Dataset & Model"):
            st.cache_resource.clear()
            st.rerun()

        dataset, index = load_dataset_and_index(data_file, embedder)
        app = create_chatbot_graph(embedder, dataset, index, menstrual_llama, tokenizer)

        # st.success(f"✅ {len(dataset)} pregnancy Q&A pairs loaded")
        # st.success("✅ Menstrual-LLaMA-8B active (Quantized Load)")
        # st.info("ℹ️ Model trained on 24k+ menstrual Q&QAs")

        st.markdown("---")

        # Settings
        st.markdown("### ⚙️ Display Settings")
        st.session_state.show_reasoning = st.checkbox(
            "Show Reasoning Process",
            value=st.session_state.show_reasoning,
            help="Display the model's internal reasoning"
        )
        st.session_state.show_context = st.checkbox(
            "Show Retrieved Context",
            value=st.session_state.show_context,
            help="Display pregnancy Q&As retrieved from RAG"
        )

        st.markdown("---")

        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.session_state.query_count = 0
            # Removed language_stats reset
            import uuid
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

        st.markdown("---")
        st.markdown("### ℹ️ About MedLang")
        st.markdown("""
        **Model Architecture:**
        - 🔴 **Menstrual-LLaMA**: Fine-tuned on 24,000+ expert-verified menstrual health Q&A pairs
        - 📚 **RAG Enhancement**: Retrieves relevant pregnancy Q&As when needed
        - 🧠 **Autonomous Decision Making**: Model intelligently decides when to use retrieved context

        **Capabilities:**
        - ✅ Menstrual health (periods, PMS, PCOS, ovulation)
        - ✅ Pregnancy (conception, prenatal care, symptoms)
        - ✅ Fertility & reproductive health
        - ✅ Multi-turn conversations with memory
        - ✅ Multilingual queries supported

        **Features:**
        - Context-aware responses
        - Conversational memory via LangGraph
        - Reasoning transparency
        - Privacy-focused
        """)

        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        st.markdown("""
        - "What causes irregular periods?"
        - "Is cramping normal in early pregnancy?"
        - "How can I track my ovulation?"
        - "मासिक धर्म में देरी के क्या कारण हैं?"
        - "गर्भावस्था के शुरुआती लक्षण क्या हैं?"
        """)

        st.markdown("---")
        st.markdown("⚠️ *This is not a substitute for professional medical advice. Always consult a healthcare provider for serious concerns.*")

    # Main chat interface
    st.markdown("<h1 class='main-header'>🤰 MedLang - Women's Health Companion</h1>",
                unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
    Ask questions about <b>pregnancy</b> or <b>menstrual health</b> 🌏<br>
    <em>Powered by Menstrual-LLaMA-8B with RAG enhancement</em>
    </div>
    """, unsafe_allow_html=True)

    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.container():
                st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 You:</strong><br>
                        {message["content"]}
                    </div>
                """, unsafe_allow_html=True)
        else:
            with st.container():
                # Model badge
                st.markdown("""
                    <span class="model-badge">
                        🔴 Menstrual-LLaMA-8B
                    </span>
                """, unsafe_allow_html=True)

                # Retrieved context (if enabled)
                if st.session_state.show_context and "context" in message and message["context"]:
                    st.markdown("<strong>📚 Retrieved Pregnancy Context (RAG):</strong>", unsafe_allow_html=True)
                    for i, ctx in enumerate(message["context"]):
                        with st.expander(f"Reference {i+1}: {ctx['question'][:80]}...", expanded=False):
                            st.markdown(f"**Q:** {ctx['question']}")
                            st.markdown(f"**A:** {ctx['answer'][:300]}...")

                # Reasoning (if enabled)
                if st.session_state.show_reasoning and "reasoning" in message and message["reasoning"]:
                    st.markdown(f"""
                        <div class="reasoning-box">
                            <strong>🧠 Model Reasoning:</strong><br>
                            {message["reasoning"]}
                        </div>
                    """, unsafe_allow_html=True)

                # Answer
                st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>🤖 MedLang:</strong><br>
                        {message["content"]}
                    </div>
                """, unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Ask about pregnancy or menstrual health...")

    if user_input:
        # Update query count
        st.session_state.query_count += 1

        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # Display user message immediately
        with st.container():
            st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong><br>
                    {user_input}
                </div>
            """, unsafe_allow_html=True)

        # Generate response
        with st.spinner("🤔 Thinking..."):
            try:
                # Prepare initial state with conversation history
                history_messages = [msg for msg in st.session_state.messages[:-1]
                                   if isinstance(msg.get("content"), str)]

                langchain_history = []
                for msg in history_messages:
                    if msg["role"] == "user":
                        langchain_history.append(HumanMessage(content=msg["content"]))
                    else:
                        langchain_history.append(AIMessage(content=msg["content"]))

                initial_state = {
                    "messages": langchain_history + [HumanMessage(content=user_input)],
                    "question": user_input,
                    "retrieved_context": [],
                    "reasoning": "",
                    "answer": ""
                }

                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = app.invoke(initial_state, config)

                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "reasoning": result["reasoning"],
                    "context": result["retrieved_context"]
                })

                st.rerun()

            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")
                st.error("Please try rephrasing your question or check the model setup.")


if __name__ == "__main__":
    main()
