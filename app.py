import streamlit as st
import json
import faiss
import numpy as np
from typing import TypedDict, Annotated, Sequence
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from huggingface_hub import InferenceClient

# Page configuration
st.set_page_config(
    page_title="MedLang - Women's Health Assistant",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load environment variables ---
@st.cache_resource
def load_environment():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        st.error("⚠️ HF_TOKEN not found in environment or Streamlit Secrets!")
        st.stop()
    return hf_token


# --- Initialize embedder ---
@st.cache_resource
def initialize_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# --- Load Meta-Llama via Inference API ---
@st.cache_resource
def load_menstrual_llama(hf_token):
    """
    Connect to Hugging Face Inference API for Meta-Llama-3-8B-Instruct.
    """
    try:
        client = InferenceClient(api_key=hf_token)
        st.success("✅ Connected to Meta-Llama-3-8B-Instruct via HF Inference API")
        return client
    except Exception as e:
        st.error(f"⚠️ Could not initialize HF client: {str(e)}")
        st.stop()


# --- Load dataset + FAISS index ---
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
    except Exception as e:
        st.error(f"⚠️ Dataset loading error: {str(e)}")
        st.stop()


# --- Graph state ---
class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    question: str
    retrieved_context: list
    reasoning: str
    answer: str


# --- Helper for chat history ---
def format_chat_history(messages, max_exchanges=3):
    if not messages:
        return ""
    history_str = "Previous conversation:\n"
    for msg in messages[-(max_exchanges*2):]:
        if isinstance(msg, HumanMessage):
            history_str += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_str += f"Assistant: {msg.content}\n"
    return history_str + "\n"


# --- Retrieve top 2 context Q&As ---
def retrieve_context(state: GraphState, dataset, index, embedder) -> GraphState:
    query = state["question"]
    query_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, 2)
    retrieved = [dataset[int(i)] for i in I[0]]
    return {"retrieved_context": retrieved}


# --- Generate reasoning + answer (Meta-Llama Inference API) ---
def generate_reasoning_and_answer(state: GraphState, client) -> GraphState:
    query = state["question"]
    context = state["retrieved_context"]
    chat_history = state["messages"][:-1]

    context_str = ""
    if context:
        context_str = "\n\nRetrieved Pregnancy Knowledge Base:\n"
        for i, ctx in enumerate(context):
            context_str += f"\nReference {i+1}:\nQ: {ctx['question']}\nA: {ctx['answer']}\n"

    history_str = format_chat_history(chat_history, max_exchanges=3)

    system_message = """You are MedLang, an expert AI assistant for women's health, specializing in menstrual health and pregnancy."""

    user_prompt = f"""
{system_message}

{history_str}
CURRENT QUESTION: {query}

{context_str}

INSTRUCTIONS:
1. Provide a short **Reasoning** section (2–3 sentences).
2. Then provide a detailed **Answer** (5–7 sentences) with bullet points if relevant.
3. Keep responses medically accurate and empathetic.
"""

    try:
        # --- Call Meta-Llama-3-8B-Instruct ---
        completion = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=400,
            temperature=0.6,
            top_p=0.9,
        )

        response_text = completion.choices[0].message["content"]

        # Separate reasoning + answer if formatted
        reasoning, answer = "", response_text
        if "**REASONING:**" in response_text and "**ANSWER:**" in response_text:
            parts = response_text.split("**ANSWER:**")
            reasoning = parts[0].replace("**REASONING:**", "").strip()
            answer = parts[1].strip()

        return {
            "reasoning": reasoning or "Reasoning generated by model.",
            "answer": answer,
            "messages": [AIMessage(content=answer)]
        }

    except Exception as e:
        error_msg = f"I encountered an error: {str(e)}"
        return {
            "reasoning": "Error during inference",
            "answer": error_msg,
            "messages": [AIMessage(content=error_msg)]
        }


# --- Build LangGraph ---
@st.cache_resource
def create_chatbot_graph(_embedder, _dataset, _index, _client):
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", lambda s: retrieve_context(s, _dataset, _index, _embedder))
    workflow.add_node("reason_and_answer", lambda s: generate_reasoning_and_answer(s, _client))
    workflow.add_edge("retrieve", "reason_and_answer")
    workflow.add_edge("reason_and_answer", END)
    workflow.set_entry_point("retrieve")
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# --- Initialize session ---
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


# --- Main Streamlit App ---
def main():
    initialize_session_state()
    hf_token = load_environment()
    embedder = initialize_embedder()
    client = load_menstrual_llama(hf_token)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🤰 MedLang")
        data_file = st.text_input("Dataset path", value="merged_preg_dataset.jsonl")
        dataset, index = load_dataset_and_index(data_file, embedder)
        app = create_chatbot_graph(embedder, dataset, index, client)
        st.markdown("---")
        st.session_state.show_reasoning = st.checkbox("Show Reasoning", value=True)
        st.session_state.show_context = st.checkbox("Show Retrieved Context", value=False)
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.session_state.query_count = 0
            import uuid
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

    # Main chat display
    st.markdown("<h1 style='text-align:center;'>🤰 MedLang - Women's Health Companion</h1>", unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"👤 **You:** {msg['content']}")
        else:
            if st.session_state.show_reasoning and "reasoning" in msg:
                st.markdown(f"🧠 **Reasoning:** {msg['reasoning']}")
            st.markdown(f"🤖 **MedLang:** {msg['content']}")

    # Chat input
    user_input = st.chat_input("Ask about pregnancy or menstrual health...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("🤔 Thinking..."):
            history = [HumanMessage(content=m["content"]) if m["role"] == "user"
                       else AIMessage(content=m["content"])
                       for m in st.session_state.messages[:-1]]
            initial_state = {
                "messages": history + [HumanMessage(content=user_input)],
                "question": user_input,
                "retrieved_context": [],
                "reasoning": "",
                "answer": ""
            }
            result = app.invoke(initial_state, {"configurable": {"thread_id": st.session_state.thread_id}})
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "reasoning": result["reasoning"]
            })
            st.rerun()


if __name__ == "__main__":
    main()
