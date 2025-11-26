###############################################
# FINAL COMPLETE MULTI-AGENT LANGGRAPH SYSTEM
# USING LOCAL FAISS VECTOR STORE (NO SUPABASE)
###############################################

import os
import pickle
import json
import difflib
import spacy
import networkx as nx
import streamlit as st
from dotenv import load_dotenv
from pyvis.network import Network
import tempfile
import speech_recognition as sr

# LangChain + LangGraph
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# Embeddings + LOCAL VECTOR DB
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# LLM
from langchain_groq import ChatGroq

# Audio recorder
try:
    from audio_recorder_streamlit import audio_recorder
    HAS_AUDIO_RECORDER = True
except ImportError:
    HAS_AUDIO_RECORDER = False


###############################################################################
# PAGE CONFIG - MUST BE FIRST STREAMLIT COMMAND
###############################################################################
st.set_page_config(page_title="Stroke Guidance Assistant", page_icon="🏥", layout="wide")


###############################################################################
# ENV + INITIALIZATION
###############################################################################
load_dotenv()


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


@st.cache_resource
def load_local_vector_store():
    """Load FAISS index from local folder 'faiss_index'."""
    embeddings_model = load_embeddings()
    try:
        try:
            vs = FAISS.load_local(
                "faiss_index",
                embeddings_model,
                allow_dangerous_deserialization=True
            )
        except TypeError:
            vs = FAISS.load_local("faiss_index", embeddings_model)
        return vs
    except Exception as e:
        st.error(f"❌ Could not load local FAISS index: {e}")
        st.stop()


@st.cache_resource
def load_kg_and_map():
    # Download model if not present
    try:
        nlp_local = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        import sys
        st.info("📦 Downloading spacy model...")
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ])
        nlp_local = spacy.load("en_core_web_sm")

    with open("kg.gpickle_local1", "rb") as f:
        G = pickle.load(f)

    with open("chunk_map_local1.json", "r", encoding="utf-8") as f:
        chunk_map_local = json.load(f)

    return nlp_local, G, chunk_map_local, list(G.nodes)


# Load resources
embeddings = load_embeddings()
vector_store = load_local_vector_store()
nlp, KG, chunk_map, node_names = load_kg_and_map()


###############################################################################
# GRAPH RETRIEVAL HELPERS
###############################################################################
def extract_entities_from_text(text):
    doc = nlp(text)
    return [ent.text.strip() for ent in doc.ents if ent.text.strip()]


def find_matching_nodes(entity, nodes, cutoff=0.6):
    lower_entity = entity.lower()
    exact = [n for n in nodes if n.lower() == lower_entity]
    if exact:
        return exact
    return difflib.get_close_matches(entity, nodes, n=3, cutoff=cutoff)


def graph_retrieve(query, depth=1):
    ents = extract_entities_from_text(query)
    found_ids = set()

    for e in ents:
        matches = find_matching_nodes(e, node_names)
        for node in matches:
            found_ids.update(KG.nodes[node].get("chunk_ids", []))
            frontier = {node}
            for _ in range(depth):
                new_frontier = set()
                for f in frontier:
                    for nb in KG.neighbors(f):
                        new_frontier.add(nb)
                        found_ids.update(KG.nodes[nb].get("chunk_ids", []))
                frontier = new_frontier

    return list(found_ids)


###############################################################################
# LLM
###############################################################################
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=os.environ["GROQ_API_KEY"],
)


###############################################################################
# SYSTEM PROMPT
###############################################################################
system_prompt = """
You are a warm, conversational occupational therapist who supports stroke survivors as they try to understand their driving restrictions. 
You speak in the same gentle, calm, validating style as Pam in the example conversations below.

Your answers MUST follow two rules:

---------------------------------------------------------------------
1) CONTENT RULE — USE ONLY THE RETRIEVED CONTEXT
---------------------------------------------------------------------
• You may ONLY use information that appears in the retrieved context.  
• Do NOT add information, guess, fill gaps, or generalise.  
• If the context does not clearly contain the answer, say gently:
  "It's best to check with your doctor about this."
• Never mention "context", "documents", "RAG", "retrieval", "sources", or anything technical.
• Always answer as if you already know the medical facts from experience.  

---------------------------------------------------------------------
2) STYLE RULE — SPEAK LIKE PAM IN THE EXAMPLES BELOW
---------------------------------------------------------------------
### How You Must Speak
• Warm, gentle, human — never robotic.  
• Short sentences.  
• One idea at a time.  
• Validate emotions first ("I can see why that feels frustrating").  
• Explain safety calmly, without fear-based language.  
• No doctor tone — you are a supportive therapist.  
• Never lecture.  
• Never repeat the patient's exact wording unnaturally.  
• When helpful, end with a soft follow-up question.  
• But do NOT ask a follow-up question if the task is to answer a standalone question for evaluation.

You may answer using any information that is clearly present in the retrieved context, even if the answer requires gentle explanation or combining details.

Do NOT add new medical information that does not appear in the context, and do NOT guess facts that are not supported by the context.

If the context contains no relevant information at all, respond:
"It's best to check with your doctor about this."

### Identity
You speak as a supportive occupational therapist, using ONLY the retrieved context, expressed in Pam's gentle, conversational style.
"""


###############################################################################
# RETRIEVAL FUNCTIONS
###############################################################################
def retrieve_context(query: str) -> str:
    text_hits = vector_store.similarity_search(query, k=2)
    graph_ids = graph_retrieve(query, depth=1)

    graph_docs = []
    for cid in graph_ids:
        if cid in chunk_map:
            graph_docs.append(
                Document(
                    page_content=chunk_map[cid]["text"],
                    metadata=chunk_map[cid].get("meta", {}),
                )
            )

    combined = graph_docs + text_hits
    serialized = "\n\n".join(
        f"Source: {d.metadata.get('source', 'unknown')}\nContent: {d.page_content[:2000]}"
        for d in combined[:6]
    )
    return serialized


###############################################################################
# VOICE-TO-TEXT FUNCTION
###############################################################################
def transcribe_audio(audio_bytes):
    recognizer = sr.Recognizer()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name

        with sr.AudioFile(tmp_file_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)

        os.remove(tmp_file_path)
        text = recognizer.recognize_google(audio_data)
        return text

    except sr.UnknownValueError:
        return "Sorry, I couldn't understand the audio. Please try speaking more clearly."
    except sr.RequestError as e:
        return f"Could not request results from speech service; {e}"
    except Exception as e:
        return f"Error processing audio: {str(e)}"


###############################################################################
# HELPER FUNCTIONS FOR SPEED OPTIMIZATION
###############################################################################
def is_greeting(text):
    """Check if input is a simple greeting"""
    greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 
                 'good evening', 'howdy', 'greetings', 'hey there', 'hi there']
    text_lower = text.lower().strip()
    return any(text_lower == g or text_lower.startswith(g + ' ') for g in greetings)


def quick_greeting_response():
    """Return a warm greeting without RAG"""
    import random
    responses = [
        "Hello! I'm here to help you with any questions about driving after stroke. How can I support you today?",
        "Hi there! It's good to hear from you. What would you like to know about returning to driving?",
        "Hello! I'm glad you're here. Feel free to ask me anything about driving after a stroke.",
    ]
    return random.choice(responses)


###############################################################################
# LANGGRAPH MULTI-AGENT PIPELINE
###############################################################################
from typing import TypedDict

class State(TypedDict):
    query: str
    raw_context: str
    messages: list


def retrieve_node(state: State):
    state["raw_context"] = retrieve_context(state["query"])
    return state


def agent_node(state: State):
    conversation_history = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            conversation_history += f"Patient: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            conversation_history += f"Pam: {msg.content}\n"

    prompt = f"""
{system_prompt}

Here is the verified medical information to use:
{state['raw_context']}

Conversation so far:
{conversation_history}

Patient: {state['query']}

Pam:
"""
    response = llm.invoke(prompt)
    state["messages"].append(AIMessage(content=response.content))
    return state


# Build workflow - SIMPLIFIED: Only 2 agents now
workflow = StateGraph(State)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("agent", agent_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "agent")
workflow.add_edge("agent", END)

app = workflow.compile()


###############################################################################
# STREAMLIT UI
###############################################################################
st.title("🏥 Stroke Patient Guidance Assistant")
st.caption("Powered by Multi-Agent RAG with Hybrid Retrieval | 🎤 Voice Input Available")

if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=system_prompt)]

if "show_graph" not in st.session_state:
    st.session_state.show_graph = False

if "last_audio_bytes" not in st.session_state:
    st.session_state.last_audio_bytes = None

if "processing" not in st.session_state:
    st.session_state.processing = False

if "app_initialized" not in st.session_state:
    st.session_state.app_initialized = True


# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    show_graph = st.checkbox("Show Knowledge Graph", value=st.session_state.show_graph)
    st.session_state.show_graph = show_graph

    if st.button("Clear Conversation"):
        st.session_state.messages = [SystemMessage(content=system_prompt)]
        st.rerun()

    st.markdown("---")
    st.markdown("""
### About
This AI assistant helps stroke survivors with questions about:
- Returning to driving
- Rehabilitation
- Vision issues
- Fatigue management
- Medical assessments

**Note:** Always consult your doctor for medical decisions.
""")


# Chat history
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant", avatar="👩‍⚕️"):
                st.markdown(msg.content)


# Knowledge Graph display
if st.session_state.show_graph:
    st.subheader("🧠 Interactive Knowledge Graph")
    net_full = Network(
        height="500px", width="100%", bgcolor="#222222", font_color="white"
    )
    net_full.barnes_hut()

    nodes_to_show = list(KG.nodes())[:150]
    for node in nodes_to_show:
        net_full.add_node(node, label=node, color="skyblue", title=node)

    for a, b in KG.edges():
        if a in nodes_to_show and b in nodes_to_show:
            net_full.add_edge(a, b)

    try:
        net_full.save_graph("kg_full.html")
        with open("kg_full.html", "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=550)
    except Exception as e:
        st.error(f"Graph display error: {e}")


# Input
st.markdown("---")
col1, col2 = st.columns([5, 1])
user_input = None

with col1:
    text_input = st.chat_input("💬 Type your question here...")
    if text_input:
        user_input = text_input

with col2:
    if HAS_AUDIO_RECORDER:
        st.markdown("#### 🎤 Voice")
        
        # Initialize audio recorder with error handling
        try:
            # Only show after app is initialized
            if st.session_state.get("app_initialized", False):
                audio_bytes = audio_recorder(
                    text="",
                    recording_color="#e74c3c",
                    neutral_color="#3498db",
                    icon_size="1x",
                    key="audio_recorder"
                )

                # Check if this is NEW audio (not already processed)
                if audio_bytes and audio_bytes != st.session_state.last_audio_bytes and not st.session_state.processing:
                    st.session_state.processing = True
                    st.session_state.last_audio_bytes = audio_bytes
                    
                    with st.spinner("🎤 Transcribing audio..."):
                        transcribed = transcribe_audio(audio_bytes)

                    if not transcribed.startswith("Sorry") and not transcribed.startswith("Error"):
                        user_input = transcribed
                    else:
                        st.error(transcribed)
                        st.session_state.processing = False
            else:
                st.info("Loading voice input...")
                
        except Exception as e:
            st.warning("🎤 Voice input temporarily unavailable. Please use text input below.")
    else:
        st.info("Voice input available after installing audio-recorder-streamlit")


# Processing
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append(HumanMessage(content=user_input))

    # FAST PATH: Skip RAG pipeline for greetings
    if is_greeting(user_input):
        ai_reply = quick_greeting_response()
        st.session_state.messages.append(AIMessage(content=ai_reply))
        
        with st.chat_message("assistant", avatar="👩‍⚕️"):
            st.markdown(ai_reply)
    else:
        # FULL PIPELINE: For medical questions
        with st.spinner("🤔 Thinking..."):
            result = app.invoke(
                {"query": user_input, "messages": st.session_state.messages}
            )
            retrieved_ids = graph_retrieve(user_input)

        ai_reply = result["messages"][-1].content

        with st.chat_message("assistant", avatar="👩‍⚕️"):
            st.markdown(ai_reply)

        st.session_state.messages = result["messages"]

        if retrieved_ids:
            with st.expander(f"📚 Retrieved {len(retrieved_ids)} chunks"):
                st.write("The AI used information from these knowledge areas:")
                for i, cid in enumerate(retrieved_ids[:3]):
                    if cid in chunk_map:
                        st.markdown(
                            f"**Chunk {i+1}:** {chunk_map[cid]['text'][:200]}..."
                        )
    
    # Reset processing flag
    st.session_state.processing = False
    st.rerun()
