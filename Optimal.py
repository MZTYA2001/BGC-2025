import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from streamlit_mic_recorder import speech_to_text
import fitz  # PyMuPDF for PDF handling
import pdfplumber  # For searching text in PDF
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from datetime import datetime

# Initialize API key variables
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

# Initialize the sentence transformer model
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Change the page title and icon
st.set_page_config(
    page_title="BGC ChatBot",  # Page title
    page_icon="BGC Logo Colored.svg",  # New page icon
    layout="wide"  # Page layout
)

# Function to apply CSS based on language direction
def apply_css_direction(direction):
    st.markdown(
        f"""
        <style>
            .stApp {{
                direction: {direction};
                text-align: {direction};
            }}
            .stChatInput {{
                direction: {direction};
            }}
            .stChatMessage {{
                direction: {direction};
                text-align: {direction};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# PDF Search and Screenshot Class
class PDFSearchAndDisplay:
    def __init__(self):
        pass

    def search_and_highlight(self, pdf_path, search_term):
        highlighted_pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages):
                text = page.extract_text()
                if search_term in text:
                    highlighted_pages.append((page_number, text))
        return highlighted_pages

    def capture_screenshots(self, pdf_path, pages):
        doc = fitz.open(pdf_path)
        screenshots = []
        for page_number, _ in pages:
            page = doc.load_page(page_number)
            pix = page.get_pixmap()
            screenshot_path = f"screenshot_page_{page_number}.png"
            pix.save(screenshot_path)
            screenshots.append(screenshot_path)
        return screenshots

# Sidebar configuration
with st.sidebar:
    # Language selection dropdown
    interface_language = st.selectbox("Interface Language", ["English", "العربية"])

    # Apply CSS direction based on selected language
    if interface_language == "العربية":
        apply_css_direction("rtl")  # Right-to-left for Arabic
        st.title("الإعدادات")  # Sidebar title in Arabic
    else:
        apply_css_direction("ltr")  # Left-to-right for English
        st.title("Settings")  # Sidebar title in English

    # Validate API key inputs and initialize components if valid
    if groq_api_key and google_api_key:
        # Set Google API key as environment variable
        os.environ["GOOGLE_API_KEY"] = google_api_key

        # Initialize ChatGroq with the provided Groq API key
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

        # Initialize the chat prompt template with improved context handling
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """You are a helpful AI assistant that provides accurate information based on the given context. 
                When you see tables, diagrams, or structured content in the context:
                1. Respond with high confidence if the information is clearly present
                2. Provide complete and accurate details from the table/content
                3. Do not express uncertainty when the information is directly available
                4. Format tables and structured data clearly in your response
                
                Current conversation memory:
                {history}
                """
            ),
            HumanMessagePromptTemplate.from_template(
                """Context information:
                {context}
                
                Please provide a clear and accurate response to this question: {question}
                If you see a table or structured content in the context, make sure to present it accurately.
                """
            )
        ])

        def prepare_context(context_docs):
            """
            Prepare context for the LLaMA model with improved handling of tables and structured content
            """
            context_text = []
            has_table = False
            
            for doc in context_docs:
                content = doc.page_content.strip()
                if "table" in content.lower() or "|" in content or "+" in content:
                    has_table = True
                    # Preserve table formatting
                    content = f"\nTable content:\n{content}\n"
                context_text.append(content)
            
            final_context = "\n\n".join(context_text)
            if has_table:
                final_context = f"Important: The following context contains structured table data. Please maintain accuracy in representing this information:\n\n{final_context}"
            
            return final_context

        def process_query(query, context_docs):
            """
            Process query with improved context handling for better accuracy
            """
            # Prepare context with better table handling
            context = prepare_context(context_docs) if context_docs else ""
            
            # Get chat history
            history = get_chat_history()
            
            # Create the chat prompt with improved context
            chat_input = chat_prompt.format_prompt(
                history=history,
                context=context,
                question=query
            )
            
            # Get response from LLaMA
            messages = [{"role": m.type, "content": m.content} for m in chat_input.to_messages()]
            response = llm.chat_completion(messages)
            
            return {
                "answer": response["choices"][0]["message"]["content"],
                "context": context_docs
            }

        def get_chat_history():
            """Get formatted chat history with improved context retention"""
            history_messages = []
            
            for message in st.session_state.messages:
                role = message["role"]
                content = message["content"]
                
                # Add special handling for messages with page references
                msg_index = st.session_state.messages.index(message)
                if msg_index in st.session_state.page_references:
                    page_refs = st.session_state.page_references[msg_index]
                    content += f"\n[Referenced pages: {page_refs}]"
                    
                history_messages.append(f"{role.capitalize()}: {content}")
            
            return "\n".join(history_messages)

        # Load existing embeddings from files
        if "vectors" not in st.session_state:
            with st.spinner("جارٍ تحميل التضميدات... الرجاء الانتظار." if interface_language == "العربية" else "Loading embeddings... Please wait."):
                # Initialize embeddings
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001"
                )

                # Load existing FAISS index with safe deserialization
                embeddings_path = "embeddings"  # Path to your embeddings folder
                embeddings_path_2 = "embeddingsocr"
                
                try:
                    # Load first FAISS index
                    vectors_1 = FAISS.load_local(
                    embeddings_path, embeddings, allow_dangerous_deserialization=True
                    )

                    # Load second FAISS index
                    vectors_2 = FAISS.load_local(
                    embeddings_path_2, embeddings, allow_dangerous_deserialization=True
                    )

                    # Merge both FAISS indexes
                    vectors_1.merge_from(vectors_2)

                    # Store in session state
                    st.session_state.vectors = vectors_1

                except Exception as e:
                    st.error(f"Error loading embeddings: {str(e)}")
                    st.session_state.vectors = None
        # Microphone button in the sidebar
        st.markdown("### الإدخال الصوتي" if interface_language == "العربية" else "### Voice Input")
        input_lang_code = "ar" if interface_language == "العربية" else "en"  # Set language code based on interface language
        voice_input = speech_to_text(
            start_prompt="🎤",
            stop_prompt="⏹️ إيقاف" if interface_language == "العربية" else "⏹️ Stop",
            language=input_lang_code,  # Language (en for English, ar for Arabic)
            use_container_width=True,
            just_once=True,
            key="mic_button",
        )

        # Reset button in the sidebar
        if st.button("إعادة تعيين الدردشة" if interface_language == "العربية" else "Reset Chat"):
            st.session_state.messages = []  # Clear chat history
            st.session_state.memory.clear()  # Clear memory
            st.success("تمت إعادة تعيين الدردشة بنجاح." if interface_language == "العربية" else "Chat has been reset successfully.")
            st.rerun()  # Rerun the app to reflect changes immediately
    else:
        st.error("الرجاء إدخال مفاتيح API للمتابعة." if interface_language == "العربية" else "Please enter both API keys to proceed.")

# Initialize the PDFSearchAndDisplay class with the default PDF file
pdf_path = "BGC.pdf"
pdf_searcher = PDFSearchAndDisplay()

# Main area for chat interface
# Use columns to display logo and title side by side
col1, col2 = st.columns([1, 4])  # Adjust the ratio as needed

# Display the logo in the first column
with col1:
    st.image("BGC Logo Colored.svg", width=100)  # Adjust the width as needed

# Display the title and description in the second column
with col2:
    if interface_language == "العربية":
        st.title("بوت الدردشة BGC")
        st.write("""
        **مرحبًا!**  
        هذا بوت الدردشة الخاص بشركة غاز البصرة (BGC). يمكنك استخدام هذا البوت للحصول على معلومات حول الشركة وأنشطتها.  
        **كيفية الاستخدام:**  
        - اكتب سؤالك في مربع النص أدناه.  
        - أو استخدم زر المايكروفون للتحدث مباشرة.  
        - سيتم الرد عليك بناءً على المعلومات المتاحة.  
        """)
    else:
        st.title("BGC ChatBot")
        st.write("""
        **Welcome!**  
        This is the Basrah Gas Company (BGC) ChatBot. You can use this bot to get information about the company and its activities.  
        **How to use:**  
        - Type your question in the text box below.  
        - Or use the microphone button to speak directly.  
        - You will receive a response based on the available information.  
        """)

# Initialize session state for chat messages if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for page references if not already done
if "page_references" not in st.session_state:
    st.session_state.page_references = {}

# Initialize memory if not already done
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )

# List of negative phrases to check for unclear or insufficient answers
negative_phrases = [
    "I'm sorry",
    "عذرًا",
    "لا أملك معلومات كافية",
    "I don't have enough information",
    "لم أتمكن من فهم سؤالك",
    "I couldn't understand your question",
    "لا يمكنني الإجابة على هذا السؤال",
    "I cannot answer this question",
    "يرجى تقديم المزيد من التفاصيل",
    "Please provide more details",
    "غير واضح",
    "Unclear",
    "غير متأكد",
    "Not sure",
    "لا أعرف",
    "I don't know",
    "غير متاح",
    "Not available",
    "غير موجود",
    "Not found",
    "غير معروف",
    "Unknown",
    "غير محدد",
    "Unspecified",
    "غير مؤكد",
    "Uncertain",
    "غير كافي",
    "Insufficient",
    "غير دقيق",
    "Inaccurate",
    "غير مفهوم",
    "Not clear",
    "غير مكتمل",
    "Incomplete",
    "غير صحيح",
    "Incorrect",
    "غير مناسب",
    "Inappropriate",
    "Please provide me",  # إضافة هذه العبارة
    "يرجى تزويدي",  # إضافة هذه العبارة
    "Can you provide more",  # إضافة هذه العبارة
    "هل يمكنك تقديم المزيد"  # إضافة هذه العبارة
]

def calculate_semantic_similarity(text1, text2):
    """Calculate semantic similarity between two texts using all-MiniLM-L6-v2"""
    # Encode texts to vectors
    embedding1 = sentence_model.encode(text1, convert_to_tensor=True)
    embedding2 = sentence_model.encode(text2, convert_to_tensor=True)
    
    # Calculate cosine similarity
    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)
    return similarity.item()

def get_relevant_pages(context_docs, query_text, min_similarity=0.4):
    """
    Find relevant pages using semantic similarity with all-MiniLM-L6-v2 model.
    Returns dict of {page_number: {"score": similarity_score, "content": content}}
    """
    page_scores = {}
    
    # Process each document/page
    for doc in context_docs:
        page_number = doc.metadata.get("page", "unknown")
        if page_number != "unknown" and str(page_number).isdigit():
            page_number = int(page_number)
            content = doc.page_content.strip()
            
            # Skip very short content
            if len(content) < 10:
                continue
            
            # Calculate similarity score
            similarity_score = calculate_semantic_similarity(query_text, content)
            
            # Store if score is higher than minimum threshold
            if similarity_score >= min_similarity:
                if page_number not in page_scores or similarity_score > page_scores[page_number]["score"]:
                    page_scores[page_number] = {
                        "score": similarity_score,
                        "content": content
                    }
    
    return dict(sorted(page_scores.items(), key=lambda x: x[1]["score"], reverse=True))

def process_response(input_text, response, is_voice=False):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": input_text})
    with st.chat_message("user"):
        st.markdown(input_text)

    # Get assistant response and context
    assistant_response = response["answer"]
    context = response.get("context", [])
    current_msg_index = len(st.session_state.messages)

    # Check if the response suggests uncertainty
    uncertain_response = any(phrase.lower() in assistant_response.lower() for phrase in negative_phrases)
    
    # If uncertain but we have context with tables or specific content, override uncertainty
    if uncertain_response and context:
        has_tables = any("table" in doc.page_content.lower() for doc in context)
        if has_tables:
            # Remove uncertainty phrases from response
            for phrase in negative_phrases:
                assistant_response = assistant_response.replace(phrase, "")
            uncertain_response = False

    # Add assistant message to session state and display it
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.markdown(assistant_response)

    # Update memory
    st.session_state.memory.chat_memory.add_user_message(input_text)
    st.session_state.memory.chat_memory.add_ai_message(assistant_response)
    
    # Process page references if we have context
    if context:
        # Combine question and answer for context
        query_text = f"{input_text} {assistant_response}"
        
        # Get relevant pages using semantic similarity
        relevant_pages = get_relevant_pages(context, query_text)
        
        # Only show page references if we found relevant pages
        if relevant_pages:
            page_numbers_str = ", ".join(map(str, sorted(relevant_pages.keys())))
            st.session_state.page_references[current_msg_index] = page_numbers_str
            
            # Display the page references expander
            with st.chat_message("assistant"):
                with st.expander("مراجع الصفحات" if interface_language == "العربية" else "Page References", expanded=False):
                    st.write(f"هذه الإجابة وفقًا للصفحات: {page_numbers_str}" if interface_language == "العربية" else f"This answer references pages: {page_numbers_str}")
                    
                    # Display relevance scores
                    st.write("Semantic Similarity Scores:")
                    for page_num in sorted(relevant_pages.keys()):
                        score = relevant_pages[page_num]["score"]
                        st.write(f"Page {page_num}: {score:.3f}")
                    
                    # Create tabs for each page
                    tabs = st.tabs([f"Page {page}" for page in sorted(relevant_pages.keys())])
                    for tab, page_num in zip(tabs, sorted(relevant_pages.keys())):
                        with tab:
                            # Capture and display screenshot
                            doc = fitz.open(pdf_path)
                            page = doc.load_page(page_num)
                            pix = page.get_pixmap()
                            screenshot_path = f"screenshot_page_{page_num}.png"
                            pix.save(screenshot_path)
                            doc.close()
                            st.image(screenshot_path)

# Display chat history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display page references for each message if they exist
        if i in st.session_state.page_references and st.session_state.page_references[i]:
            with st.expander("مراجع الصفحات" if interface_language == "العربية" else "Page References", expanded=False):
                page_numbers_str = st.session_state.page_references[i]
                st.write(f"هذه الإجابة وفقًا للصفحات: {page_numbers_str}" if interface_language == "العربية" else f"This answer references pages: {page_numbers_str}")
                
                # Display relevance scores
                st.write("Semantic Similarity Scores:")
                page_numbers = [int(page.strip()) for page in page_numbers_str.split(",")]
                
                # Create tabs for each page
                tabs = st.tabs([f"Page {page}" for page in page_numbers])
                for tab, page_num in zip(tabs, page_numbers):
                    with tab:
                        # Capture and display screenshot
                        doc = fitz.open(pdf_path)
                        page = doc.load_page(page_num)
                        pix = page.get_pixmap()
                        screenshot_path = f"screenshot_page_{page_num}.png"
                        pix.save(screenshot_path)
                        doc.close()
                        st.image(screenshot_path)

# Process voice input
if voice_input:
    if "vectors" in st.session_state and st.session_state.vectors is not None:
        response = process_query(voice_input, st.session_state.vectors.get_relevant_documents(voice_input))
        
        process_response(voice_input, response, is_voice=True)
    else:
        assistant_response = (
            "لم يتم تحميل التضميدات. يرجى التحقق مما إذا كان مسار التضميدات صحيحًا." if interface_language == "العربية" else "Embeddings not loaded. Please check if the embeddings path is correct."
        )
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

# Text input field
if interface_language == "العربية":
    human_input = st.chat_input("اكتب سؤالك هنا...")
else:
    human_input = st.chat_input("Type your question here...")

# Process text input
if human_input:
    if "vectors" in st.session_state and st.session_state.vectors is not None:
        response = process_query(human_input, st.session_state.vectors.get_relevant_documents(human_input))
        
        process_response(human_input, response)
    else:
        assistant_response = (
            "لم يتم تحميل التضميدات. يرجى التحقق مما إذا كان مسار التضميدات صحيحًا." if interface_language == "العربية" else "Embeddings not loaded. Please check if the embeddings path is correct."
        )
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

# Clear chat history
