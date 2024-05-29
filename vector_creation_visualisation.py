import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession
import vertexai.preview.generative_models as generative_models
import time

# Initialize Vertex AI
vertexai.init(project="miracleinternproj1", location="us-central1")

# Safety settings
safety_settings = {
  generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
  generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
  generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
  generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# Initialize model
model = GenerativeModel(
  "gemini-1.5-flash-preview-0514",
  system_instruction=["""Your detailed system instruction here."""])

def send_message_with_exponential_backoff(session, message):
    base_delay = 1  # initial delay 1 sec
    max_attempts = 5
    for attempt in range(max_attempts):  # 5 retries
        try:
            response = session.send_message(
                content=message,
                generation_config={"max_output_tokens": 8192},
                safety_settings=safety_settings
            )
            return response
        except Exception as e:
            if '429' in str(e):  # rate limit error code
                wait_time = base_delay * (2 ** attempt)
                st.warning(f"Rate limit exceeded, retrying in {wait_time} seconds... {max_attempts-1-attempt} attempts remaining")
                if(attempt == 0):
                    st.warning("Will start a new chat if failed.")
                time.sleep(wait_time)
                continue
            raise  # Reraise other exceptions
        

def main():
    st.title("Miracle Software Systems Chatbot")
    st.caption("Implementation for Streamlit powered chatbot using Vertex AI")

    # Initialize chat or reset chat
    if "chat_session" not in st.session_state or st.button("Start New Chat"):
        st.session_state["chat_session"] = ChatSession(model=model)
        st.session_state["chat_history"] = [{"role": "assistant", "content": "Ask our Miracle bot anything. except the current time!"}]

    chat_session = st.session_state["chat_session"]
    chat_history = st.session_state["chat_history"]

    # Chat loop
    if prompt := st.chat_input("Enter your message:", key="chat_input"):
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        response = send_message_with_exponential_backoff(chat_session, prompt)
        if response is not None:
            st.session_state["chat_history"].append({"role": "assistant", "content": response.text})
        else:
            st.session_state.pop("chat_session", None)  # Optionally clear chat session

    # Display chat history
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if __name__ == "__main__":
    main()
  