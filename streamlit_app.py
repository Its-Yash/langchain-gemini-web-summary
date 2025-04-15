import streamlit as st
import os
from dotenv import load_dotenv

# Import the updated agent execution function
from summarizer_agent import run_agent

# --- Configuration ---
load_dotenv()

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📰 Enhanced Website Summarizer Agent")
st.write("Enter a website URL and optionally specify the desired summary format (e.g., 'in bullet points', 'as 3 key sentences'). The agent will fetch, analyze, summarize, and format the content.")

# Check for API Key early
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("🔴 Error: GOOGLE_API_KEY not found in environment variables. Please create a `.env` file with the key.")
    st.stop() # Stop execution if key is missing

# Input fields
user_input = st.text_input(
    "Enter URL and optional format request:",
    placeholder="e.g., https://example.com summarize in bullet points"
)

# Model Selection (Agent uses this model)
model_name = st.selectbox(
    "Select the Gemini model for the Agent:",
    # Ensure these models support function/tool calling well
    ["gemini-1.5-flash", "gemini-pro", "gemini-1.5-pro-latest"],
     index=0,  # Default to Flash for speed
    help="The model used for agent reasoning and summarization/formatting steps."
)

# REMOVED chain_type selector - Agent decides this now

# Button to trigger agent
if st.button("✨ Generate Summary with Agent"):
    if not user_input or not user_input.strip():
        st.error("❌ Please enter a URL and optional instructions.")
    # Basic check for something that looks like a URL
    elif "http://" not in user_input.lower() and "https://" not in user_input.lower():
         st.error("❌ Please include a valid URL (starting with http:// or https://) in your request.")
    else:
        # Use columns for better layout during processing
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info(f"📝 Request: {user_input[:80]}...") # Show truncated request
            st.info(f"🤖 Agent Model: {model_name}")

        with col2:
            with st.spinner(f"🚀 Agent is working on your request... This involves fetching, analyzing, summarizing, and formatting. Please wait."):
                # Call the agent execution function with the full user input
                summary_result = run_agent(user_input=user_input, model_name=model_name)

            st.subheader("📝 Summary Result:")
            # Display the result (formatted summary or an error message)
            st.markdown(summary_result) # Use markdown for potential formatting