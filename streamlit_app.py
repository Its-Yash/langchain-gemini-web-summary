import streamlit as st
from summarizer_gemini import summarize_website

# Streamlit UI
st.title("Website Summarizer")
st.write("Summarize content from a website URL using Google Gemini.")

# Input fields
url = st.text_input("Enter the website URL:")
chain_type = st.selectbox(
    "Select summarization chain type:",
    ["stuff", "map_reduce", "refine"],
    index=1  # Default to "map_reduce"
)
model_name = st.text_input("Enter the Gemini model name (default: gemini-pro):", value="gemini-pro")

# Button to generate summary
if st.button("Generate Summary"):
    if not url.strip():
        st.error("Please enter a valid URL.")
    else:
        with st.spinner("Generating summary..."):
            summary = summarize_website(url, chain_type=chain_type, model_name=model_name)
        st.subheader("Summary:")
        st.write(summary)
