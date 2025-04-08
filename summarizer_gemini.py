import os
from dotenv import load_dotenv
import argparse
import requests  # Import requests for API calls

# --- LangChain Imports ---
# LLM Wrapper for Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
# Standard summarization chain
from langchain.chains.summarize import load_summarize_chain
# Document Loader for web pages
from langchain_community.document_loaders import WebBaseLoader
# Text splitter for handling long documents
from langchain_text_splitters import RecursiveCharacterTextSplitter # Corrected import

# --- Configuration ---
# Load environment variables from .env file (specifically GOOGLE_API_KEY)
load_dotenv()

# --- Core Summarization Function ---
def summarize_website(url, chain_type="map_reduce", chunk_size=2000, chunk_overlap=200, model_name=""): # Default to map_reduce also can pass model as empty in code and pass the model in command line as a flag.w
    """
    Fetches content from a URL and generates a summary using LangChain and Google Gemini.

    Args:
        url (str): The URL of the website to summarize.
        chain_type (str): The LangChain summarization strategy ('stuff', 'map_reduce', 'refine').
        chunk_size (int): Size of text chunks for processing (relevant for map_reduce/refine).
        chunk_overlap (int): Overlap between text chunks.
        model_name (str): The Gemini model to use (e.g., "gemini-pro").

    Returns:
        str: The generated summary, or an error message.
    """
    # --- 1. Check for API Key ---
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY not found in environment variables. Please create a .env file."

    try:
        # --- 2. Load Content ---
        print(f"Loading content from: {url}...")
        # WebBaseLoader fetches and parses the HTML content
        loader = WebBaseLoader(url)
        docs = loader.load() # Returns a list of LangChain Document objects

        if not docs:
            return "Error: Could not load any content from the URL."

        # Combine document content for splitting (if needed)
        full_text = "\n".join([doc.page_content for doc in docs])
        if not full_text.strip():
             return "Error: Loaded document content is empty."

        # --- 3. Initialize LLM (Google Gemini) ---
        print(f"Initializing Google Gemini model: {model_name}...")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0, # Lower temperature for more factual summaries
            convert_system_message_to_human=True # Often helpful for compatibility
        )

        # --- 4. Split Text (if using map_reduce or refine) ---
        # The 'stuff' method doesn't require splitting if the content fits the context window.
        # Map_reduce and refine work by processing chunks.
        if chain_type in ["map_reduce", "refine"]:
            print(f"Splitting content into chunks (size={chunk_size}, overlap={chunk_overlap})...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len # Function to measure chunk size
            )
            # Create new Document objects for each chunk
            split_docs = text_splitter.create_documents([full_text])
            print(f"Number of text chunks: {len(split_docs)}")
            # Check if splitting resulted in empty docs (edge case)
            if not split_docs:
                return "Error: Text splitting resulted in no processable chunks."
        else: # For 'stuff' method
             # We still wrap the original docs list
             # The chain expects a list of documents
             split_docs = docs

        # --- 5. Initialize Summarization Chain ---
        # load_summarize_chain handles the logic based on chain_type
        print(f"Initializing summarization chain with type: {chain_type}...")
        # verbose=True can show more details about the chain's execution
        chain = load_summarize_chain(llm, chain_type=chain_type, verbose=False)

        # --- 6. Generate Summary ---
        print("Generating summary...")
        # Use invoke for the standard LangChain interface
        # The input is a dictionary with 'input_documents'
        result = chain.invoke({"input_documents": split_docs}, return_only_outputs=True)

        # The summary is usually in the 'output_text' field of the result
        summary = result.get("output_text", "Error: Could not extract summary from chain result.")

        return summary.strip() # Remove leading/trailing whitespace

    except Exception as e:
        # Catch potential errors during loading, API calls, or processing
        return f"An error occurred: {e}"

def list_available_models(access_token):
    """
    Lists available models and their supported methods using the Google Generative AI API.

    Args:
        access_token (str): The OAuth 2.0 access token.

    Returns:
        str: A formatted string of available models and their methods.
    """
    try:
        # Define the API endpoint for listing models
        url = "https://generativelanguage.googleapis.com/v1beta2/models"
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(url, headers=headers)

        # Check for errors in the response
        if response.status_code != 200:
            return f"Error: Unable to fetch models. HTTP {response.status_code}: {response.text}"

        # Parse the response JSON
        models = response.json().get("models", [])
        if not models:
            return "No models found."

        # Format the models and their supported methods
        return "\n".join([f"{model['name']}: {model.get('supportedMethods', [])}" for model in models])
    except Exception as e:
        return f"An error occurred while listing models: {e}"

# --- Command-Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize content from a website URL using Google Gemini.")
    parser.add_argument("--url", help="The URL of the website to summarize.")  # Change 'url' to optional with --url
    parser.add_argument(
        "--chain_type",
        default="map_reduce",  # map_reduce is robust for varying content lengths
        choices=["stuff", "map_reduce", "refine"],
        help="LangChain summarization chain type to use (default: map_reduce)."
    )
    parser.add_argument(
        "--model",
        default="gemini-pro",
        help="The Google Gemini model name to use (default: gemini-pro)."
    )
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List available models and their supported methods."
    )
    args = parser.parse_args()

    # Handle the case where no arguments are provided
    if not args.url and not args.list_models:
        parser.print_help()  # Show help message if no arguments are provided
    elif args.list_models:
        print("\n--- Available Models ---")
        access_token = os.getenv("GOOGLE_ACCESS_TOKEN")
        if not access_token:
            print("Error: GOOGLE_ACCESS_TOKEN not found in environment variables.")
        else:
            print(list_available_models(access_token))
        print("\n------------------------")
    elif args.url:  # Only require 'url' when not listing models
        print("\n--- Website Summarizer (Google Gemini) ---")
        final_summary = summarize_website(args.url, chain_type=args.chain_type, model_name=args.model)
        print("\n--- Summary ---")
        print(final_summary)
        print("\n-----------------\n")
    else:
        parser.error("Please provide a URL or use --list_models to list available models.")