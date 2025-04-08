# Website Summarizer with Google Gemini

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project provides a tool to summarize the content of websites using the power of Google's Gemini AI models. It offers both a command-line interface (CLI) for quick summarizations and a Streamlit web view for a more interactive user experience.

## Features

* **URL Input:** Summarize content directly from website URLs.
* **Multiple Summarization Strategies:** Supports different LangChain summarization chain types:
    * `stuff`: Simple approach where all the text is passed to the LLM at once.
    * `map_reduce`: Processes the text in chunks and then combines the summaries.
    * `refine`: Iteratively refines the summary using sequential passes over the document.
* **Gemini Model Selection:** Allows you to choose between different Google Gemini models for summarization (currently supporting `gemini-pro` and `gemini-1.5-flash`).
* **CLI Interface:** A simple command-line tool for quick and easy summarizations.
* **Streamlit Web View:** An interactive web interface built with Streamlit for a more user-friendly experience.
* **Configurable Parameters:** Options to adjust chunk size and overlap for the `map_reduce` and `refine` chain types.

## Getting Started

Follow these steps to get the summarizer up and running on your local machine.

### Prerequisites

* **Python 3.9+**
* **pip** (Python package installer)
* A **Google Cloud Project** with the **Generative Language API** enabled.
* A **Google API Key** for accessing the Gemini models.

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Its-Yash/langchain-gemini-web-summary.git
    cd website-summarizer
    ```
   
2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS and Linux
    # venv\Scripts\activate  # On Windows
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Google API Key:**
    * Create a `.env` file in the root of your project directory.
    * Add your Google API key to the `.env` file:
        ```
        GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
        ```
        *(Replace `YOUR_GOOGLE_API_KEY` with your actual API key from Google Cloud Console)*

### Usage

The project offers both a command-line interface and a Streamlit web view.

#### Command-Line Interface (CLI)

To use the CLI, run the `summarizer_gemini.py` script with the following arguments:

```bash
python summarizer_gemini.py --url <WEBSITE_URL> [--chain_type <TYPE>] [--model <MODEL_NAME>]
