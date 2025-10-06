# Semantic Caching Framework for Context-Aware Conversational AI

A conversational AI cache that reduces redundant LLM calls by detecting semantically similar user queries across multi-turn chats using context-aware embeddings.

# How It Works
The system combines recent conversation history with a user's current query to generate an embedding that captures semantic meaning. When a similar question is asked (even if phrased differently), it returns a cached LLM response instead of calling the Gemini API again.

# Setup
Requirements: Python 3.8+, Google Gemini API key

Clone this repository and navigate to the project folder.

# Create a virtual environment:

bash/terminal:
python -m venv venv

# For Windows:
venv\Scripts\activate

# For Mac/Linux:
source venv/bin/activate
Install dependencies:

bash/teminal:
pip install -r requirements.txt
Create a .env file in the project root like this:

# text:
GEMINI_API_KEY=your_api_key_here
Usage
Run the main script:

bash/terminal:
python src/main.py
You'll be prompted for a session ID (can use any name), and then you can chat. The system will show [Cache Hit] or [Cache Miss] for each query depending on whether a similar answer is already cached.

# Example:

text
You: What is the impact of climate change on corn yields?
[Cache Miss] Calling Gemini...
Gemini: (detailed response...)

You: What about wheat?
[Cache Miss] Calling Gemini...
Gemini: (response...)

You: How does climate change affect wheat?
[Cache Hit]
Gemini: (cached wheat response...)
Environment Variables
Your .env file should contain:

text
GEMINI_API_KEY=your_actual_api_key
Get your free API key from Google AI Studio.

# Design Notes
Uses the last 3 conversation turns plus the current query to build embeddings.

Cache threshold is set at 0.85 cosine similarity (can be changed in cache.py).

Each session ID keeps conversations and caches separate.

Embeddings: Google's text-embedding-004 model.

LLM: Google's gemini-2.5-pro model.

# Dependencies
Listed in requirements.txt:

requests

python-dotenv

numpy

faiss-cpu

tqdm

pandas
