# Semantic Caching Framework for Conversational AI

A context-aware semantic cache for conversational AI. Uses embeddings to find similar previous queries and serves cached responses, minimizing redundant calls to the Gemini API.

---

## Requirements

- Python 3.8+
- Google Gemini API key

---

## # Execution Steps

### # 1. Clone the repository

git clone https://github.com/manojkmohan5/Semantic-Caching-Framework-for-Context-Aware-Conversational-AI.git


cd Semantic-Caching-Framework-for-Context-Aware-Conversational-AI



### # 2. Create and activate a virtual environment

**Windows:**
python -m venv venv
venv\Scripts\activate

**Mac/Linux:**
python3 -m venv venv
source venv/bin/activate


### # 3. Install dependencies

pip install -r requirements.txt


### # 4. Set up your environment variables

1. Copy `.env.example` to `.env`
2. Add your Gemini API key:
    ```
    GEMINI_API_KEY=your_actual_key_here
    ```

### # 5. Run the application

Make sure your virtual environment is activated, then run:
python src/main.py


### # 6. Start chatting

- When prompted, enter any session ID (ex: your name)
- The app will display `[Cache Hit]` or `[Cache Miss]` depending on whether your question has a similar cached response.

---

## # Example Interaction

You: What is the impact of climate change on corn yields?
[Cache Miss] Calling Gemini...
Gemini: (detailed answer...)

You: What about wheat?
[Cache Miss] Calling Gemini...
Gemini: (answer...)

You: How does climate change affect wheat?
[Cache Hit]
Gemini: (cached wheat answer...)


---

## # Environment Variables

Your `.env` file should look like:
GEMINI_API_KEY=your_actual_api_key

**Never commit your real API key. Use `.env.example` for sharing.**

---

## # Dependencies

See `requirements.txt`:
- requests
- python-dotenv
- numpy
- faiss-cpu
- tqdm
- pandas

---
