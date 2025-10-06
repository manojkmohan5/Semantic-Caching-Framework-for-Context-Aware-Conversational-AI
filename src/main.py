from session_manager import SessionManager
from gemini_client import GeminiClient
from cache import SemanticCache

def main():
    session_mgr = SessionManager()
    gemini = GeminiClient()
    cache = SemanticCache(threshold=0.85)

    session_id = "user1"
    print("Enter your messages to Gemini (type 'exit' to quit):")

    while True:
        user_message = input("\nYou: ")
        if user_message.lower() == 'exit':
            break
        context = session_mgr.get_context(session_id)
        full_query = (context + " " + user_message).strip() if context else user_message

        embedding = gemini.get_embedding(full_query)
        if embedding is None:
            print("Error generating embedding. Try again.")
            continue

        cached_response = cache.search(embedding, session_id)
        if cached_response:
            print("[Cache Hit]\nGemini:", cached_response)
            session_mgr.add_message(session_id, user_message)
            session_mgr.add_message(session_id, cached_response)
        else:
            print("[Cache Miss] Calling Gemini...")
            response = gemini.call_llm(full_query)
            cache.add(embedding, response, session_id, full_query)
            print("Gemini:", response)
            session_mgr.add_message(session_id, user_message)
            session_mgr.add_message(session_id, response)

if __name__ == "__main__":
    main()
