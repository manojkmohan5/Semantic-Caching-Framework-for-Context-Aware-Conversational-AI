class SessionManager:
    def __init__(self):
        self.sessions = {}

    def add_message(self, session_id, message):
        self.sessions.setdefault(session_id, []).append(message)

    def get_context(self, session_id, window=3):
        # Returns last N messages for context, as a single string
        history = self.sessions.get(session_id, [])
        return " ".join(history[-window:])
