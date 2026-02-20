class ContentFilterService:

    def __init__(self):
        pass
    
    def is_code(self, text: str) -> bool:
        """
        Returns True if the conversation is natural language only.
        Returns False if it contains code blocks or technical markers.
        """
        forbidden_markers = ["```", "import ", "def ", "class ", "extern ", "iostream"]
        
        text = text.lower()
        for marker in forbidden_markers:
            if marker in text:
                return True
                    
        return False