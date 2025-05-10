class LLMService:
    def __init__(self, api_key):
        self.api_key = api_key

    def __call__(self, prompt: str) -> str:
        # Simulate LLM response for testing
        return """
        {
            "mandatory_skills": ["Python", "AWS"],
            "preferred_skills": ["Healthcare"],
            "location": "Boston, MA",
            "years_experience": 5,
            "education_requirements": ["Bachelor's"]
        }
        """
