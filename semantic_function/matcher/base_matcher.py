class TalentMatcher:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key

    def calculate_match_score(self, job_description: str, resume: str):
        # Dummy implementation for base score
        return {
            "overall_match_percentage": 60.0  # Replace with real logic
        }
