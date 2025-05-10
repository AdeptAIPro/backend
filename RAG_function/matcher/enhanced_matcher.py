import os
import joblib
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from linkedin_api import Linkedin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

from matcher.base_matcher import TalentMatcher
from matcher.models import CandidateProfile
from matcher.llm_service import LLMService
from matcher.utils import dummy_parser

class EnhancedTalentMatcher(TalentMatcher):
    def __init__(self, openai_api_key, linkedin_username, linkedin_password):
        super().__init__(openai_api_key)
        self.llm = LLMService(openai_api_key)
        self.linkedin_api = Linkedin(linkedin_username, linkedin_password)
        self.setup_selenium()
        self.learning_data = []
        self.model_path = "model/matchmaking_model.pkl"
        self.load_or_create_model()

    def setup_selenium(self):
        options = Options()
        options.add_argument("--headless")
        self.driver = webdriver.Chrome(options=options)

    def load_or_create_model(self):
        if os.path.exists(self.model_path):
            self.learning_model = joblib.load(self.model_path)
        else:
            self.learning_model = RandomForestClassifier()
            self.learning_model.fit([[0]*10], [0])

    def extract_job_requirements(self, job_description):
        raw = self.llm(job_description)
        return dummy_parser(raw)

    def validate_linkedin_profile(self, url): return {}
    def validate_nursing_license(self, license_number, state): return {}
    def _calculate_location_match(self, required, actual): return 1.0 if required in actual else 0.0
    def _validate_mandatory_skills(self, required, resume): return sum(skill in resume for skill in required) / len(required)

    def _create_feature_vector(self, job, candidate):
        r = self.extract_job_requirements(job)
        return [
            len(job),
            len(candidate.resume),
            len(r["mandatory_skills"]),
            len(r.get("preferred_skills", [])),
            1 if candidate.linkedin_url else 0,
            1 if candidate.license_number else 0,
            self._calculate_location_match(r["location"], candidate.current_location),
            r.get("years_experience", 0),
            len(r.get("education_requirements", [])),
            len(candidate.resume.split())
        ]

    def _retrain_model(self):
        if len(self.learning_data) < 5: return
        X, y = zip(*self.learning_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.learning_model.fit(X_train, y_train)
        joblib.dump(self.learning_model, self.model_path)

    def _get_learning_adjusted_score(self, job, candidate):
        vec = self._create_feature_vector(job, candidate)
        return self.learning_model.predict([vec])[0] * 100

    def calculate_enhanced_match_score(self, job, candidate):
        req = self.extract_job_requirements(job)
        base = super().calculate_match_score(job, candidate.resume)
        score = {
            **base,
            "linkedin_verification": {},
            "license_verification": {},
            "location_match": self._calculate_location_match(req["location"], candidate.current_location),
            "mandatory_skills_match": self._validate_mandatory_skills(req["mandatory_skills"], candidate.resume),
        }
        self._add_learning_data(job, candidate, score)
        score["learning_adjusted_score"] = self._get_learning_adjusted_score(job, candidate)
        return score

    def _add_learning_data(self, job, candidate, score):
        vec = self._create_feature_vector(job, candidate)
        target = score["overall_match_percentage"] / 100
        self.learning_data.append((vec, target))
        self._retrain_model()

    def find_top_candidates(self, job, candidates, top_n=10):
        scores = [{"candidate": c, "score": self.calculate_enhanced_match_score(job, c)} for c in candidates]
        return sorted(scores, key=lambda x: x["score"]["learning_adjusted_score"], reverse=True)[:top_n]
