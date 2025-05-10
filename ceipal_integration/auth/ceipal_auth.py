import os
import requests
from datetime import datetime, timedelta
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CeipalAuth:
    """
    Handles authentication with the CEIPAL API.
    Stores and refreshes tokens as needed.
    """

    def __init__(self):
        self.auth_url = "https://api.ceipal.com/v1/createAuthtoken/"
        self.email = os.getenv("CEIPAL_EMAIL")
        self.password = os.getenv("CEIPAL_PASSWORD")
        self.token = None
        self.token_expiry = None

    def authenticate(self) -> bool:
        """Send credentials and API key to retrieve token"""
        payload = {
            "email": self.email,
            "password": self.password,
            "api_key": os.getenv("CEIPAL_API_KEY"),
            "json": "1"
        }

        try:
            response = requests.post(self.auth_url, json=payload)
            response.raise_for_status()
            data = response.json()

            if  "access_token" in data:
                self.token = data.get("access_token")
                # Assuming token lasts 1 hour
                self.token_expiry = datetime.now() + timedelta(hours=1)
                logger.info("Successfully authenticated with CEIPAL.")
                return True
            else:
                logger.error(f"Authentication failed: {data.get('message')}")
                return False

        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False

    def get_token(self) -> str:
        """Returns valid token or re-authenticates"""
        if not self.token or datetime.now() >= self.token_expiry:
            if not self.authenticate():
                return None
        return self.token

