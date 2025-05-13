import requests
from typing import List
from models.job_posting import JobPosting
from auth.ceipal_auth import CeipalAuth
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CeipalJobPostingsAPI:
    """
    A client to interact with CEIPAL's job postings API.
    """

    def __init__(self, auth: CeipalAuth):
        self.auth = auth
        self.base_url = "https://api.ceipal.com"
        self.job_postings_endpoint = "/getCustomJobPostingDetails/Z3RkUkt2OXZJVld2MjFpOVRSTXoxZz09/e6e04af381e7f42eeb7f942c8bf5ab6d"
        self.job_details_endpoint = "/v1/getJobPostingDetails/"
        

    def parse_job_data(self, data: dict) -> dict:
        """Parse and clean job data before creating JobPosting instance"""
        # Convert empty strings to None for numeric fields
        if data.get('referral_bonus') == '':
            data['referral_bonus'] = None
        if data.get('vms_fee_percentage') == '':
            data['vms_fee_percentage'] = None
            
        # Convert numeric strings to float
        try:
            if data.get('referral_bonus'):
                data['referral_bonus'] = float(data['referral_bonus'])
            if data.get('vms_fee_percentage'):
                data['vms_fee_percentage'] = float(data['vms_fee_percentage'])
        except (ValueError, TypeError):
            logger.warning(f"Could not convert numeric fields for job {data.get('job_code')}")
            
        return data

    def get_job_postings(self, paging_length: int = 60) -> List[JobPosting]:
        """Fetch job postings using the CEIPAL API"""
        token = self.auth.get_token()
        if not token:
            raise Exception("Authentication failed")

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        params = {
            "paging_length": paging_length
        }

        try:
            response = requests.get(
                f"{self.base_url}{self.job_postings_endpoint}",
                headers=headers,
                params=params
            )
            response.raise_for_status()

            job_data = response.json()
            logger.info(f"Received job data: {job_data}")

            job_results = job_data.get("results", [])
            
            # Parse response into list of job postings
            jobs = []
            if isinstance(job_results, list):
                for job in job_results:
                    try:
                        parsed_job = self.parse_job_data(job)
                        jobs.append(JobPosting(**parsed_job))
                    except Exception as e:
                        logger.error(f"Error parsing job {job.get('job_code')}: {str(e)}")
                        continue
            elif isinstance(job_results, dict):
                try:
                    parsed_job = self.parse_job_data(job_results)
                    jobs.append(JobPosting(**parsed_job))
                except Exception as e:
                    logger.error(f"Error parsing job {job_results.get('job_code')}: {str(e)}")
            
            return jobs

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching job postings: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching job postings: {str(e)}")
            raise

    def get_job_details(self, job_code: str):
        """Fetch job details using the CEIPAL API"""
        token = self.auth.get_token()
        if not token:
            raise Exception("Authentication failed")

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        params = {
            "job_id": job_code
        }

        try:
            # Fetch job details using the job_code
            response = requests.get(
                f"{self.base_url}{self.job_details_endpoint}",
                headers=headers,
                params=params
            )
            print(f"{self.base_url}{self.job_details_endpoint}",
                headers,
                params)
            print(f"------------{response.text}")
            response.raise_for_status()

            job_data = response.json()

            print(f"{job_data}")

            # # Parse response into list of job postings
            # if isinstance(job_data, dict):
            #     return JobPosting(**job_data)
            # else:
            #     logger.error(f"Unexpected response format: {type(job_data)}")
            #     return None

        except Exception as e:
            logger.error(f"Error fetching job details: {str(e)}")
            raise
