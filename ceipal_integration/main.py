from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
from auth.ceipal_auth import CeipalAuth
from services.ceipal_jobs_api import CeipalJobPostingsAPI
from models.job_posting import JobPosting
from dotenv import load_dotenv
import logging

load_dotenv()

app = FastAPI()

# ✅ Add CORS Middleware with correct frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8081",         # <-- Add your frontend dev server here
        "http://127.0.0.1:8081"          # <-- Also allow this if frontend uses 127.0.0.1
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@app.get("/api/v1/ceipal/jobs")
async def get_ceipal_jobs():
    """
    Fetch job postings from CEIPAL API.
    """
    auth = CeipalAuth()
    api = CeipalJobPostingsAPI(auth)

    try:
        jobs = api.get_job_postings()
        return {"jobs": jobs}
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP Status Error: {e}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Error fetching jobs from Ceipal: {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.get("/api/v1/ceipal/getJobDetails")
async def get_job_details(job_code: str):
    """
    Get job details by job_code from CEIPAL.
    """
    auth = CeipalAuth()
    api = CeipalJobPostingsAPI(auth)

    try:
        job_details = api.get_job_details(job_code)
        return {"job_details": job_details}
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP Status Error: {e}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Error fetching job details from Ceipal: {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )



# @app.get("")
# async def get_job_details(: str):
#     """
#     Get job details by job_code from CEIPAL.
#     """
#     auth = CeipalAuth()
#     api = CeipalJobPostingsAPI(auth)

#     try:
#          = api.get_()
#         return {": }
#     except httpx.HTTPStatusError as e:
#         logger.error(f"HTTP Status Error: {e}")
#         raise HTTPException(
#             status_code=e.response.status_code,
#             detail=f"Error fetching job details from Ceipal: {e.response.text}"
#         )
#     except Exception as e:
#         logger.error(f"Unexpected error: {e}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Unexpected error: {str(e)}"
#         )




# from fastapi import FastAPI, HTTPException, logger
# import httpx
# from auth.ceipal_auth import CeipalAuth
# from services.ceipal_jobs_api import CeipalJobPostingsAPI
# from models.job_posting import JobPosting
# from dotenv import load_dotenv

# load_dotenv()

# app = FastAPI()

# @app.get("/api/v1/ceipal/jobs")
# async def get_ceipal_jobs():
#     """
#     Fetch job postings from CEIPAL API.
#     """
#     # Initialize auth with your credentials
#     auth = CeipalAuth()

#     # Create the API client
#     api = CeipalJobPostingsAPI(auth)

#     try:
#         # Fetch job postings
#         jobs = api.get_job_postings()
#         return {"jobs": jobs}
#     except httpx.HTTPStatusError as e:
#         logger.error(f"HTTP Status Error: {e}")
#         raise HTTPException(
#             status_code=e.response.status_code,
#             detail=f"Error fetching jobs from Ceipal: {e.response.text}"
#         )
#     except Exception as e:
#         logger.error(f"Unexpected error: {e}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Unexpected error: {str(e)}"
#         )
    

# @app.get("/api/v1/ceipal/getJobDetails")
# async def get_job_details(job_code: str):
#     try:
#         # Initialize auth with your credentials
#         auth = CeipalAuth()

#         # Create the API client
#         api = CeipalJobPostingsAPI(auth)

#         # Fetch job details
#         job_details = api.get_job_details(job_code)
#         return {"job_details": job_details}
#     except httpx.HTTPStatusError as e:
#         logger.error(f"HTTP Status Error: {e}")
#         raise HTTPException(
#             status_code=e.response.status_code,
#             detail=f"Error fetching job details from Ceipal: {e.response.text}"
#         )




# # @app.get("/api/v1/ceipal/job_code")
# # async def get_job_code():
# #     """
# #     Fetch job postings from CEIPAL API.
# #     """
# #     # Initialize auth with your credentials
# #     auth = CeipalAuth()

# #     # Create the API client
# #     api = CeipalJobPostingsAPI(auth)

# #     try:
# #         # Fetch job postings
# #         job_code = api.get_job_code()
# #         return {"job_code": job_code}
# #     except httpx.HTTPStatusError as e:
# #         logger.error(f"HTTP Status Error: {e}")
# #         raise HTTPException(
# #             status_code=e.response.status_code,
# #             detail=f"Error fetching jobs from Ceipal: {e.response.text}"
# #         )
# #     except Exception as e:
# #         logger.error(f"Unexpected error: {e}")
# #         raise HTTPException(
# #             status_code=500,
# #             detail=f"Unexpected error: {str(e)}"
# #         )





# # if __name__ == "__main__":
#     # Initialize auth with your credentials
#     auth = CeipalAuth()

#     # Create the API client
#     api = CeipalJobPostingsAPI(auth)

#     try:
#         # Fetch job postings
#         jobs = api.get_job_postings()
#         print(f"Retrieved {len(jobs)} job postings")

#         # Display job info
#         for job in jobs:
#             print(f"Job Code: {job.job_code}")
#             print(f"Title: {job.job_title}")
#             print(f"bill_rate:{job.bill_rate}")
#             print(f"pay_rate:{job.pay_rate}")
#             print(f"job_start_date:{job.job_start_date}")
#             print(f"job_end_date:{job.job_end_date}")
#             print(f"remote_job:{job.remote_job}")
#             print(f"country:{job.country}")
#             print(f"states:{job.states}")
#             print(f"city:{job.city}")
#             print(f"zip_code:{job.zip_code}")
#             print(f"job_status:{job.job_status}")
#             print(f"job_type:{job.job_type}")
#             print(f"client:{job. client}")
#             print(f"client_manager:{job.client_manager}")
#             print(f"end_client:{job.end_client}")
#             print(f"client_job_id:{job.client_job_id}")
#             print(f" turnaround_time:{job. turnaround_time}")
#             print(f"required_skill_checklist_for_submission:{job.required_skill_checklist_for_submission}")
#             print(f"priority:{job.priority}")
#             print(f"duration:{job.duration}")
#             print(f"work_authorization:{job.work_authorization}")
#             print(f"ceipal_ref__:{job.ceipal_ref__}")
#             print(f"application_form:{job.application_form}")
#             print(f"clearance:{job.clearance}")   
#             print(f"address:{job.address}")
#             print(f"degree:{job.degree}")
#             print(f"experience:{job.experience}")
#             print(f"evaluation_template:{job.evaluation_template}")
#             print(f"skills:{job.skills}")
#             print(f"languages:{job.languages}")  
#             print(f"number_of_positions:{job.number_of_positions}")
#             print(f"maximum_allowed_submissions:{job.maximum_allowed_submissions}")
#             print(f"tax_terms:{job.tax_terms}")
#             print(f"sales_manager:{job.sales_manager}")
#             print(f"department:{job.department}")     
#             print(f"recruitment_manager:{job.recruitment_manager}")
#             print(f"account_manager:{job.account_manager}")
#             print(f"assigned_to:{job.assigned_to}")
#             print(f"primary_recruiter:{job.primary_recruiter}")
#             print(f"comments:{job.comments}")
#             print(f"additional_notifications:{job.additional_notifications}")
#             print(f"career_portal_published_date:{job.career_portal_published_date}")
#             print(f"job_description:{job.job_description}")
#             print(f"post_job_on_career_portal:{job.post_job_on_career_portal}")
#             print(f"display_contact_details_on_career_portal:{job.display_contact_details_on_career_portal}")
#             print(f"location:{job.location}")
#             print(f"notice_period:{job.notice_period}")
#             print(f"referral_bonus:{job.referral_bonus}")
#             print(f"job_category:{job.job_category}")
#             print(f"public_job_description:{job.public_job_description}")
#             print(f"offerings:{job.offerings}")   
#             print(f"profession:{job.profession}")
#             print(f"speciality:{job.speciality}")
#             print(f"required_certifications_for_submission:{job.required_certifications_for_submission}")
#             print(f"onboarding_owner:{job.onboarding_owner}")
#             print(f"required_license_for_submission:{job.required_license_for_submission}")
#             print(f"no__of_beds:{job.no__of_beds}")
#             print(f"required_documents_for_eboarding:{job.required_documents_for_eboarding}")         
#             print(f"required_certifications_for_eboarding:{job.required_certifications_for_eboarding}")
#             print(f"required_license_for_eboarding:{job.required_license_for_eboarding}")
#             print(f"vms_fee_percentage:{job.vms_fee_percentage}")
#             print(f"required_skill_checklist_for_submission:{job.required_skill_checklist_for_submission}")
#             # Add location if it exists in your model
#             # print(f"Location: {job.location}")
#             print("-" * 40)

#     except Exception as e:
#         print(f"Error: {str(e)}")
