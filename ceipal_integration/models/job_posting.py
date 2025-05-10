from pydantic import BaseModel, Field
from typing import Optional

# class JobPosting(BaseModel):



class JobPosting(BaseModel):
    # title: str
    # other required fields
    location: str
    # ... your other fields ...

    # ✅ Make these optional to fix the error
    # referral_bonus: Optional[float] = None
    # vms_fee_percentage: Optional[float] = None
    # title: Optional[str] = None
    # referral_bonus: Optional[float] = None
    # vms_fee_percentage: Optional[float] = None
    # description: Optional[str]
    """
    Represents a single CEIPAL Job Posting.
    Automatically parses and validates fields returned by the API.
    """
    job_code: str = Field(..., alias="job_code")
    job_title: str = Field(..., alias="job_title")
    bill_rate: Optional[str] = Field(None, alias="bill_rate")
    pay_rate: Optional[str] = Field(None, alias="pay_rate")
    job_start_date: Optional[str] = Field(None, alias="job_start_date")
    job_end_date: Optional[str] = Field(None, alias="job_end_date")
    remote_job: Optional[str] = Field(None, alias="remote_job")
    country: Optional[str] = Field(None, alias="country")
    states: Optional[str] = Field(None, alias="states")
    city: Optional[str] = Field(None, alias="city")
    zip_code: Optional[str] = Field(None, alias="zip_code")
    job_status: Optional[str] = Field(None, alias="job_status")
    job_type: Optional[str] = Field(None, alias="job_type")
    client: Optional[str] = Field(None, alias="client")
    client_manager: Optional[str] = Field(None, alias="client_manager")
    end_client: Optional[str] = Field(None, alias="end_client")
    client_job_id: Optional[str] = Field(None, alias="client_job_id")
    turnaround_time: Optional[str] = Field(None, alias="turnaround_time")
    # ... (full list is unchanged, just shown partially here)
    required_skill_checklist_for_submission: Optional[str] = Field(None, alias="required_skill_checklist_for_submission")
    priority: Optional[str] = Field(None, alias="priority")
    duration: Optional[str] = Field(None, alias="duration")
    work_authorization: Optional[str] = Field(None, alias="work_authorization")
    ceipal_ref__: Optional[str] = Field(None, alias="ceipal_ref__")
    application_form: Optional[str] = Field(None, alias="application_form")
    clearance: Optional[str] = Field(None, alias="clearance")
    address: Optional[str] = Field(None, alias="address")
    degree: Optional[str] = Field(None, alias="degree")
    experience: Optional[str] = Field(None, alias="experience")
    evaluation_template: Optional[str] = Field(None, alias="evaluation_template")
    skills: Optional[str] = Field(None, alias="skills")
    languages: Optional[str] = Field(None, alias="languages")
    number_of_positions: Optional[int] = Field(None, alias="number_of_positions")
    maximum_allowed_submissions: Optional[str] = Field(None, alias="maximum_allowed_submissions")
    tax_terms: Optional[str] = Field(None, alias="tax_terms")
    sales_manager: Optional[str] = Field(None, alias="sales_manager")
    department: Optional[str] = Field(None, alias="department")
    recruitment_manager: Optional[str] = Field(None, alias="recruitment_manager")
    account_manager: Optional[str] = Field(None, alias="account_manager")
    assigned_to: Optional[str] = Field(None, alias="assigned_to")
    primary_recruiter: Optional[str] = Field(None, alias="primary_recruiter")
    comments: Optional[str] = Field(None, alias="comments")
    additional_notifications: Optional[str] = Field(None, alias="additional_notifications")
    career_portal_published_date: Optional[str] = Field(None, alias="career_portal_published_date")
    job_description: Optional[str] = Field(None, alias="job_description")
    post_job_on_career_portal: Optional[str] = Field(None, alias="post_job_on_career_portal")
    display_contact_details_on_career_portal: Optional[str] = Field(None, alias="display_contact_details_on_career_portal")
    location: Optional[str] = Field(None, alias="location")
    notice_period: Optional[str] = Field(None, alias="notice_period")
    referral_bonus: Optional[float] = Field(None, alias="referral_bonus")
    job_category: Optional[str] = Field(None, alias="job_category")
    public_job_description: Optional[str] = Field(None, alias="public_job_description")
    offerings: Optional[str] = Field(None, alias="offerings")
    profession: Optional[str] = Field(None, alias="profession")
    speciality: Optional[str] = Field(None, alias="speciality")
    required_certifications_for_submission: Optional[str] = Field(None, alias="required_certifications_for_submission")
    onboarding_owner: Optional[str] = Field(None, alias="onboarding_owner")
    required_license_for_submission: Optional[str] = Field(None, alias="required_license_for_submission")
    no__of_beds: Optional[int] = Field(None, alias="no__of_beds")
    required_documents_for_eboarding: Optional[str] = Field(None, alias="required_documents_for_eboarding")
    required_certifications_for_eboarding: Optional[str] = Field(None, alias="required_certifications_for_eboarding")
    required_license_for_eboarding: Optional[str] = Field(None, alias="required_license_for_eboarding")
    vms_fee_percentage: Optional[float] = Field(None, alias="vms_fee_percentage")
    required_skill_checklist_for_submission: Optional[str] = Field(None, alias="required_skill_checklist_for_submission")

    class Config:
        allow_population_by_field_name = True
        extra = "ignore"  # Ignore extra fields from API response
   