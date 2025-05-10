from models.job_posting import JobPosting

def filter_job_postings(job_postings: list[JobPosting], min_bill_rate: str) -> list[JobPosting]:
    """Filter job postings based on a minimum bill rate."""
    return [job for job in job_postings if job.bill_rate and float(job.bill_rate) >= float(min_bill_rate)]
