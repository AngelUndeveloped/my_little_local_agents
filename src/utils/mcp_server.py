from redis import Redis
from rq import Queue, Worker
from rq.job import Job
from typing import Dict, Any, Optional
import json
from config import get_settings
from email.gmail_client import EmailMessage


class MCPServer:
    def __init__(self):
        self.settings = get_settings()
        self.redis_conn = Redis(
            host=self.settings.redis_host,
            port=self.settings.redis_port,
            password=self.settings.redis_password,
            decode_responses=True
        )
        self.queue = Queue('email_processing', connection=self.redis_conn)
        self.processing_queue = Queue('processing', connection=self.redis_conn)
        self.results_queue = Queue('results', connection=self.redis_conn)

    async def enqueue_email(self, email: EmailMessage) -> str:
        """Add an email to the processing queue."""
        job = self.queue.enqueue(
            'src.llm.chain.EmailProcessor.process_email',
            args=(email,),
            job_timeout='10m',
            result_ttl=86400  # Keep results for 24 hours
        )
        return job.id

    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a processing job."""
        job = Job.fetch(job_id, connection=self.redis_conn)
        return {
            'id': job.id,
            'status': job.get_status(),
            'result': job.result if job.result else None,
            'error': str(job.exc_info) if job.exc_info else None
        }

    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about the processing queue."""
        return {
            'queued': self.queue.count,
            'processing': self.processing_queue.count,
            'completed': self.results_queue.count,
            'failed': len(self.queue.failed_job_registry)
        }

    async def retry_failed_job(self, job_id: str) -> bool:
        """Retry a failed job."""
        try:
            job = Job.fetch(job_id, connection=self.redis_conn)
            if job.is_failed:
                job.requeue()
                return True
            return False
        except Exception:
            return False

    async def clear_old_jobs(self, days: int = 7) -> int:
        """Clear old completed jobs from the queue."""
        try:
            count = 0
            for job in self.results_queue.jobs:
                if job.ended_at and (job.ended_at - job.started_at).days > days:
                    job.delete()
                    count += 1
            return count
        except Exception:
            return 0 