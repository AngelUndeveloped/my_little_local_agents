import logging
import asyncio
from config import get_settings
from email.gmail_client import GmailClient
from utils.mcp_server import MCPServer
from storage.database import DatabaseManager

# Configure logging
logging.basicConfig(
    level=get_settings().log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmailAgent:
    def __init__(self):
        self.settings = get_settings()
        self.gmail_client = GmailClient()
        self.mcp_server = MCPServer()
        self.db_manager = DatabaseManager()

    async def process_new_emails(self):
        """Process any new emails in the inbox."""
        try:
            # Get new emails
            new_emails = await self.gmail_client.get_new_emails()
            
            for email in new_emails:
                # Enqueue email for processing
                job_id = await self.mcp_server.enqueue_email(email)
                logger.info(f"Enqueued email {email.subject} for processing. Job ID: {job_id}")
                
                # Monitor job status
                while True:
                    status = await self.mcp_server.get_job_status(job_id)
                    if status['status'] == 'finished':
                        # Store results in database
                        await self.db_manager.store_email_processing(email, status['result'])
                        logger.info(f"Processed email: {email.subject}")
                        break
                    elif status['status'] == 'failed':
                        logger.error(f"Failed to process email {email.subject}: {status['error']}")
                        break
                    await asyncio.sleep(5)  # Wait before checking status again
                
        except Exception as e:
            logger.error(f"Error processing emails: {str(e)}")

    async def run(self):
        """Main run loop for the email agent."""
        logger.info("Starting Email Agent...")
        
        # Start background task for cleaning old jobs
        asyncio.create_task(self._cleanup_old_jobs())
        
        while True:
            await self.process_new_emails()
            await asyncio.sleep(self.settings.check_interval)

    async def _cleanup_old_jobs(self):
        """Background task to clean up old jobs."""
        while True:
            try:
                cleared = await self.mcp_server.clear_old_jobs()
                if cleared > 0:
                    logger.info(f"Cleared {cleared} old jobs")
            except Exception as e:
                logger.error(f"Error cleaning up old jobs: {str(e)}")
            await asyncio.sleep(3600)  # Run cleanup every hour


if __name__ == "__main__":
    agent = EmailAgent()
    asyncio.run(agent.run()) 