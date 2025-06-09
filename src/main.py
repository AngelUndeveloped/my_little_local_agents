import logging
from config import get_settings
from email.gmail_client import GmailClient
from llm.chain import EmailProcessor
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
        self.email_processor = EmailProcessor()
        self.db_manager = DatabaseManager()

    async def process_new_emails(self):
        """Process any new emails in the inbox."""
        try:
            # Get new emails
            new_emails = await self.gmail_client.get_new_emails()
            
            for email in new_emails:
                # Process email with LLM
                summary = await self.email_processor.process_email(email)
                
                # Store results
                await self.db_manager.store_email_processing(email, summary)
                
                logger.info(f"Processed email: {email.subject}")
                
        except Exception as e:
            logger.error(f"Error processing emails: {str(e)}")

    async def run(self):
        """Main run loop for the email agent."""
        logger.info("Starting Email Agent...")
        while True:
            await self.process_new_emails()
            await asyncio.sleep(self.settings.check_interval)


if __name__ == "__main__":
    import asyncio
    agent = EmailAgent()
    asyncio.run(agent.run()) 