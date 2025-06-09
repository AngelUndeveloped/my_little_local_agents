from sqlalchemy import create_engine, Column, String, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from config import get_settings
from email.gmail_client import EmailMessage
from typing import Dict, Any


Base = declarative_base()


class ProcessedEmail(Base):
    __tablename__ = 'processed_emails'

    id = Column(String, primary_key=True)
    subject = Column(String)
    sender = Column(String)
    date = Column(DateTime)
    is_important = Column(Boolean)
    importance_explanation = Column(Text)
    summary = Column(Text)
    processed_at = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    def __init__(self):
        self.settings = get_settings()
        self.engine = create_engine(self.settings.database_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    async def store_email_processing(
        self,
        email: EmailMessage,
        processing_result: Dict[str, Any]
    ) -> None:
        """Store email processing results in the database."""
        try:
            session = self.Session()
            
            processed_email = ProcessedEmail(
                id=email.id,
                subject=email.subject,
                sender=email.sender,
                date=datetime.fromisoformat(email.date.replace('Z', '+00:00')),
                is_important=processing_result['is_important'],
                importance_explanation=processing_result['importance_explanation'],
                summary=processing_result['summary']
            )
            
            session.add(processed_email)
            session.commit()
            
        except Exception as e:
            session.rollback()
            raise Exception(f"Error storing email processing: {str(e)}")
            
        finally:
            session.close()

    async def get_processed_emails(
        self,
        limit: int = 10,
        important_only: bool = False
    ) -> list[ProcessedEmail]:
        """Retrieve processed emails from the database."""
        try:
            session = self.Session()
            query = session.query(ProcessedEmail)
            
            if important_only:
                query = query.filter(ProcessedEmail.is_important == True)
                
            return query.order_by(
                ProcessedEmail.processed_at.desc()
            ).limit(limit).all()
            
        except Exception as e:
            raise Exception(f"Error retrieving processed emails: {str(e)}")
            
        finally:
            session.close() 