from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import base64
import email
from typing import List, Optional
from dataclasses import dataclass
from config import get_settings


@dataclass
class EmailMessage:
    id: str
    subject: str
    sender: str
    body: str
    date: str
    is_important: bool = False


class GmailClient:
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

    def __init__(self):
        self.settings = get_settings()
        self.service = self._get_gmail_service()
        self.last_check_id = None

    def _get_gmail_service(self):
        """Initialize and return Gmail API service."""
        creds = Credentials(
            None,
            refresh_token=self.settings.gmail_refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=self.settings.gmail_client_id,
            client_secret=self.settings.gmail_client_secret,
            scopes=self.SCOPES
        )

        return build('gmail', 'v1', credentials=creds)

    async def get_new_emails(self) -> List[EmailMessage]:
        """Fetch new emails since last check."""
        try:
            # Get list of messages
            results = self.service.users().messages().list(
                userId='me',
                q=f'after:{self.last_check_id}' if self.last_check_id else None
            ).execute()

            messages = results.get('messages', [])
            if not messages:
                return []

            # Update last check ID
            self.last_check_id = messages[0]['id']

            # Process each message
            email_messages = []
            for message in messages:
                msg = self.service.users().messages().get(
                    userId='me',
                    id=message['id'],
                    format='full'
                ).execute()

                email_data = self._parse_email_message(msg)
                if email_data:
                    email_messages.append(email_data)

            return email_messages

        except Exception as e:
            raise Exception(f"Error fetching emails: {str(e)}")

    def _parse_email_message(self, message) -> Optional[EmailMessage]:
        """Parse Gmail message into EmailMessage object."""
        try:
            headers = message['payload']['headers']
            subject = next(h['value'] for h in headers if h['name'] == 'Subject')
            sender = next(h['value'] for h in headers if h['name'] == 'From')
            date = next(h['value'] for h in headers if h['name'] == 'Date')

            # Get message body
            if 'parts' in message['payload']:
                body = self._get_message_body(message['payload']['parts'])
            else:
                body = base64.urlsafe_b64decode(
                    message['payload']['body']['data']
                ).decode('utf-8')

            return EmailMessage(
                id=message['id'],
                subject=subject,
                sender=sender,
                body=body,
                date=date
            )

        except Exception as e:
            print(f"Error parsing email message: {str(e)}")
            return None

    def _get_message_body(self, parts) -> str:
        """Extract message body from email parts."""
        body = ""
        for part in parts:
            if part['mimeType'] == 'text/plain':
                body += base64.urlsafe_b64decode(
                    part['body']['data']
                ).decode('utf-8')
            elif 'parts' in part:
                body += self._get_message_body(part['parts'])
        return body 