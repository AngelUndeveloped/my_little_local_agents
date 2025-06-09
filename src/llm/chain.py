from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.llms import Ollama
from typing import Dict, Any
from config import get_settings
from email.gmail_client import EmailMessage


class EmailProcessor:
    def __init__(self):
        self.settings = get_settings()
        self.llm = self._initialize_llm()
        self.importance_chain = self._create_importance_chain()
        self.summary_chain = self._create_summary_chain()

    def _initialize_llm(self):
        """Initialize the appropriate LLM based on configuration."""
        if self.settings.llm_model_type == "gemini":
            return ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=self.settings.llm_api_key,
                temperature=0.7
            )
        else:  # local
            return Ollama(
                base_url=self.settings.local_llm_url,
                model="llama2"  # or your preferred model
            )

    def _create_importance_chain(self) -> LLMChain:
        """Create chain for determining email importance."""
        importance_template = """
        Analyze the following email and determine if it's important.
        Consider factors like:
        - Urgency of the content
        - Sender's importance
        - Action items or deadlines
        - Financial or business impact

        Email Subject: {subject}
        Sender: {sender}
        Content: {content}

        Respond with either "IMPORTANT" or "NOT_IMPORTANT" followed by a brief explanation.
        """

        prompt = PromptTemplate(
            input_variables=["subject", "sender", "content"],
            template=importance_template
        )

        return LLMChain(llm=self.llm, prompt=prompt)

    def _create_summary_chain(self) -> LLMChain:
        """Create chain for generating email summaries."""
        summary_template = """
        Summarize the following email in a concise way, highlighting:
        - Main points
        - Action items
        - Deadlines
        - Important details

        Email Subject: {subject}
        Sender: {sender}
        Content: {content}

        Provide a clear and concise summary.
        """

        prompt = PromptTemplate(
            input_variables=["subject", "sender", "content"],
            template=summary_template
        )

        return LLMChain(llm=self.llm, prompt=prompt)

    async def process_email(self, email: EmailMessage) -> Dict[str, Any]:
        """Process an email to determine importance and generate summary."""
        try:
            # Determine importance
            importance_result = await self.importance_chain.arun(
                subject=email.subject,
                sender=email.sender,
                content=email.body
            )

            # Generate summary
            summary_result = await self.summary_chain.arun(
                subject=email.subject,
                sender=email.sender,
                content=email.body
            )

            return {
                "is_important": "IMPORTANT" in importance_result.upper(),
                "importance_explanation": importance_result,
                "summary": summary_result
            }

        except Exception as e:
            raise Exception(f"Error processing email: {str(e)}") 