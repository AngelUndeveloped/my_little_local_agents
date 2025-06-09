# Email Agent with LLM Integration

An intelligent email agent that monitors your Gmail account, processes incoming emails, and generates summaries for important messages using LangChain and LLM technology.

## Features

- Real-time Gmail monitoring
- Intelligent email importance classification
- Automated email summarization
- Support for both cloud (Gemini) and local LLM models
- Containerized deployment ready
- Secure credential management

## Prerequisites

- Python 3.8+
- Docker (for containerization)
- Gmail account with API access
- LLM API key (if using Gemini) or local LLM setup (LMStudio/Ollama)

## Project Structure

```
email_agent/
├── Dockerfile
├── requirements.txt
├── .env
├── src/
│   ├── main.py
│   ├── config.py
│   ├── email/
│   │   ├── gmail_client.py
│   │   └── processor.py
│   ├── llm/
│   │   ├── chain.py
│   │   └── prompts.py
│   ├── storage/
│   │   └── database.py
│   └── utils/
│       ├── logger.py
│       └── helpers.py
└── tests/
```

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd email_agent
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with the following variables:
```
GMAIL_CLIENT_ID=your_client_id
GMAIL_CLIENT_SECRET=your_client_secret
GMAIL_REFRESH_TOKEN=your_refresh_token
LLM_API_KEY=your_llm_api_key  # If using Gemini
```

5. Configure Gmail API:
- Go to Google Cloud Console
- Create a new project
- Enable Gmail API
- Create OAuth 2.0 credentials
- Download and save the credentials

## Usage

### Running Locally

```bash
python src/main.py
```

### Running with Docker

```bash
docker build -t email-agent .
docker run -d --env-file .env email-agent
```

## Configuration

The agent can be configured to use either:
- Gemini API (cloud-based)
- Local LLM models (via LMStudio or Ollama)

To switch between models, modify the `config.py` file.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

This project follows PEP 8 guidelines. Use the following tools for code quality:
- black for formatting
- flake8 for linting
- mypy for type checking

## Deployment

The application is designed to be deployed on Hostinger or any other hosting platform that supports Docker containers.

### Deployment Steps

1. Build the Docker image
2. Push to your container registry
3. Deploy on Hostinger using their Docker support

## Security

- All credentials are stored in environment variables
- Gmail API tokens are securely managed
- Rate limiting is implemented to prevent API abuse
- Regular security audits are recommended

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain for LLM orchestration
- Google Gmail API
- Gemini/Local LLM providers 