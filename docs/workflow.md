# Email Agent Workflow

## System Architecture

```mermaid
graph TB
    subgraph "Email Monitoring"
        Gmail[Gmail API]
        Monitor[Email Monitor]
    end

    subgraph "MCP Server"
        Redis[(Redis Server)]
        Queue[Processing Queue]
        Worker[Worker Process]
    end

    subgraph "Processing Pipeline"
        LLM[LLM Processor]
        Classifier[Importance Classifier]
        Summarizer[Email Summarizer]
    end

    subgraph "Storage"
        DB[(SQLite Database)]
    end

    %% Main Flow
    Gmail -->|New Email| Monitor
    Monitor -->|Enqueue| Queue
    Queue -->|Process| Worker
    Worker -->|Process| LLM
    LLM -->|Classify| Classifier
    LLM -->|Summarize| Summarizer
    Classifier -->|Store Results| DB
    Summarizer -->|Store Results| DB

    %% MCP Server Components
    Redis -->|Manage| Queue
    Redis -->|Track| Worker
    Redis -->|Store| Results

    %% Background Processes
    Monitor -.->|Background| Cleanup[Cleanup Process]
    Cleanup -.->|Remove Old Jobs| Redis

    %% Error Handling
    Worker -.->|Retry Failed| Queue
    Queue -.->|Status Updates| Monitor

    classDef process fill:#f9f,stroke:#333,stroke-width:2px;
    classDef storage fill:#bbf,stroke:#333,stroke-width:2px;
    classDef external fill:#bfb,stroke:#333,stroke-width:2px;
    
    class Gmail,Monitor,Worker,LLM,Classifier,Summarizer process;
    class Redis,DB storage;
    class Gmail external;
```

## Detailed Processing Flow

```mermaid
sequenceDiagram
    participant Gmail as Gmail API
    participant Monitor as Email Monitor
    participant MCP as MCP Server
    participant Worker as Worker Process
    participant LLM as LLM Processor
    participant DB as Database

    loop Every Check Interval
        Monitor->>Gmail: Check for new emails
        Gmail-->>Monitor: Return new emails
        
        loop For each new email
            Monitor->>MCP: Enqueue email
            MCP-->>Monitor: Return job ID
            
            loop Monitor job status
                Monitor->>MCP: Check job status
                MCP-->>Monitor: Return status
                
                alt Job Finished
                    MCP->>Worker: Process email
                    Worker->>LLM: Analyze email
                    LLM-->>Worker: Return analysis
                    Worker->>DB: Store results
                    Monitor->>DB: Update status
                else Job Failed
                    MCP->>MCP: Retry job
                end
            end
        end
    end

    loop Background Cleanup
        Monitor->>MCP: Clean old jobs
        MCP-->>Monitor: Cleanup complete
    end
```

## Component Responsibilities

### Email Monitor
- Polls Gmail API for new emails
- Enqueues emails for processing
- Monitors job status
- Manages background tasks

### MCP Server
- Manages job queues
- Tracks job status
- Handles job retries
- Maintains processing statistics

### Worker Process
- Processes queued emails
- Manages LLM interactions
- Handles error recovery
- Updates job status

### LLM Processor
- Classifies email importance
- Generates email summaries
- Handles model-specific logic

### Database
- Stores processed results
- Maintains email history
- Tracks processing status 