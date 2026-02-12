"""
AgentSwarm Processing Module - System przetwarzania plików.

Moduł zapewnia wydajne przetwarzanie dużych zbiorów plików z obsługą:
- Async/await dla operacji I/O
- Batch processing z konfigurowalnym rozmiarem
- Memory-efficient streaming dla dużych plików
- Checkpointi i wznawianie przerwanych zadań
- Śledzenie postępu i raportowanie

Przykład użycia:
    ```python
    from agentswarm.processing import FileProcessor
    
    processor = FileProcessor(
        checkpoint_dir="./checkpoints",
        max_concurrent=10,
        batch_size=100
    )
    
    results = await processor.process_directory(
        "/path/to/files",
        patterns=["*.py", "*.md"],
        processor=my_processing_function
    )
    ```
"""

__version__ = "1.0.0"
__author__ = "AgentSwarm Team"

# Główne klasy
from .file_scanner import (
    FileScanner,
    FileInfo,
    ScanConfig,
    MultiRootScanner,
    EncodingDetector,
)

from .content_extractor import (
    ContentExtractor,
    ExtractedContent,
    TextExtractor,
    CodeExtractor,
    JSONExtractor,
    CSVExtractor,
    XMLExtractor,
    YAMLExtractor,
    PDFExtractor,
    DocxExtractor,
    ExtractionError,
)

from .checkpoint_manager import (
    CheckpointManager,
    JobCheckpoint,
    FileCheckpoint,
    JobStatus,
)

from .progress_tracker import (
    ProgressTracker,
    ProgressStats,
    FileProgress,
    ProgressEvent,
    ConsoleProgressReporter,
    FileProgressLogger,
)

from .batch_processor import (
    BatchProcessor,
    BatchConfig,
    BatchResult,
    ProcessingResult,
    RetryManager,
    create_default_processor,
    retry_on_error,
    rate_limited,
)

from .orchestrator import (
    FileProcessor,
    ProcessingConfig,
    ProcessingSummary,
    ProcessingPipeline,
    PipelineStage,
    quick_process,
    process_with_checkpoints,
    resume_processing,
)

__all__ = [
    # File Scanner
    'FileScanner',
    'FileInfo',
    'ScanConfig',
    'MultiRootScanner',
    'EncodingDetector',
    
    # Content Extractor
    'ContentExtractor',
    'ExtractedContent',
    'TextExtractor',
    'CodeExtractor',
    'JSONExtractor',
    'CSVExtractor',
    'XMLExtractor',
    'YAMLExtractor',
    'PDFExtractor',
    'DocxExtractor',
    'ExtractionError',
    
    # Checkpoint Manager
    'CheckpointManager',
    'JobCheckpoint',
    'FileCheckpoint',
    'JobStatus',
    
    # Progress Tracker
    'ProgressTracker',
    'ProgressStats',
    'FileProgress',
    'ProgressEvent',
    'ConsoleProgressReporter',
    'FileProgressLogger',
    
    # Batch Processor
    'BatchProcessor',
    'BatchConfig',
    'BatchResult',
    'ProcessingResult',
    'RetryManager',
    'create_default_processor',
    'retry_on_error',
    'rate_limited',
    
    # Orchestrator
    'FileProcessor',
    'ProcessingConfig',
    'ProcessingSummary',
    'ProcessingPipeline',
    'PipelineStage',
    'quick_process',
    'process_with_checkpoints',
    'resume_processing',
]
