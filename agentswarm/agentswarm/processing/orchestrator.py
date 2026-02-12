"""
Orchestrator - Główny koordynator przetwarzania plików.

Łączy wszystkie komponenty w spójny interfejs do przetwarzania plików.
"""

import asyncio
from pathlib import Path
from typing import (
    List, Callable, Optional, Dict, Any, AsyncIterator, 
    Union, Coroutine
)
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .file_scanner import FileScanner, FileInfo, ScanConfig, MultiRootScanner
from .content_extractor import ContentExtractor, ExtractedContent
from .checkpoint_manager import CheckpointManager, JobStatus
from .progress_tracker import (
    ProgressTracker, 
    ConsoleProgressReporter, 
    FileProgressLogger,
    ProgressEvent
)
from .batch_processor import BatchProcessor, BatchConfig, BatchResult, ProcessingResult

logger = logging.getLogger(__name__)


ProcessorFunction = Callable[[ExtractedContent], Coroutine[Any, Any, Any]]


@dataclass
class ProcessingConfig:
    """Konfiguracja przetwarzania."""
    # Skanowanie
    include_patterns: List[str] = field(default_factory=lambda: ['*'])
    exclude_patterns: List[str] = field(default_factory=list)
    exclude_hidden: bool = True
    exclude_dirs: List[str] = field(default_factory=lambda: [
        '.git', '__pycache__', 'node_modules', '.venv', 'venv'
    ])
    max_file_size: int = 100 * 1024 * 1024  # 100 MB
    follow_symlinks: bool = False
    
    # Batch processing
    batch_size: int = 100
    max_concurrent: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    continue_on_error: bool = True
    
    # Checkpoints
    use_checkpoints: bool = True
    checkpoint_dir: Optional[Path] = None
    checkpoint_save_interval: int = 50
    
    # Progress tracking
    enable_progress: bool = True
    progress_interval: float = 1.0
    log_file: Optional[Path] = None
    
    # Streaming
    use_streaming: bool = True
    streaming_threshold: int = 10 * 1024 * 1024  # 10 MB


@dataclass
class ProcessingSummary:
    """Podsumowanie przetwarzania."""
    job_id: str
    job_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    total_bytes: int = 0
    duration_seconds: float = 0.0
    errors: List[Dict[str, str]] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Wskaźnik sukcesu."""
        if self.total_files == 0:
            return 0.0
        return self.processed_files / self.total_files
    
    @property
    def processing_speed(self) -> float:
        """Prędkość przetwarzania (pliki/s)."""
        if self.duration_seconds == 0:
            return 0.0
        return (self.processed_files + self.failed_files + self.skipped_files) / self.duration_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuj do słownika."""
        return {
            'job_id': self.job_id,
            'job_name': self.job_name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_files': self.total_files,
            'processed_files': self.processed_files,
            'failed_files': self.failed_files,
            'skipped_files': self.skipped_files,
            'total_bytes': self.total_bytes,
            'duration_seconds': round(self.duration_seconds, 2),
            'success_rate': round(self.success_rate, 4),
            'processing_speed': round(self.processing_speed, 2),
            'errors': self.errors
        }


class FileProcessor:
    """Główna klasa do przetwarzania plików."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Inicjalizuj procesor plików.
        
        Args:
            config: Konfiguracja przetwarzania
        """
        self.config = config or ProcessingConfig()
        
        # Inicjalizuj komponenty
        self._init_components()
    
    def _init_components(self):
        """Zainicjalizuj komponenty przetwarzania."""
        # Scanner
        scan_config = ScanConfig(
            include_patterns=self.config.include_patterns,
            exclude_patterns=self.config.exclude_patterns,
            exclude_hidden=self.config.exclude_hidden,
            exclude_dirs=self.config.exclude_dirs,
            max_file_size=self.config.max_file_size,
            follow_symlinks=self.config.follow_symlinks,
            detect_encoding=True,
            detect_binary=True
        )
        self.scanner = FileScanner(scan_config)
        
        # Content Extractor
        self.content_extractor = ContentExtractor()
        
        # Checkpoint Manager
        self.checkpoint_manager = None
        if self.config.use_checkpoints and self.config.checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(
                self.config.checkpoint_dir,
                auto_save_interval=self.config.checkpoint_save_interval
            )
        
        # Progress Tracker
        self.progress_tracker = None
        if self.config.enable_progress:
            self.progress_tracker = ProgressTracker(
                report_interval=self.config.progress_interval
            )
            
            # Dodaj reporter konsolowy
            console_reporter = ConsoleProgressReporter()
            self.progress_tracker.add_callback(console_reporter)
            
            # Dodaj logger do pliku jeśli podano
            if self.config.log_file:
                file_logger = FileProgressLogger(self.config.log_file)
                self.progress_tracker.add_callback(file_logger)
        
        # Batch Processor
        batch_config = BatchConfig(
            batch_size=self.config.batch_size,
            max_concurrent=self.config.max_concurrent,
            max_retries=self.config.max_retries,
            retry_delay=self.config.retry_delay,
            continue_on_error=self.config.continue_on_error,
            progress_interval=self.config.progress_interval,
            use_checkpoints=self.config.use_checkpoints,
            auto_save_checkpoints=self.config.use_checkpoints,
            checkpoint_save_interval=self.config.checkpoint_save_interval,
            use_streaming=self.config.use_streaming,
            streaming_threshold=self.config.streaming_threshold
        )
        
        self.batch_processor = BatchProcessor(
            config=batch_config,
            checkpoint_manager=self.checkpoint_manager,
            progress_tracker=self.progress_tracker,
            content_extractor=self.content_extractor
        )
    
    async def process_directory(
        self,
        directory: Path,
        processor: ProcessorFunction,
        patterns: Optional[List[str]] = None,
        job_name: Optional[str] = None,
        job_id: Optional[str] = None
    ) -> ProcessingSummary:
        """
        Przetwórz wszystkie pliki w katalogu.
        
        Args:
            directory: Ścieżka do katalogu
            processor: Funkcja przetwarzająca
            patterns: Opcjonalne patterny do filtrowania
            job_name: Nazwa jobu
            job_id: ID jobu (do wznawiania)
        
        Returns:
            ProcessingSummary - podsumowanie przetwarzania
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Katalog nie istnieje: {directory}")
        
        # Zaktualizuj patterny jeśli podano
        if patterns:
            self.scanner.config.include_patterns = patterns
        
        # Zeskanuj katalog
        logger.info(f"Skanowanie katalogu: {directory}")
        files = self.scanner.scan(directory)
        logger.info(f"Znaleziono {len(files)} plików")
        
        if not files:
            return ProcessingSummary(
                job_id=job_id or "no_job",
                job_name=job_name or "empty_job",
                start_time=datetime.now()
            )
        
        # Przetwórz pliki
        return await self.process_files(
            files, processor, job_name, job_id
        )
    
    async def process_files(
        self,
        files: List[FileInfo],
        processor: ProcessorFunction,
        job_name: Optional[str] = None,
        job_id: Optional[str] = None
    ) -> ProcessingSummary:
        """
        Przetwórz listę plików.
        
        Args:
            files: Lista plików do przetworzenia
            processor: Funkcja przetwarzająca
            job_name: Nazwa jobu
            job_id: ID jobu (do wznawiania)
        
        Returns:
            ProcessingSummary - podsumowanie przetwarzania
        """
        start_time = datetime.now()
        job_name = job_name or f"job_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        # Przetwórz w batchach
        batch_results = await self.batch_processor.process_all(
            files, processor, job_name, job_id
        )
        
        # Zbierz statystyki
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        total_processed = sum(len(br.successful) for br in batch_results)
        total_failed = sum(len(br.failed) for br in batch_results)
        total_bytes = sum(f.size for f in files)
        
        # Zbierz błędy
        errors = []
        for br in batch_results:
            for result in br.failed:
                errors.append({
                    'file': str(result.file_path),
                    'error': result.error or 'Unknown error'
                })
        
        # Pobierz job_id z batch processor
        final_job_id = self.batch_processor._current_job_id or job_id or "unknown"
        
        summary = ProcessingSummary(
            job_id=final_job_id,
            job_name=job_name,
            start_time=start_time,
            end_time=end_time,
            total_files=len(files),
            processed_files=total_processed,
            failed_files=total_failed,
            skipped_files=0,  # TODO: track skipped
            total_bytes=total_bytes,
            duration_seconds=duration,
            errors=errors
        )
        
        logger.info(f"Przetwarzanie zakończone: {summary.to_dict()}")
        
        return summary
    
    async def process_streaming(
        self,
        files: List[FileInfo],
        processor: ProcessorFunction
    ) -> AsyncIterator[ProcessingResult]:
        """
        Przetwarzaj pliki w trybie streaming.
        
        Args:
            files: Lista plików do przetworzenia
            processor: Funkcja przetwarzająca
        
        Yields:
            ProcessingResult - wyniki na bieżąco
        """
        async for result in self.batch_processor.process_streaming(files, processor):
            yield result
    
    async def resume_job(self, job_id: str, processor: ProcessorFunction) -> ProcessingSummary:
        """
        Wznów przerwany job.
        
        Args:
            job_id: ID jobu do wznowienia
            processor: Funkcja przetwarzająca
        
        Returns:
            ProcessingSummary - podsumowanie przetwarzania
        """
        if not self.checkpoint_manager:
            raise ValueError("Checkpoint manager nie jest skonfigurowany")
        
        # Wczytaj checkpoint
        checkpoint = await self.checkpoint_manager.load_checkpoint(job_id)
        
        if not checkpoint:
            raise ValueError(f"Nie znaleziono checkpointu dla jobu: {job_id}")
        
        # Pobierz pliki do wznowienia
        resumable_paths = await self.checkpoint_manager.get_resumable_files(job_id)
        
        if not resumable_paths:
            logger.info(f"Brak plików do wznowienia dla jobu {job_id}")
            # Zwróć podsumowanie z checkpointu
            return ProcessingSummary(
                job_id=job_id,
                job_name=checkpoint.job_name,
                start_time=datetime.fromisoformat(checkpoint.created_at),
                end_time=datetime.fromisoformat(checkpoint.completed_at) if checkpoint.completed_at else None,
                total_files=checkpoint.total_files,
                processed_files=checkpoint.processed_files,
                failed_files=checkpoint.failed_files,
                skipped_files=checkpoint.skipped_files
            )
        
        # Utwórz FileInfo dla resumable files
        files = []
        for path in resumable_paths:
            try:
                stat = path.stat()
                files.append(FileInfo(
                    path=path,
                    size=stat.st_size,
                    modified_time=stat.st_mtime
                ))
            except Exception as e:
                logger.warning(f"Nie można odczytać {path}: {e}")
        
        logger.info(f"Wznawianie jobu {job_id}, {len(files)} plików do przetworzenia")
        
        # Przetwórz pliki
        return await self.process_files(
            files, processor, checkpoint.job_name, job_id
        )
    
    def pause(self):
        """Wstrzymaj przetwarzanie."""
        self.batch_processor.pause()
    
    def resume(self):
        """Wznów przetwarzanie."""
        self.batch_processor.resume()
    
    def cancel(self):
        """Anuluj przetwarzanie."""
        self.batch_processor.cancel()
    
    async def get_job_stats(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Pobierz statystyki jobu."""
        if not self.checkpoint_manager:
            return None
        return self.checkpoint_manager.get_job_stats(job_id)
    
    async def list_jobs(self) -> List[Dict[str, Any]]:
        """Pobierz listę wszystkich jobów."""
        if not self.checkpoint_manager:
            return []
        
        jobs = await self.checkpoint_manager.list_jobs()
        return [job.to_dict() for job in jobs]


@dataclass
class PipelineStage:
    """Pojedynczy etap pipeline'u przetwarzania."""
    name: str
    processor: ProcessorFunction
    enabled: bool = True


class ProcessingPipeline:
    """Pipeline wieloetapowego przetwarzania."""
    
    def __init__(self, processor: Optional[FileProcessor] = None):
        """
        Inicjalizuj pipeline.
        
        Args:
            processor: Opcjonalny FileProcessor (tworzony domyślny jeśli nie podano)
        """
        self.processor = processor or FileProcessor()
        self.stages: List[PipelineStage] = []
    
    def add_stage(self, name: str, processor: ProcessorFunction, enabled: bool = True):
        """Dodaj etap do pipeline'u."""
        self.stages.append(PipelineStage(name, processor, enabled))
        return self
    
    def remove_stage(self, name: str):
        """Usuń etap z pipeline'u."""
        self.stages = [s for s in self.stages if s.name != name]
        return self
    
    def enable_stage(self, name: str):
        """Włącz etap."""
        for stage in self.stages:
            if stage.name == name:
                stage.enabled = True
        return self
    
    def disable_stage(self, name: str):
        """Wyłącz etap."""
        for stage in self.stages:
            if stage.name == name:
                stage.enabled = False
        return self
    
    async def process(
        self,
        directory: Path,
        patterns: Optional[List[str]] = None,
        job_name: Optional[str] = None
    ) -> ProcessingSummary:
        """
        Przetwórz pliki przez wszystkie etapy pipeline'u.
        
        Args:
            directory: Katalog z plikami
            patterns: Patterny do filtrowania
            job_name: Nazwa jobu
        
        Returns:
            ProcessingSummary - podsumowanie przetwarzania
        """
        # Utwórz chain processor
        async def chain_processor(content: ExtractedContent) -> Dict[str, Any]:
            result = {'content': content, 'stage_results': {}}
            
            for stage in self.stages:
                if not stage.enabled:
                    continue
                
                try:
                    stage_result = await stage.processor(content)
                    result['stage_results'][stage.name] = stage_result
                except Exception as e:
                    logger.error(f"Błąd w etapie '{stage.name}': {e}")
                    raise
            
            return result
        
        # Przetwórz katalog
        return await self.processor.process_directory(
            directory, chain_processor, patterns, job_name
        )


# Funkcje pomocnicze

async def quick_process(
    directory: Path,
    processor: ProcessorFunction,
    patterns: Optional[List[str]] = None,
    max_concurrent: int = 10,
    batch_size: int = 100
) -> ProcessingSummary:
    """Szybkie przetwarzanie katalogu bez konfiguracji."""
    config = ProcessingConfig(
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        use_checkpoints=False,
        enable_progress=True
    )
    
    file_processor = FileProcessor(config)
    return await file_processor.process_directory(directory, processor, patterns)


async def process_with_checkpoints(
    directory: Path,
    processor: ProcessorFunction,
    checkpoint_dir: Path,
    patterns: Optional[List[str]] = None,
    job_name: Optional[str] = None,
    job_id: Optional[str] = None
) -> ProcessingSummary:
    """Przetwarzanie z checkpointami."""
    config = ProcessingConfig(
        use_checkpoints=True,
        checkpoint_dir=checkpoint_dir,
        enable_progress=True
    )
    
    file_processor = FileProcessor(config)
    return await file_processor.process_directory(
        directory, processor, patterns, job_name, job_id
    )


async def resume_processing(
    checkpoint_dir: Path,
    job_id: str,
    processor: ProcessorFunction
) -> ProcessingSummary:
    """Wznów przetwarzanie z checkpointu."""
    config = ProcessingConfig(
        use_checkpoints=True,
        checkpoint_dir=checkpoint_dir,
        enable_progress=True
    )
    
    file_processor = FileProcessor(config)
    return await file_processor.resume_job(job_id, processor)
