"""
BatchProcessor - Przetwarzanie batchy plików z async/await.

Funkcjonalności:
- Przetwarzanie batchy z async/await
- Limit równoległych zadań (semafory)
- Memory-efficient streaming
- Retry logic dla błędów
- Integracja z checkpointami i progress tracking
"""

import asyncio
import time
from pathlib import Path
from typing import (
    List, Callable, Optional, Dict, Any, AsyncIterator, 
    TypeVar, Generic, Union, Coroutine
)
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import logging

from .file_scanner import FileInfo
from .content_extractor import ContentExtractor, ExtractedContent
from .checkpoint_manager import CheckpointManager, JobStatus
from .progress_tracker import ProgressTracker, ProgressEvent, FileProgress

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class ProcessingError(Exception):
    """Błąd podczas przetwarzania."""
    pass


class RetryExhaustedError(ProcessingError):
    """Wyczerpane próby retry."""
    pass


@dataclass
class BatchConfig:
    """Konfiguracja przetwarzania batchy."""
    # Rozmiar batcha
    batch_size: int = 100
    
    # Maksymalna liczba równoległych zadań
    max_concurrent: int = 10
    
    # Liczba retry przy błędzie
    max_retries: int = 3
    
    # Opóźnienie między retry (sekundy)
    retry_delay: float = 1.0
    
    # Czy kontynuować przy błędzie pojedynczego pliku
    continue_on_error: bool = True
    
    # Interwał raportowania postępu (sekundy)
    progress_interval: float = 1.0
    
    # Czy używać checkpointów
    use_checkpoints: bool = True
    
    # Czy zapisywać checkpoint automatycznie
    auto_save_checkpoints: bool = True
    
    # Interwał auto-zapisu checkpoint (liczba plików)
    checkpoint_save_interval: int = 50
    
    # Limit pamięci dla batcha (MB), 0 = brak limitu
    memory_limit_mb: int = 0
    
    # Czy używać streaming dla dużych plików
    use_streaming: bool = True
    
    # Próg rozmiaru dla streaming (bajty)
    streaming_threshold: int = 10 * 1024 * 1024  # 10 MB


@dataclass
class ProcessingResult:
    """Wynik przetwarzania pliku."""
    file_path: Path
    success: bool
    data: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    retries: int = 0
    
    @property
    def failed(self) -> bool:
        """Czy przetwarzanie się nie powiodło."""
        return not self.success


@dataclass
class BatchResult:
    """Wynik przetwarzania batcha."""
    batch_number: int
    results: List[ProcessingResult]
    duration: float
    
    @property
    def successful(self) -> List[ProcessingResult]:
        """Pomyślne wyniki."""
        return [r for r in self.results if r.success]
    
    @property
    def failed(self) -> List[ProcessingResult]:
        """Niepomyślne wyniki."""
        return [r for r in self.results if not r.success]
    
    @property
    def success_rate(self) -> float:
        """Wskaźnik sukcesu (0-1)."""
        if not self.results:
            return 0.0
        return len(self.successful) / len(self.results)


class RetryManager:
    """Manager retry dla operacji."""
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        exponential_backoff: bool = True
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.exponential_backoff = exponential_backoff
    
    async def execute(
        self,
        operation: Callable[[], Coroutine[Any, Any, T]],
        on_retry: Optional[Callable[[int, Exception], None]] = None
    ) -> T:
        """Wykonaj operację z retry."""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await operation()
            except Exception as e:
                last_error = e
                
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt if self.exponential_backoff else 1)
                    
                    if on_retry:
                        on_retry(attempt + 1, e)
                    
                    logger.warning(f"Próba {attempt + 1} nieudana, retry za {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    break
        
        raise RetryExhaustedError(f"Wyczerpane próby po {self.max_retries} retry. Ostatni błąd: {last_error}")


class BatchProcessor:
    """Procesor batchy plików z async/await."""
    
    def __init__(
        self,
        config: Optional[BatchConfig] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        progress_tracker: Optional[ProgressTracker] = None,
        content_extractor: Optional[ContentExtractor] = None
    ):
        """
        Inicjalizuj procesor.
        
        Args:
            config: Konfiguracja przetwarzania
            checkpoint_manager: Manager checkpointów
            progress_tracker: Tracker postępu
            content_extractor: Ekstraktor treści
        """
        self.config = config or BatchConfig()
        self.checkpoint_manager = checkpoint_manager
        self.progress_tracker = progress_tracker
        self.content_extractor = content_extractor or ContentExtractor()
        self.retry_manager = RetryManager(
            max_retries=self.config.max_retries,
            retry_delay=self.config.retry_delay
        )
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._current_job_id: Optional[str] = None
        self._is_running = False
        self._is_paused = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Początkowo nie wstrzymane
    
    async def _get_semaphore(self) -> asyncio.Semaphore:
        """Pobierz lub utwórz semafor."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        return self._semaphore
    
    async def _wait_if_paused(self):
        """Poczekaj jeśli przetwarzanie jest wstrzymane."""
        await self._pause_event.wait()
    
    def pause(self):
        """Wstrzymaj przetwarzanie."""
        self._is_paused = True
        self._pause_event.clear()
        if self.progress_tracker:
            asyncio.create_task(self.progress_tracker.pause())
        logger.info("Przetwarzanie wstrzymane")
    
    def resume(self):
        """Wznów przetwarzanie."""
        self._is_paused = False
        self._pause_event.set()
        if self.progress_tracker:
            asyncio.create_task(self.progress_tracker.resume())
        logger.info("Przetwarzanie wznowione")
    
    def cancel(self):
        """Anuluj przetwarzanie."""
        self._is_running = False
        logger.info("Przetwarzanie anulowane")
    
    async def process_file(
        self,
        file_info: FileInfo,
        processor: Callable[[ExtractedContent], Coroutine[Any, Any, R]],
        encoding: Optional[str] = None
    ) -> ProcessingResult:
        """
        Przetwórz pojedynczy plik.
        
        Args:
            file_info: Informacje o pliku
            processor: Funkcja przetwarzająca
            encoding: Kodowanie pliku
        
        Returns:
            ProcessingResult - wynik przetwarzania
        """
        start_time = time.time()
        file_path = file_info.path
        
        try:
            # Poczekaj jeśli wstrzymane
            await self._wait_if_paused()
            
            # Zgłoś rozpoczęcie pliku
            if self.progress_tracker:
                await self.progress_tracker.file_started(file_path, file_info.size)
            
            # Ekstrahuj treść
            extracted = await self.content_extractor.extract_async(
                file_path, 
                encoding or file_info.encoding
            )
            
            if not extracted.success:
                raise ProcessingError(f"Błąd ekstrakcji: {extracted.error_message}")
            
            # Przetwórz treść z retry
            async def process_operation():
                return await processor(extracted)
            
            result_data = await self.retry_manager.execute(process_operation)
            
            duration = time.time() - start_time
            
            # Zgłoś zakończenie
            if self.progress_tracker:
                await self.progress_tracker.file_completed(file_path, file_info.size)
            
            return ProcessingResult(
                file_path=file_path,
                success=True,
                data=result_data,
                duration=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            logger.error(f"Błąd przetwarzania {file_path}: {e}")
            
            # Zgłoś błąd
            if self.progress_tracker:
                await self.progress_tracker.file_failed(file_path, error_msg)
            
            return ProcessingResult(
                file_path=file_path,
                success=False,
                error=error_msg,
                duration=duration
            )
    
    async def process_batch(
        self,
        files: List[FileInfo],
        processor: Callable[[ExtractedContent], Coroutine[Any, Any, R]],
        batch_number: int = 1
    ) -> BatchResult:
        """
        Przetwórz batch plików.
        
        Args:
            files: Lista plików do przetworzenia
            processor: Funkcja przetwarzająca
            batch_number: Numer batcha
        
        Returns:
            BatchResult - wynik batcha
        """
        start_time = time.time()
        semaphore = await self._get_semaphore()
        
        async def process_with_limit(file_info: FileInfo) -> ProcessingResult:
            """Przetwórz plik z limitem równoległości."""
            async with semaphore:
                return await self.process_file(file_info, processor)
        
        # Utwórz zadania dla wszystkich plików
        tasks = [process_with_limit(f) for f in files]
        
        # Wykonuj zadania i zbieraj wyniki
        results = []
        for completed in asyncio.as_completed(tasks):
            result = await completed
            results.append(result)
            
            # Aktualizuj checkpoint jeśli włączony
            if self.config.use_checkpoints and self._current_job_id:
                status = 'completed' if result.success else 'failed'
                await self.checkpoint_manager.update_file_status(
                    self._current_job_id,
                    result.file_path,
                    status,
                    result.error if not result.success else None
                )
        
        duration = time.time() - start_time
        
        # Zgłoś zakończenie batcha
        if self.progress_tracker:
            await self.progress_tracker.batch_completed(batch_number)
        
        return BatchResult(
            batch_number=batch_number,
            results=results,
            duration=duration
        )
    
    async def process_all(
        self,
        files: List[FileInfo],
        processor: Callable[[ExtractedContent], Coroutine[Any, Any, R]],
        job_name: str = "batch_job",
        job_id: Optional[str] = None
    ) -> List[BatchResult]:
        """
        Przetwórz wszystkie pliki w batchach.
        
        Args:
            files: Lista plików do przetworzenia
            processor: Funkcja przetwarzająca
            job_name: Nazwa jobu
            job_id: Opcjonalne ID jobu
        
        Returns:
            Lista wyników batchy
        """
        if not files:
            logger.warning("Brak plików do przetworzenia")
            return []
        
        self._is_running = True
        
        # Utwórz lub wczytaj checkpoint
        if self.config.use_checkpoints and self.checkpoint_manager:
            if job_id:
                checkpoint = await self.checkpoint_manager.load_checkpoint(job_id)
                if checkpoint:
                    self._current_job_id = job_id
                    # Filtruj już przetworzone pliki
                    pending_files = [
                        f for f in files 
                        if str(f.path) in checkpoint.get_pending_files()
                    ]
                    files = pending_files or files
                    logger.info(f"Wznowiono job {job_id}, pozostało {len(files)} plików")
                else:
                    checkpoint = await self.checkpoint_manager.create_job(
                        job_name, [f.path for f in files], job_id=job_id
                    )
                    self._current_job_id = checkpoint.job_id
            else:
                checkpoint = await self.checkpoint_manager.create_job(
                    job_name, [f.path for f in files]
                )
                self._current_job_id = checkpoint.job_id
        
        # Inicjalizuj progress tracker
        if self.progress_tracker:
            total_bytes = sum(f.size for f in files)
            num_batches = (len(files) + self.config.batch_size - 1) // self.config.batch_size
            self.progress_tracker.stats.total_files = len(files)
            self.progress_tracker.stats.total_bytes = total_bytes
            self.progress_tracker.stats.total_batches = num_batches
            await self.progress_tracker.start()
        
        # Aktualizuj status jobu
        if self.config.use_checkpoints and self.checkpoint_manager and self._current_job_id:
            await self.checkpoint_manager.update_job_status(
                self._current_job_id, 
                JobStatus.RUNNING
            )
        
        # Podziel na batch'e
        batches = [
            files[i:i + self.config.batch_size]
            for i in range(0, len(files), self.config.batch_size)
        ]
        
        batch_results = []
        
        try:
            for batch_num, batch_files in enumerate(batches, 1):
                if not self._is_running:
                    logger.info("Przetwarzanie zostało anulowane")
                    break
                
                logger.info(f"Przetwarzanie batcha {batch_num}/{len(batches)} ({len(batch_files)} plików)")
                
                result = await self.process_batch(batch_files, processor, batch_num)
                batch_results.append(result)
                
                # Sprawdź czy kontynuować przy błędach
                if not self.config.continue_on_error and result.failed:
                    logger.error(f"Batch {batch_num} zawiera błędy, przerywam")
                    break
                
                # Zapisz checkpoint
                if (self.config.use_checkpoints and 
                    self.config.auto_save_checkpoints and 
                    self.checkpoint_manager and 
                    self._current_job_id):
                    await self.checkpoint_manager.save_progress(self._current_job_id)
            
            # Zakończ tracking
            if self.progress_tracker:
                await self.progress_tracker.stop()
            
            # Oznacz job jako ukończony
            if (self.config.use_checkpoints and 
                self.checkpoint_manager and 
                self._current_job_id):
                await self.checkpoint_manager.update_job_status(
                    self._current_job_id,
                    JobStatus.COMPLETED
                )
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Błąd podczas przetwarzania: {e}")
            
            # Oznacz job jako nieudany
            if (self.config.use_checkpoints and 
                self.checkpoint_manager and 
                self._current_job_id):
                await self.checkpoint_manager.update_job_status(
                    self._current_job_id,
                    JobStatus.FAILED,
                    {'error': str(e)}
                )
            
            raise
        
        finally:
            self._is_running = False
    
    async def process_streaming(
        self,
        files: List[FileInfo],
        processor: Callable[[ExtractedContent], Coroutine[Any, Any, R]]
    ) -> AsyncIterator[ProcessingResult]:
        """
        Przetwarzaj pliki w trybie streaming (wyniki na bieżąco).
        
        Args:
            files: Lista plików do przetworzenia
            processor: Funkcja przetwarzająca
        
        Yields:
            ProcessingResult - wyniki przetwarzania
        """
        semaphore = await self._get_semaphore()
        
        async def process_with_limit(file_info: FileInfo) -> ProcessingResult:
            async with semaphore:
                return await self.process_file(file_info, processor)
        
        # Przetwarzaj pliki i yielduj wyniki
        for file_info in files:
            if not self._is_running:
                break
            
            result = await process_with_limit(file_info)
            yield result


def create_default_processor(
    checkpoint_dir: Optional[Path] = None,
    enable_progress: bool = True,
    max_concurrent: int = 10,
    batch_size: int = 100
) -> BatchProcessor:
    """Utwórz domyślny procesor batchy."""
    config = BatchConfig(
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        use_checkpoints=checkpoint_dir is not None
    )
    
    checkpoint_manager = None
    if checkpoint_dir:
        checkpoint_manager = CheckpointManager(checkpoint_dir)
    
    progress_tracker = None
    if enable_progress:
        from .progress_tracker import create_progress_tracker_with_console
        # Będzie zainicjalizowany w process_all
        progress_tracker = ProgressTracker()
    
    return BatchProcessor(
        config=config,
        checkpoint_manager=checkpoint_manager,
        progress_tracker=progress_tracker
    )


# Dekoratory pomocnicze

def retry_on_error(max_retries: int = 3, retry_delay: float = 1.0):
    """Dekorator do retry operacji."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retry_mgr = RetryManager(max_retries, retry_delay)
            return await retry_mgr.execute(lambda: func(*args, **kwargs))
        return wrapper
    return decorator


def rate_limited(max_calls: int, period: float = 1.0):
    """Dekorator do rate limiting."""
    def decorator(func):
        semaphore = asyncio.Semaphore(max_calls)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with semaphore:
                return await func(*args, **kwargs)
        return wrapper
    return decorator
