"""
ProgressTracker - Śledzenie i raportowanie postępu przetwarzania.

Funkcjonalności:
- Śledzenie postępu w czasie rzeczywistym
- Szacowanie czasu pozostałego (ETA)
- Raportowanie statystyk
- Callbacki dla zdarzeń postępu
"""

import time
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ProgressEvent(Enum):
    """Typy zdarzeń postępu."""
    STARTED = "started"
    FILE_STARTED = "file_started"
    FILE_COMPLETED = "file_completed"
    FILE_FAILED = "file_failed"
    FILE_SKIPPED = "file_skipped"
    BATCH_COMPLETED = "batch_completed"
    PAUSED = "paused"
    RESUMED = "resumed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"
    PROGRESS = "progress"


@dataclass
class FileProgress:
    """Postęp dla pojedynczego pliku."""
    file_path: Path
    status: str = "pending"  # pending, processing, completed, failed, skipped
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    bytes_processed: int = 0
    total_bytes: int = 0
    error_message: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """Czas przetwarzania w sekundach."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def is_complete(self) -> bool:
        """Czy plik został przetworzony."""
        return self.status in ('completed', 'failed', 'skipped')


@dataclass
class ProgressStats:
    """Statystyki postępu przetwarzania."""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    total_bytes: int = 0
    processed_bytes: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    current_file: Optional[Path] = None
    current_batch: int = 0
    total_batches: int = 0
    
    @property
    def completed_files(self) -> int:
        """Liczba ukończonych plików (wszystkie statusy)."""
        return self.processed_files + self.failed_files + self.skipped_files
    
    @property
    def remaining_files(self) -> int:
        """Liczba pozostałych plików."""
        return self.total_files - self.completed_files
    
    @property
    def progress_percentage(self) -> float:
        """Procent ukończenia (0-100)."""
        if self.total_files == 0:
            return 0.0
        return (self.completed_files / self.total_files) * 100
    
    @property
    def bytes_percentage(self) -> float:
        """Procent przetworzonych bajtów."""
        if self.total_bytes == 0:
            return 0.0
        return (self.processed_bytes / self.total_bytes) * 100
    
    @property
    def elapsed_time(self) -> float:
        """Czas od rozpoczęcia w sekundach."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def elapsed_time_formatted(self) -> str:
        """Sformatowany czas od rozpoczęcia."""
        return self._format_duration(self.elapsed_time)
    
    @property
    def estimated_time_remaining(self) -> Optional[float]:
        """Szacowany pozostały czas w sekundach."""
        if self.completed_files == 0 or self.total_files == 0:
            return None
        
        avg_time_per_file = self.elapsed_time / self.completed_files
        remaining = self.remaining_files * avg_time_per_file
        return remaining
    
    @property
    def estimated_time_remaining_formatted(self) -> str:
        """Sformatowany szacowany pozostały czas."""
        eta = self.estimated_time_remaining
        if eta is None:
            return "N/A"
        return self._format_duration(eta)
    
    @property
    def processing_speed(self) -> float:
        """Prędkość przetwarzania (pliki/sekundę)."""
        if self.elapsed_time == 0:
            return 0.0
        return self.completed_files / self.elapsed_time
    
    @property
    def bytes_per_second(self) -> float:
        """Prędkość przetwarzania (bajty/sekundę)."""
        if self.elapsed_time == 0:
            return 0.0
        return self.processed_bytes / self.elapsed_time
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Sformatuj czas w sekundach do czytelnej postaci."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuj do słownika."""
        return {
            'total_files': self.total_files,
            'processed_files': self.processed_files,
            'failed_files': self.failed_files,
            'skipped_files': self.skipped_files,
            'completed_files': self.completed_files,
            'remaining_files': self.remaining_files,
            'progress_percentage': round(self.progress_percentage, 2),
            'bytes_percentage': round(self.bytes_percentage, 2),
            'elapsed_time': self.elapsed_time_formatted,
            'estimated_time_remaining': self.estimated_time_remaining_formatted,
            'processing_speed': round(self.processing_speed, 2),
            'bytes_per_second': round(self.bytes_per_second, 2),
            'current_file': str(self.current_file) if self.current_file else None,
            'current_batch': self.current_batch,
            'total_batches': self.total_batches,
        }


ProgressCallback = Callable[[ProgressEvent, ProgressStats, Optional[FileProgress]], None]


class ProgressTracker:
    """Śledzenie postępu przetwarzania plików."""
    
    def __init__(
        self,
        total_files: int = 0,
        total_bytes: int = 0,
        total_batches: int = 0,
        callbacks: Optional[List[ProgressCallback]] = None,
        report_interval: float = 1.0  # sekundy
    ):
        """
        Inicjalizuj tracker.
        
        Args:
            total_files: Całkowita liczba plików
            total_bytes: Całkowity rozmiar plików w bajtach
            total_batches: Całkowita liczba batchy
            callbacks: Lista callbacków do wywołania przy zdarzeniach
            report_interval: Interwał raportowania w sekundach
        """
        self.stats = ProgressStats(
            total_files=total_files,
            total_bytes=total_bytes,
            total_batches=total_batches
        )
        self.callbacks = callbacks or []
        self.report_interval = report_interval
        self._file_progress: Dict[Path, FileProgress] = {}
        self._is_running = False
        self._is_paused = False
        self._last_report_time = 0.0
        self._lock = asyncio.Lock()
    
    def add_callback(self, callback: ProgressCallback):
        """Dodaj callback dla zdarzeń postępu."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: ProgressCallback):
        """Usuń callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def _notify_callbacks(
        self,
        event: ProgressEvent,
        file_progress: Optional[FileProgress] = None
    ):
        """Powiadom wszystkie callbacki."""
        for callback in self.callbacks:
            try:
                callback(event, self.stats, file_progress)
            except Exception as e:
                logger.warning(f"Błąd w callbacku postępu: {e}")
    
    async def start(self):
        """Rozpocznij śledzenie postępu."""
        async with self._lock:
            self.stats.start_time = time.time()
            self._is_running = True
            self._is_paused = False
            self._last_report_time = time.time()
        
        self._notify_callbacks(ProgressEvent.STARTED)
        logger.info(f"Rozpoczęto przetwarzanie {self.stats.total_files} plików")
    
    async def stop(self):
        """Zatrzymaj śledzenie postępu."""
        async with self._lock:
            self.stats.end_time = time.time()
            self._is_running = False
        
        self._notify_callbacks(ProgressEvent.COMPLETED)
        logger.info(f"Zakończono przetwarzanie. Czas: {self.stats.elapsed_time_formatted}")
    
    async def pause(self):
        """Wstrzymaj śledzenie."""
        async with self._lock:
            self._is_paused = True
        
        self._notify_callbacks(ProgressEvent.PAUSED)
        logger.info("Wstrzymano przetwarzanie")
    
    async def resume(self):
        """Wznów śledzenie."""
        async with self._lock:
            self._is_paused = False
        
        self._notify_callbacks(ProgressEvent.RESUMED)
        logger.info("Wznowiono przetwarzanie")
    
    async def file_started(self, file_path: Path, total_bytes: int = 0):
        """Zgłoś rozpoczęcie przetwarzania pliku."""
        async with self._lock:
            file_progress = FileProgress(
                file_path=file_path,
                status='processing',
                start_time=time.time(),
                total_bytes=total_bytes
            )
            self._file_progress[file_path] = file_progress
            self.stats.current_file = file_path
        
        self._notify_callbacks(ProgressEvent.FILE_STARTED, file_progress)
    
    async def file_completed(
        self,
        file_path: Path,
        bytes_processed: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Zgłoś zakończenie przetwarzania pliku."""
        async with self._lock:
            file_progress = self._file_progress.get(file_path)
            if file_progress:
                file_progress.status = 'completed'
                file_progress.end_time = time.time()
                file_progress.bytes_processed = bytes_processed
            
            self.stats.processed_files += 1
            self.stats.processed_bytes += bytes_processed
        
        self._notify_callbacks(ProgressEvent.FILE_COMPLETED, file_progress)
        await self._maybe_report_progress()
    
    async def file_failed(
        self,
        file_path: Path,
        error_message: str,
        bytes_processed: int = 0
    ):
        """Zgłoś błąd przetwarzania pliku."""
        async with self._lock:
            file_progress = self._file_progress.get(file_path)
            if file_progress:
                file_progress.status = 'failed'
                file_progress.end_time = time.time()
                file_progress.error_message = error_message
                file_progress.bytes_processed = bytes_processed
            
            self.stats.failed_files += 1
            self.stats.processed_bytes += bytes_processed
        
        self._notify_callbacks(ProgressEvent.FILE_FAILED, file_progress)
        await self._maybe_report_progress()
    
    async def file_skipped(self, file_path: Path, reason: str = ""):
        """Zgłoś pominięcie pliku."""
        async with self._lock:
            file_progress = self._file_progress.get(file_path)
            if file_progress:
                file_progress.status = 'skipped'
                file_progress.end_time = time.time()
            else:
                file_progress = FileProgress(
                    file_path=file_path,
                    status='skipped',
                    end_time=time.time()
                )
                self._file_progress[file_path] = file_progress
            
            self.stats.skipped_files += 1
        
        self._notify_callbacks(ProgressEvent.FILE_SKIPPED, file_progress)
        await self._maybe_report_progress()
    
    async def batch_completed(self, batch_number: int):
        """Zgłoś zakończenie batcha."""
        async with self._lock:
            self.stats.current_batch = batch_number
        
        self._notify_callbacks(ProgressEvent.BATCH_COMPLETED)
    
    async def _maybe_report_progress(self):
        """Raportuj postęp jeśli minął interwał."""
        current_time = time.time()
        
        if current_time - self._last_report_time >= self.report_interval:
            self._last_report_time = current_time
            self._notify_callbacks(ProgressEvent.PROGRESS)
    
    def get_current_stats(self) -> ProgressStats:
        """Pobierz aktualne statystyki."""
        return self.stats
    
    def get_file_progress(self, file_path: Path) -> Optional[FileProgress]:
        """Pobierz postęp dla konkretnego pliku."""
        return self._file_progress.get(file_path)
    
    def get_summary(self) -> Dict[str, Any]:
        """Pobierz podsumowanie przetwarzania."""
        return {
            'stats': self.stats.to_dict(),
            'is_running': self._is_running,
            'is_paused': self._is_paused,
            'files_with_errors': [
                {
                    'path': str(fp.file_path),
                    'error': fp.error_message
                }
                for fp in self._file_progress.values()
                if fp.status == 'failed'
            ]
        }
    
    def print_progress_bar(self, width: int = 50):
        """Wydrukuj pasek postępu."""
        percentage = self.stats.progress_percentage
        filled = int(width * percentage / 100)
        bar = '█' * filled + '░' * (width - filled)
        
        print(f"\r[{bar}] {percentage:.1f}% | "
              f"{self.stats.completed_files}/{self.stats.total_files} plików | "
              f"Czas: {self.stats.elapsed_time_formatted} | "
              f"ETA: {self.stats.estimated_time_remaining_formatted}",
              end='', flush=True)
    
    def print_summary(self):
        """Wydrukuj podsumowanie."""
        print("\n" + "=" * 60)
        print("PODSUMOWANIE PRZETWARZANIA")
        print("=" * 60)
        print(f"Całkowita liczba plików: {self.stats.total_files}")
        print(f"Przetworzone: {self.stats.processed_files}")
        print(f"Błędy: {self.stats.failed_files}")
        print(f"Pominięte: {self.stats.skipped_files}")
        print(f"Czas wykonania: {self.stats.elapsed_time_formatted}")
        print(f"Prędkość: {self.stats.processing_speed:.2f} plików/s")
        print("=" * 60)


class ConsoleProgressReporter:
    """Reporter postępu do konsoli."""
    
    def __init__(
        self,
        show_progress_bar: bool = True,
        show_eta: bool = True,
        update_interval: float = 0.5
    ):
        self.show_progress_bar = show_progress_bar
        self.show_eta = show_eta
        self.update_interval = update_interval
        self._last_update = 0
    
    def __call__(
        self,
        event: ProgressEvent,
        stats: ProgressStats,
        file_progress: Optional[FileProgress] = None
    ):
        """Callback dla zdarzeń postępu."""
        current_time = time.time()
        
        if event == ProgressEvent.STARTED:
            print(f"\nRozpoczynanie przetwarzania {stats.total_files} plików...")
        
        elif event == ProgressEvent.COMPLETED:
            self._print_final_summary(stats)
        
        elif event in (ProgressEvent.PROGRESS, ProgressEvent.FILE_COMPLETED):
            if current_time - self._last_update >= self.update_interval:
                self._last_update = current_time
                self._print_progress(stats)
        
        elif event == ProgressEvent.FILE_FAILED and file_progress:
            print(f"\n⚠ Błąd: {file_progress.file_path.name} - {file_progress.error_message}")
    
    def _print_progress(self, stats: ProgressStats):
        """Wydrukuj aktualny postęp."""
        width = 40
        percentage = stats.progress_percentage
        filled = int(width * percentage / 100)
        bar = '█' * filled + '░' * (width - filled)
        
        eta = f" | ETA: {stats.estimated_time_remaining_formatted}" if self.show_eta else ""
        
        print(f"\r[{bar}] {percentage:.1f}% | "
              f"{stats.completed_files}/{stats.total_files} | "
              f"✓{stats.processed_files} ✗{stats.failed_files} »{stats.skipped_files} | "
              f"⏱ {stats.elapsed_time_formatted}{eta}",
              end='', flush=True)
    
    def _print_final_summary(self, stats: ProgressStats):
        """Wydrukuj końcowe podsumowanie."""
        print("\n" + "─" * 60)
        print("✓ Przetwarzanie zakończone!")
        print(f"  Przetworzone: {stats.processed_files} plików")
        print(f"  Błędy: {stats.failed_files}")
        print(f"  Pominięte: {stats.skipped_files}")
        print(f"  Czas: {stats.elapsed_time_formatted}")
        print(f"  Prędkość: {stats.processing_speed:.2f} plików/s")
        print("─" * 60)


class FileProgressLogger:
    """Logger postępu do pliku."""
    
    def __init__(self, log_file: Path, log_interval: int = 100):
        self.log_file = Path(log_file)
        self.log_interval = log_interval
        self._processed_since_last_log = 0
    
    def __call__(
        self,
        event: ProgressEvent,
        stats: ProgressStats,
        file_progress: Optional[FileProgress] = None
    ):
        """Callback dla zdarzeń postępu."""
        if event == ProgressEvent.STARTED:
            self._log(f"[{datetime.now().isoformat()}] STARTED - {stats.total_files} files")
        
        elif event == ProgressEvent.COMPLETED:
            self._log(f"[{datetime.now().isoformat()}] COMPLETED - "
                     f"processed: {stats.processed_files}, "
                     f"failed: {stats.failed_files}, "
                     f"time: {stats.elapsed_time_formatted}")
        
        elif event == ProgressEvent.FILE_FAILED and file_progress:
            self._log(f"[{datetime.now().isoformat()}] FAILED - {file_progress.file_path} - "
                     f"{file_progress.error_message}")
        
        elif event in (ProgressEvent.FILE_COMPLETED, ProgressEvent.FILE_SKIPPED):
            self._processed_since_last_log += 1
            if self._processed_since_last_log >= self.log_interval:
                self._processed_since_last_log = 0
                self._log(f"[{datetime.now().isoformat()}] PROGRESS - "
                         f"{stats.completed_files}/{stats.total_files} "
                         f"({stats.progress_percentage:.1f}%)")
    
    def _log(self, message: str):
        """Zapisz wiadomość do pliku logu."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')


async def create_progress_tracker_with_console(
    total_files: int,
    total_bytes: int = 0,
    report_interval: float = 1.0
) -> ProgressTracker:
    """Utwórz tracker z reporterem konsolowym."""
    tracker = ProgressTracker(
        total_files=total_files,
        total_bytes=total_bytes,
        report_interval=report_interval
    )
    
    console_reporter = ConsoleProgressReporter()
    tracker.add_callback(console_reporter)
    
    return tracker
