"""
CheckpointManager - Zarządzanie punktami kontrolnymi dla przetwarzania.

Funkcjonalności:
- Zapisywanie postępu przetwarzania
- Wznawianie przerwanych zadań
- Zarządzanie stanem przetwarzania
- Obsługa wielu jobów jednocześnie
"""

import json
import hashlib
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import aiofiles
import logging

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status jobu przetwarzania."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class FileCheckpoint:
    """Punkt kontrolny dla pojedynczego pliku."""
    file_path: str
    file_hash: str  # Hash zawartości lub metadanych
    status: str = "pending"  # pending, processing, completed, failed, skipped
    processed_at: Optional[str] = None
    error_message: Optional[str] = None
    result_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuj do słownika."""
        return {
            'file_path': self.file_path,
            'file_hash': self.file_hash,
            'status': self.status,
            'processed_at': self.processed_at,
            'error_message': self.error_message,
            'result_data': self.result_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileCheckpoint':
        """Utwórz z słownika."""
        return cls(
            file_path=data['file_path'],
            file_hash=data['file_hash'],
            status=data.get('status', 'pending'),
            processed_at=data.get('processed_at'),
            error_message=data.get('error_message'),
            result_data=data.get('result_data', {})
        )


@dataclass
class JobCheckpoint:
    """Punkt kontrolny dla całego jobu przetwarzania."""
    job_id: str
    job_name: str
    status: str = "pending"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_checkpoints: Dict[str, FileCheckpoint] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuj do słownika."""
        return {
            'job_id': self.job_id,
            'job_name': self.job_name,
            'status': self.status,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'completed_at': self.completed_at,
            'total_files': self.total_files,
            'processed_files': self.processed_files,
            'failed_files': self.failed_files,
            'skipped_files': self.skipped_files,
            'config': self.config,
            'metadata': self.metadata,
            'file_checkpoints': {
                k: v.to_dict() for k, v in self.file_checkpoints.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobCheckpoint':
        """Utwórz z słownika."""
        file_checkpoints = {
            k: FileCheckpoint.from_dict(v) 
            for k, v in data.get('file_checkpoints', {}).items()
        }
        
        return cls(
            job_id=data['job_id'],
            job_name=data['job_name'],
            status=data.get('status', 'pending'),
            created_at=data.get('created_at', datetime.now().isoformat()),
            updated_at=data.get('updated_at', datetime.now().isoformat()),
            completed_at=data.get('completed_at'),
            total_files=data.get('total_files', 0),
            processed_files=data.get('processed_files', 0),
            failed_files=data.get('failed_files', 0),
            skipped_files=data.get('skipped_files', 0),
            config=data.get('config', {}),
            metadata=data.get('metadata', {}),
            file_checkpoints=file_checkpoints
        )
    
    @property
    def progress_percentage(self) -> float:
        """Procent ukończenia."""
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100
    
    @property
    def remaining_files(self) -> int:
        """Liczba pozostałych plików."""
        return self.total_files - self.processed_files - self.failed_files - self.skipped_files
    
    def get_pending_files(self) -> List[str]:
        """Zwróć listę plików oczekujących na przetworzenie."""
        return [
            cp.file_path for cp in self.file_checkpoints.values()
            if cp.status in ('pending', 'failed')
        ]
    
    def get_completed_files(self) -> List[str]:
        """Zwróć listę przetworzonych plików."""
        return [
            cp.file_path for cp in self.file_checkpoints.values()
            if cp.status == 'completed'
        ]


class CheckpointManager:
    """Manager punktów kontrolnych dla przetwarzania."""
    
    def __init__(self, checkpoint_dir: Path, auto_save_interval: int = 10):
        """
        Inicjalizuj manager.
        
        Args:
            checkpoint_dir: Katalog do przechowywania checkpointów
            auto_save_interval: Automatyczny zapis co N plików (0 = wyłączone)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.auto_save_interval = auto_save_interval
        self._current_jobs: Dict[str, JobCheckpoint] = {}
        self._save_lock = asyncio.Lock()
        self._pending_saves: Set[str] = set()
    
    def _get_checkpoint_path(self, job_id: str) -> Path:
        """Pobierz ścieżkę do pliku checkpointu."""
        return self.checkpoint_dir / f"{job_id}.json"
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Oblicz hash pliku (metadanych)."""
        try:
            stat = file_path.stat()
            content = f"{file_path}:{stat.st_size}:{stat.st_mtime}"
            return hashlib.md5(content.encode()).hexdigest()
        except Exception:
            # Fallback - użyj samej ścieżki
            return hashlib.md5(str(file_path).encode()).hexdigest()
    
    async def create_job(
        self,
        job_name: str,
        file_paths: List[Path],
        config: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None
    ) -> JobCheckpoint:
        """
        Utwórz nowy job przetwarzania.
        
        Args:
            job_name: Nazwa jobu
            file_paths: Lista ścieżek do plików
            config: Konfiguracja przetwarzania
            job_id: Opcjonalne ID jobu (generowane jeśli nie podane)
        
        Returns:
            JobCheckpoint - punkt kontrolny jobu
        """
        if job_id is None:
            # Generuj ID na podstawie nazwy i timestampu
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            hash_input = f"{job_name}:{timestamp}:{len(file_paths)}"
            job_id = hashlib.md5(hash_input.encode()).hexdigest()[:12]
        
        # Utwórz checkpointy dla plików
        file_checkpoints = {}
        for file_path in file_paths:
            file_path_str = str(file_path)
            file_hash = self._compute_file_hash(file_path)
            file_checkpoints[file_path_str] = FileCheckpoint(
                file_path=file_path_str,
                file_hash=file_hash
            )
        
        checkpoint = JobCheckpoint(
            job_id=job_id,
            job_name=job_name,
            status=JobStatus.PENDING.value,
            total_files=len(file_paths),
            config=config or {},
            file_checkpoints=file_checkpoints
        )
        
        self._current_jobs[job_id] = checkpoint
        await self._save_checkpoint_async(checkpoint)
        
        logger.info(f"Utworzono job '{job_name}' (ID: {job_id}) z {len(file_paths)} plikami")
        
        return checkpoint
    
    def create_job_sync(
        self,
        job_name: str,
        file_paths: List[Path],
        config: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None
    ) -> JobCheckpoint:
        """Synchroniczna wersja create_job."""
        if job_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            hash_input = f"{job_name}:{timestamp}:{len(file_paths)}"
            job_id = hashlib.md5(hash_input.encode()).hexdigest()[:12]
        
        file_checkpoints = {}
        for file_path in file_paths:
            file_path_str = str(file_path)
            file_hash = self._compute_file_hash(file_path)
            file_checkpoints[file_path_str] = FileCheckpoint(
                file_path=file_path_str,
                file_hash=file_hash
            )
        
        checkpoint = JobCheckpoint(
            job_id=job_id,
            job_name=job_name,
            status=JobStatus.PENDING.value,
            total_files=len(file_paths),
            config=config or {},
            file_checkpoints=file_checkpoints
        )
        
        self._current_jobs[job_id] = checkpoint
        self._save_checkpoint_sync(checkpoint)
        
        logger.info(f"Utworzono job '{job_name}' (ID: {job_id}) z {len(file_paths)} plikami")
        
        return checkpoint
    
    async def load_checkpoint(self, job_id: str) -> Optional[JobCheckpoint]:
        """Wczytaj checkpoint jobu."""
        # Sprawdź czy jest w pamięci
        if job_id in self._current_jobs:
            return self._current_jobs[job_id]
        
        # Wczytaj z pliku
        checkpoint_path = self._get_checkpoint_path(job_id)
        
        if not checkpoint_path.exists():
            return None
        
        try:
            async with aiofiles.open(checkpoint_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            data = json.loads(content)
            checkpoint = JobCheckpoint.from_dict(data)
            self._current_jobs[job_id] = checkpoint
            
            return checkpoint
        except Exception as e:
            logger.error(f"Błąd wczytywania checkpointu {job_id}: {e}")
            return None
    
    def load_checkpoint_sync(self, job_id: str) -> Optional[JobCheckpoint]:
        """Synchroniczna wersja load_checkpoint."""
        if job_id in self._current_jobs:
            return self._current_jobs[job_id]
        
        checkpoint_path = self._get_checkpoint_path(job_id)
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            checkpoint = JobCheckpoint.from_dict(data)
            self._current_jobs[job_id] = checkpoint
            
            return checkpoint
        except Exception as e:
            logger.error(f"Błąd wczytywania checkpointu {job_id}: {e}")
            return None
    
    async def _save_checkpoint_async(self, checkpoint: JobCheckpoint):
        """Asynchroniczny zapis checkpointu."""
        async with self._save_lock:
            checkpoint_path = self._get_checkpoint_path(checkpoint.job_id)
            checkpoint.updated_at = datetime.now().isoformat()
            
            try:
                async with aiofiles.open(checkpoint_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(checkpoint.to_dict(), indent=2, ensure_ascii=False))
            except Exception as e:
                logger.error(f"Błąd zapisu checkpointu {checkpoint.job_id}: {e}")
    
    def _save_checkpoint_sync(self, checkpoint: JobCheckpoint):
        """Synchroniczny zapis checkpointu."""
        checkpoint_path = self._get_checkpoint_path(checkpoint.job_id)
        checkpoint.updated_at = datetime.now().isoformat()
        
        try:
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Błąd zapisu checkpointu {checkpoint.job_id}: {e}")
    
    async def update_file_status(
        self,
        job_id: str,
        file_path: Path,
        status: str,
        error_message: Optional[str] = None,
        result_data: Optional[Dict[str, Any]] = None,
        auto_save: bool = True
    ):
        """Zaktualizuj status pliku w checkpoint."""
        checkpoint = await self.load_checkpoint(job_id)
        
        if checkpoint is None:
            logger.warning(f"Nie znaleziono checkpointu dla jobu {job_id}")
            return
        
        file_path_str = str(file_path)
        
        if file_path_str not in checkpoint.file_checkpoints:
            # Utwórz nowy checkpoint dla pliku
            file_hash = self._compute_file_hash(file_path)
            checkpoint.file_checkpoints[file_path_str] = FileCheckpoint(
                file_path=file_path_str,
                file_hash=file_hash
            )
        
        file_cp = checkpoint.file_checkpoints[file_path_str]
        
        # Aktualizuj statystyki jeśli status się zmienia
        old_status = file_cp.status
        if old_status != status:
            if old_status == 'completed':
                checkpoint.processed_files -= 1
            elif old_status == 'failed':
                checkpoint.failed_files -= 1
            elif old_status == 'skipped':
                checkpoint.skipped_files -= 1
            
            if status == 'completed':
                checkpoint.processed_files += 1
            elif status == 'failed':
                checkpoint.failed_files += 1
            elif status == 'skipped':
                checkpoint.skipped_files += 1
        
        # Zaktualizuj checkpoint pliku
        file_cp.status = status
        file_cp.processed_at = datetime.now().isoformat()
        if error_message:
            file_cp.error_message = error_message
        if result_data:
            file_cp.result_data.update(result_data)
        
        # Zapisz checkpoint
        if auto_save and self.auto_save_interval > 0:
            total_processed = checkpoint.processed_files + checkpoint.failed_files + checkpoint.skipped_files
            if total_processed % self.auto_save_interval == 0:
                await self._save_checkpoint_async(checkpoint)
    
    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Zaktualizuj status jobu."""
        checkpoint = await self.load_checkpoint(job_id)
        
        if checkpoint is None:
            return
        
        checkpoint.status = status.value
        
        if status == JobStatus.COMPLETED:
            checkpoint.completed_at = datetime.now().isoformat()
        
        if metadata:
            checkpoint.metadata.update(metadata)
        
        await self._save_checkpoint_async(checkpoint)
        
        logger.info(f"Job {job_id} zmienił status na {status.value}")
    
    async def save_progress(self, job_id: str):
        """Wymuś zapis postępu."""
        checkpoint = await self.load_checkpoint(job_id)
        if checkpoint:
            await self._save_checkpoint_async(checkpoint)
    
    def save_progress_sync(self, job_id: str):
        """Synchroniczny zapis postępu."""
        checkpoint = self.load_checkpoint_sync(job_id)
        if checkpoint:
            self._save_checkpoint_sync(checkpoint)
    
    async def get_resumable_files(self, job_id: str) -> List[Path]:
        """Pobierz listę plików do wznowienia przetwarzania."""
        checkpoint = await self.load_checkpoint(job_id)
        
        if checkpoint is None:
            return []
        
        resumable = []
        
        for file_path_str, file_cp in checkpoint.file_checkpoints.items():
            file_path = Path(file_path_str)
            
            # Sprawdź czy plik istnieje
            if not file_path.exists():
                continue
            
            # Sprawdź czy plik się zmienił
            current_hash = self._compute_file_hash(file_path)
            
            if file_cp.status in ('pending', 'failed'):
                # Plik do przetworzenia
                resumable.append(file_path)
            elif file_cp.status == 'completed' and file_cp.file_hash != current_hash:
                # Plik zmienił się od ostatniego przetworzenia
                resumable.append(file_path)
        
        return resumable
    
    def get_resumable_files_sync(self, job_id: str) -> List[Path]:
        """Synchroniczna wersja get_resumable_files."""
        checkpoint = self.load_checkpoint_sync(job_id)
        
        if checkpoint is None:
            return []
        
        resumable = []
        
        for file_path_str, file_cp in checkpoint.file_checkpoints.items():
            file_path = Path(file_path_str)
            
            if not file_path.exists():
                continue
            
            current_hash = self._compute_file_hash(file_path)
            
            if file_cp.status in ('pending', 'failed'):
                resumable.append(file_path)
            elif file_cp.status == 'completed' and file_cp.file_hash != current_hash:
                resumable.append(file_path)
        
        return resumable
    
    async def list_jobs(self) -> List[JobCheckpoint]:
        """Pobierz listę wszystkich jobów."""
        jobs = []
        
        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            job_id = checkpoint_file.stem
            checkpoint = await self.load_checkpoint(job_id)
            if checkpoint:
                jobs.append(checkpoint)
        
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
    def list_jobs_sync(self) -> List[JobCheckpoint]:
        """Synchroniczna wersja list_jobs."""
        jobs = []
        
        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            job_id = checkpoint_file.stem
            checkpoint = self.load_checkpoint_sync(job_id)
            if checkpoint:
                jobs.append(checkpoint)
        
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
    async def delete_checkpoint(self, job_id: str) -> bool:
        """Usuń checkpoint jobu."""
        checkpoint_path = self._get_checkpoint_path(job_id)
        
        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            
            if job_id in self._current_jobs:
                del self._current_jobs[job_id]
            
            return True
        except Exception as e:
            logger.error(f"Błąd usuwania checkpointu {job_id}: {e}")
            return False
    
    async def cleanup_old_checkpoints(self, max_age_days: int = 30) -> int:
        """Wyczyść stare checkpointy."""
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        removed_count = 0
        
        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            try:
                stat = checkpoint_file.stat()
                file_mtime = datetime.fromtimestamp(stat.st_mtime)
                
                if file_mtime < cutoff_date:
                    checkpoint_file.unlink()
                    removed_count += 1
            except Exception as e:
                logger.warning(f"Nie można usunąć {checkpoint_file}: {e}")
        
        logger.info(f"Usunięto {removed_count} starych checkpointów")
        return removed_count
    
    def get_job_stats(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Pobierz statystyki jobu."""
        checkpoint = self.load_checkpoint_sync(job_id)
        
        if checkpoint is None:
            return None
        
        return {
            'job_id': checkpoint.job_id,
            'job_name': checkpoint.job_name,
            'status': checkpoint.status,
            'progress_percentage': checkpoint.progress_percentage,
            'total_files': checkpoint.total_files,
            'processed_files': checkpoint.processed_files,
            'failed_files': checkpoint.failed_files,
            'skipped_files': checkpoint.skipped_files,
            'remaining_files': checkpoint.remaining_files,
            'created_at': checkpoint.created_at,
            'updated_at': checkpoint.updated_at,
            'completed_at': checkpoint.completed_at
        }
