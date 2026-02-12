"""
FileScanner - Moduł skanowania katalogów dla AgentSwarm.

Funkcjonalności:
- Rekurencyjne skanowanie katalogów
- Filtrowanie plików po patternach (glob)
- Wykrywanie kodowania plików tekstowych
- Obsługa symlinków i ukrytych plików
"""

import os
import fnmatch
import asyncio
import aiofiles
from pathlib import Path
from typing import List, Set, Optional, Callable, Iterator, AsyncIterator, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EncodingDetector:
    """Wykrywanie kodowania plików tekstowych."""
    
    # Kolejność próbowanych kodowań
    DEFAULT_ENCODINGS = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
    
    # BOM markers
    BOM_MAP = {
        b'\xef\xbb\xbf': 'utf-8-sig',
        b'\xff\xfe': 'utf-16-le',
        b'\xfe\xff': 'utf-16-be',
        b'\xff\xfe\x00\x00': 'utf-32-le',
        b'\x00\x00\xfe\xff': 'utf-32-be',
    }
    
    @classmethod
    def detect_from_bom(cls, file_path: Path) -> Optional[str]:
        """Wykryj kodowanie na podstawie BOM."""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                for bom, encoding in cls.BOM_MAP.items():
                    if header.startswith(bom):
                        return encoding
        except Exception:
            pass
        return None
    
    @classmethod
    def detect_by_content(cls, file_path: Path, encodings: Optional[List[str]] = None) -> str:
        """Wykryj kodowanie przez próbę odczytu."""
        encodings = encodings or cls.DEFAULT_ENCODINGS
        
        # Najpierw sprawdź BOM
        bom_encoding = cls.detect_from_bom(file_path)
        if bom_encoding:
            return bom_encoding
        
        # Próbuj kolejnych kodowań
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1024)  # Przeczytaj próbkę
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # Fallback do latin-1 (zawsze działa)
        return 'latin-1'
    
    @classmethod
    async def detect_async(cls, file_path: Path, encodings: Optional[List[str]] = None) -> str:
        """Asynchroniczne wykrywanie kodowania."""
        encodings = encodings or cls.DEFAULT_ENCODINGS
        
        # Sprawdź BOM
        bom_encoding = cls.detect_from_bom(file_path)
        if bom_encoding:
            return bom_encoding
        
        # Próbuj kolejnych kodowań
        for encoding in encodings:
            try:
                async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                    await f.read(1024)
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        return 'latin-1'


@dataclass
class FileInfo:
    """Informacje o przeskanowanym pliku."""
    path: Path
    size: int
    modified_time: float
    encoding: Optional[str] = None
    is_binary: bool = False
    mime_type: Optional[str] = None
    
    @property
    def extension(self) -> str:
        """Zwróć rozszerzenie pliku."""
        return self.path.suffix.lower()
    
    @property
    def name(self) -> str:
        """Zwróć nazwę pliku."""
        return self.path.name
    
    @property
    def relative_to(self) -> Callable[[Path], Path]:
        """Zwróć ścieżkę względem podanej bazy."""
        return lambda base: self.path.relative_to(base)


@dataclass
class ScanConfig:
    """Konfiguracja skanowania."""
    # Patterny do włączenia (glob)
    include_patterns: List[str] = field(default_factory=lambda: ['*'])
    
    # Patterny do wykluczenia (glob)
    exclude_patterns: List[str] = field(default_factory=list)
    
    # Wyklucz ukryte pliki (zaczynające się od .)
    exclude_hidden: bool = True
    
    # Wyklucz katalogi
    exclude_dirs: List[str] = field(default_factory=lambda: [
        '.git', '.svn', '.hg', '__pycache__', '.pytest_cache',
        'node_modules', '.venv', 'venv', 'env', '.tox', '.idea', '.vscode',
        'dist', 'build', '*.egg-info', '.mypy_cache'
    ])
    
    # Maksymalny rozmiar pliku (w bajtach), 0 = brak limitu
    max_file_size: int = 100 * 1024 * 1024  # 100 MB
    
    # Minimalny rozmiar pliku (w bajtach)
    min_file_size: int = 0
    
    # Czy śledzić symlinki
    follow_symlinks: bool = False
    
    # Czy wykrywać kodowanie
    detect_encoding: bool = True
    
    # Czy wykrywać pliki binarne
    detect_binary: bool = True
    
    # Dodatkowy filtr (funkcja przyjmująca Path, zwracająca bool)
    custom_filter: Optional[Callable[[Path], bool]] = None


class FileScanner:
    """Skaner plików z obsługą async i filtrowaniem."""
    
    # Rozszerzenia plików tekstowych (do wykrywania binarnych)
    TEXT_EXTENSIONS = {
        '.txt', '.md', '.rst', '.py', '.js', '.ts', '.jsx', '.tsx',
        '.html', '.htm', '.css', '.scss', '.sass', '.less',
        '.json', '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg',
        '.c', '.cpp', '.h', '.hpp', '.java', '.go', '.rs', '.swift',
        '.rb', '.php', '.pl', '.sh', '.bash', '.zsh', '.ps1',
        '.sql', '.r', '.m', '.scala', '.kt', '.kts',
        '.csv', '.tsv', '.log', '.conf', '.properties',
    }
    
    # Rozszerzenia plików binarnych
    BINARY_EXTENSIONS = {
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar',
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.ico', '.webp',
        '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv',
        '.exe', '.dll', '.so', '.dylib', '.bin',
        '.db', '.sqlite', '.sqlite3',
    }
    
    def __init__(self, config: Optional[ScanConfig] = None):
        self.config = config or ScanConfig()
        self._scanned_count = 0
        self._filtered_count = 0
    
    def _is_hidden(self, path: Path) -> bool:
        """Sprawdź czy plik/katalog jest ukryty."""
        return any(part.startswith('.') for part in path.parts)
    
    def _matches_patterns(self, name: str, patterns: List[str]) -> bool:
        """Sprawdź czy nazwa pasuje do któregokolwiek patternu."""
        return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)
    
    def _should_exclude_dir(self, path: Path) -> bool:
        """Sprawdź czy katalog powinien być wykluczony."""
        name = path.name
        
        # Wyklucz ukryte
        if self.config.exclude_hidden and name.startswith('.'):
            return True
        
        # Wyklucz po patternach
        if self._matches_patterns(name, self.config.exclude_dirs):
            return True
        
        return False
    
    def _is_binary_file(self, file_path: Path) -> bool:
        """Sprawdź czy plik jest binarny."""
        ext = file_path.suffix.lower()
        
        # Szybka ścieżka dla znanych rozszerzeń
        if ext in self.TEXT_EXTENSIONS:
            return False
        if ext in self.BINARY_EXTENSIONS:
            return True
        
        # Sprawdź zawartość (heurystyka)
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(8192)
                if not chunk:
                    return False
                
                # Sprawdź czy chunk zawiera null bytes
                if b'\x00' in chunk:
                    return True
                
                # Sprawdź stosunek bajtów niedrukowalnych
                non_printable = sum(1 for b in chunk if b < 32 and b not in (9, 10, 13))
                return non_printable / len(chunk) > 0.3
        except Exception:
            return True
        
        return False
    
    def _should_include_file(self, file_path: Path) -> bool:
        """Sprawdź czy plik powinien być uwzględniony."""
        name = file_path.name
        
        # Wyklucz ukryte
        if self.config.exclude_hidden and name.startswith('.'):
            return False
        
        # Sprawdź include patterns
        if not self._matches_patterns(name, self.config.include_patterns):
            return False
        
        # Sprawdź exclude patterns
        if self._matches_patterns(name, self.config.exclude_patterns):
            return False
        
        # Sprawdź rozmiar
        try:
            size = file_path.stat().st_size
            if self.config.max_file_size > 0 and size > self.config.max_file_size:
                return False
            if size < self.config.min_file_size:
                return False
        except OSError:
            return False
        
        # Custom filter
        if self.config.custom_filter and not self.config.custom_filter(file_path):
            return False
        
        return True
    
    def scan_sync(self, root_path: Path) -> Iterator[FileInfo]:
        """Synchroniczne skanowanie katalogu."""
        root_path = Path(root_path).resolve()
        
        if not root_path.exists():
            raise FileNotFoundError(f"Katalog nie istnieje: {root_path}")
        
        if not root_path.is_dir():
            raise NotADirectoryError(f"Ścieżka nie jest katalogiem: {root_path}")
        
        for dir_path, dir_names, file_names in os.walk(
            root_path, 
            followlinks=self.config.follow_symlinks
        ):
            dir_path = Path(dir_path)
            
            # Filtruj katalogi
            dir_names[:] = [
                d for d in dir_names 
                if not self._should_exclude_dir(dir_path / d)
            ]
            
            for file_name in file_names:
                file_path = dir_path / file_name
                
                # Sprawdź czy plik powinien być uwzględniony
                if not self._should_include_file(file_path):
                    self._filtered_count += 1
                    continue
                
                try:
                    stat = file_path.stat()
                    
                    # Wykryj czy binarny
                    is_binary = False
                    if self.config.detect_binary:
                        is_binary = self._is_binary_file(file_path)
                    
                    # Wykryj kodowanie
                    encoding = None
                    if self.config.detect_encoding and not is_binary:
                        encoding = EncodingDetector.detect_by_content(file_path)
                    
                    self._scanned_count += 1
                    
                    yield FileInfo(
                        path=file_path,
                        size=stat.st_size,
                        modified_time=stat.st_mtime,
                        encoding=encoding,
                        is_binary=is_binary
                    )
                    
                except (OSError, PermissionError) as e:
                    logger.warning(f"Nie można odczytać pliku {file_path}: {e}")
                    continue
    
    async def scan_async(self, root_path: Path) -> AsyncIterator[FileInfo]:
        """Asynchroniczne skanowanie katalogu."""
        root_path = Path(root_path).resolve()
        
        if not root_path.exists():
            raise FileNotFoundError(f"Katalog nie istnieje: {root_path}")
        
        if not root_path.is_dir():
            raise NotADirectoryError(f"Ścieżka nie jest katalogiem: {root_path}")
        
        # Użyj thread pool dla operacji I/O
        loop = asyncio.get_event_loop()
        
        def scan_generator():
            return list(self.scan_sync(root_path))
        
        files = await loop.run_in_executor(None, scan_generator)
        
        for file_info in files:
            yield file_info
    
    def scan(self, root_path: Path) -> List[FileInfo]:
        """Skanuj katalog i zwróć listę plików."""
        return list(self.scan_sync(root_path))
    
    def get_stats(self) -> Dict[str, int]:
        """Zwróć statystyki skanowania."""
        return {
            'scanned': self._scanned_count,
            'filtered': self._filtered_count,
            'total': self._scanned_count + self._filtered_count
        }
    
    def reset_stats(self):
        """Zresetuj statystyki."""
        self._scanned_count = 0
        self._filtered_count = 0


class MultiRootScanner:
    """Skaner wielu katalogów jednocześnie."""
    
    def __init__(self, config: Optional[ScanConfig] = None):
        self.config = config or ScanConfig()
        self._scanner = FileScanner(config)
    
    def scan_multiple(self, root_paths: List[Path]) -> Iterator[FileInfo]:
        """Skanuj wiele katalogów."""
        seen_paths: Set[Path] = set()
        
        for root_path in root_paths:
            root_path = Path(root_path).resolve()
            
            for file_info in self._scanner.scan_sync(root_path):
                # Unikaj duplikatów
                if file_info.path not in seen_paths:
                    seen_paths.add(file_info.path)
                    yield file_info
    
    async def scan_multiple_async(self, root_paths: List[Path]) -> AsyncIterator[FileInfo]:
        """Asynchroniczne skanowanie wielu katalogów."""
        seen_paths: Set[Path] = set()
        
        for root_path in root_paths:
            async for file_info in self._scanner.scan_async(root_path):
                if file_info.path not in seen_paths:
                    seen_paths.add(file_info.path)
                    yield file_info
