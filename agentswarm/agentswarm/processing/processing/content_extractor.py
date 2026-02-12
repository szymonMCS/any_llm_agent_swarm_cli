"""
ContentExtractor - Ekstrakcja tekstu z różnych formatów plików.

Wspierane formaty:
- Tekstowe: .txt, .md, .rst, .log
- Kod: .py, .js, .ts, .jsx, .tsx, .html, .css, .java, .go, .rs, .cpp, .c, .h
- Dane: .json, .csv, .tsv, .xml, .yaml, .yml
- Dokumenty: .pdf (opcjonalnie), .docx (opcjonalnie)
"""

import os
import json
import csv
import io
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List, Union, AsyncIterator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
import aiofiles
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExtractedContent:
    """Wynik ekstrakcji tekstu z pliku."""
    text: str
    source_path: Path
    encoding: str = 'utf-8'
    mime_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    chunks: List[str] = field(default_factory=list)
    
    @property
    def length(self) -> int:
        """Długość tekstu."""
        return len(self.text)
    
    @property
    def line_count(self) -> int:
        """Liczba linii."""
        return len(self.text.splitlines())


class ExtractionError(Exception):
    """Błąd podczas ekstrakcji treści."""
    pass


class BaseExtractor(ABC):
    """Bazowa klasa ekstraktora."""
    
    SUPPORTED_EXTENSIONS: set = set()
    
    @classmethod
    def supports(cls, file_path: Path) -> bool:
        """Sprawdź czy ekstraktor obsługuje dany plik."""
        return file_path.suffix.lower() in cls.SUPPORTED_EXTENSIONS
    
    @abstractmethod
    async def extract_async(self, file_path: Path, encoding: Optional[str] = None) -> ExtractedContent:
        """Asynchroniczna ekstrakcja treści."""
        pass
    
    @abstractmethod
    def extract_sync(self, file_path: Path, encoding: Optional[str] = None) -> ExtractedContent:
        """Synchroniczna ekstrakcja treści."""
        pass
    
    def _create_error_result(self, file_path: Path, error: Exception) -> ExtractedContent:
        """Utwórz wynik błędu."""
        return ExtractedContent(
            text="",
            source_path=file_path,
            success=False,
            error_message=str(error)
        )


class TextExtractor(BaseExtractor):
    """Ekstraktor plików tekstowych."""
    
    SUPPORTED_EXTENSIONS = {
        '.txt', '.md', '.rst', '.log', '.conf', '.cfg', '.ini',
        '.properties', '.env', '.gitignore', '.dockerignore'
    }
    
    async def extract_async(self, file_path: Path, encoding: Optional[str] = None) -> ExtractedContent:
        """Asynchroniczna ekstrakcja pliku tekstowego."""
        try:
            encoding = encoding or 'utf-8'
            async with aiofiles.open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = await f.read()
            
            return ExtractedContent(
                text=content,
                source_path=file_path,
                encoding=encoding,
                mime_type='text/plain'
            )
        except Exception as e:
            logger.error(f"Błąd ekstrakcji {file_path}: {e}")
            return self._create_error_result(file_path, e)
    
    def extract_sync(self, file_path: Path, encoding: Optional[str] = None) -> ExtractedContent:
        """Synchroniczna ekstrakcja pliku tekstowego."""
        try:
            encoding = encoding or 'utf-8'
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            
            return ExtractedContent(
                text=content,
                source_path=file_path,
                encoding=encoding,
                mime_type='text/plain'
            )
        except Exception as e:
            logger.error(f"Błąd ekstrakcji {file_path}: {e}")
            return self._create_error_result(file_path, e)


class CodeExtractor(BaseExtractor):
    """Ekstraktor plików kodu źródłowego."""
    
    SUPPORTED_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.htm', '.css',
        '.scss', '.sass', '.less', '.java', '.go', '.rs', '.swift',
        '.c', '.cpp', '.h', '.hpp', '.cs', '.vb', '.fs', '.fsx',
        '.rb', '.php', '.pl', '.pm', '.sh', '.bash', '.zsh', '.ps1',
        '.sql', '.r', '.m', '.mm', '.scala', '.kt', '.kts', '.groovy',
        '.dart', '.lua', '.vim', '.el', '.clj', '.cljs', '.edn',
        '.erl', '.hrl', '.ex', '.exs', '.elm', '.hs', '.lhs',
        '.ml', '.mli', '.fsi', '.fs', '.fsx', '.fsi',
    }
    
    MIME_TYPES = {
        '.py': 'text/x-python',
        '.js': 'text/javascript',
        '.ts': 'text/typescript',
        '.jsx': 'text/jsx',
        '.tsx': 'text/tsx',
        '.html': 'text/html',
        '.htm': 'text/html',
        '.css': 'text/css',
        '.scss': 'text/x-scss',
        '.sass': 'text/x-sass',
        '.less': 'text/x-less',
        '.java': 'text/x-java',
        '.go': 'text/x-go',
        '.rs': 'text/x-rust',
        '.swift': 'text/x-swift',
        '.c': 'text/x-c',
        '.cpp': 'text/x-c++',
        '.h': 'text/x-c',
        '.hpp': 'text/x-c++',
        '.cs': 'text/x-csharp',
        '.rb': 'text/x-ruby',
        '.php': 'text/x-php',
        '.sh': 'text/x-shellscript',
    }
    
    async def extract_async(self, file_path: Path, encoding: Optional[str] = None) -> ExtractedContent:
        """Asynchroniczna ekstrakcja pliku kodu."""
        try:
            encoding = encoding or 'utf-8'
            async with aiofiles.open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = await f.read()
            
            mime_type = self.MIME_TYPES.get(file_path.suffix.lower(), 'text/plain')
            
            return ExtractedContent(
                text=content,
                source_path=file_path,
                encoding=encoding,
                mime_type=mime_type,
                metadata={'language': file_path.suffix.lower()[1:]}
            )
        except Exception as e:
            logger.error(f"Błąd ekstrakcji {file_path}: {e}")
            return self._create_error_result(file_path, e)
    
    def extract_sync(self, file_path: Path, encoding: Optional[str] = None) -> ExtractedContent:
        """Synchroniczna ekstrakcja pliku kodu."""
        try:
            encoding = encoding or 'utf-8'
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            
            mime_type = self.MIME_TYPES.get(file_path.suffix.lower(), 'text/plain')
            
            return ExtractedContent(
                text=content,
                source_path=file_path,
                encoding=encoding,
                mime_type=mime_type,
                metadata={'language': file_path.suffix.lower()[1:]}
            )
        except Exception as e:
            logger.error(f"Błąd ekstrakcji {file_path}: {e}")
            return self._create_error_result(file_path, e)


class JSONExtractor(BaseExtractor):
    """Ekstraktor plików JSON."""
    
    SUPPORTED_EXTENSIONS = {'.json', '.jsonl'}
    
    async def extract_async(self, file_path: Path, encoding: Optional[str] = None) -> ExtractedContent:
        """Asynchroniczna ekstrakcja JSON."""
        try:
            encoding = encoding or 'utf-8'
            async with aiofiles.open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = await f.read()
            
            # Parsuj JSON dla walidacji
            try:
                parsed = json.loads(content)
                metadata = {
                    'type': 'json',
                    'keys': list(parsed.keys()) if isinstance(parsed, dict) else None,
                    'length': len(parsed) if isinstance(parsed, (list, dict)) else None
                }
            except json.JSONDecodeError:
                metadata = {'type': 'json', 'valid': False}
            
            return ExtractedContent(
                text=content,
                source_path=file_path,
                encoding=encoding,
                mime_type='application/json',
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Błąd ekstrakcji {file_path}: {e}")
            return self._create_error_result(file_path, e)
    
    def extract_sync(self, file_path: Path, encoding: Optional[str] = None) -> ExtractedContent:
        """Synchroniczna ekstrakcja JSON."""
        try:
            encoding = encoding or 'utf-8'
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            
            try:
                parsed = json.loads(content)
                metadata = {
                    'type': 'json',
                    'keys': list(parsed.keys()) if isinstance(parsed, dict) else None,
                    'length': len(parsed) if isinstance(parsed, (list, dict)) else None
                }
            except json.JSONDecodeError:
                metadata = {'type': 'json', 'valid': False}
            
            return ExtractedContent(
                text=content,
                source_path=file_path,
                encoding=encoding,
                mime_type='application/json',
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Błąd ekstrakcji {file_path}: {e}")
            return self._create_error_result(file_path, e)


class CSVExtractor(BaseExtractor):
    """Ekstraktor plików CSV i TSV."""
    
    SUPPORTED_EXTENSIONS = {'.csv', '.tsv'}
    
    async def extract_async(self, file_path: Path, encoding: Optional[str] = None) -> ExtractedContent:
        """Asynchroniczna ekstrakcja CSV."""
        try:
            encoding = encoding or 'utf-8'
            delimiter = '\t' if file_path.suffix.lower() == '.tsv' else ','
            
            async with aiofiles.open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = await f.read()
            
            # Parsuj CSV
            try:
                reader = csv.reader(io.StringIO(content), delimiter=delimiter)
                rows = list(reader)
                metadata = {
                    'type': 'csv',
                    'delimiter': delimiter,
                    'rows': len(rows),
                    'columns': len(rows[0]) if rows else 0
                }
            except Exception:
                metadata = {'type': 'csv', 'valid': False}
            
            return ExtractedContent(
                text=content,
                source_path=file_path,
                encoding=encoding,
                mime_type='text/csv',
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Błąd ekstrakcji {file_path}: {e}")
            return self._create_error_result(file_path, e)
    
    def extract_sync(self, file_path: Path, encoding: Optional[str] = None) -> ExtractedContent:
        """Synchroniczna ekstrakcja CSV."""
        try:
            encoding = encoding or 'utf-8'
            delimiter = '\t' if file_path.suffix.lower() == '.tsv' else ','
            
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            
            try:
                reader = csv.reader(io.StringIO(content), delimiter=delimiter)
                rows = list(reader)
                metadata = {
                    'type': 'csv',
                    'delimiter': delimiter,
                    'rows': len(rows),
                    'columns': len(rows[0]) if rows else 0
                }
            except Exception:
                metadata = {'type': 'csv', 'valid': False}
            
            return ExtractedContent(
                text=content,
                source_path=file_path,
                encoding=encoding,
                mime_type='text/csv',
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Błąd ekstrakcji {file_path}: {e}")
            return self._create_error_result(file_path, e)


class XMLExtractor(BaseExtractor):
    """Ekstraktor plików XML."""
    
    SUPPORTED_EXTENSIONS = {'.xml', '.svg', '.wsdl', '.xsd'}
    
    async def extract_async(self, file_path: Path, encoding: Optional[str] = None) -> ExtractedContent:
        """Asynchroniczna ekstrakcja XML."""
        try:
            encoding = encoding or 'utf-8'
            async with aiofiles.open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = await f.read()
            
            return ExtractedContent(
                text=content,
                source_path=file_path,
                encoding=encoding,
                mime_type='application/xml'
            )
        except Exception as e:
            logger.error(f"Błąd ekstrakcji {file_path}: {e}")
            return self._create_error_result(file_path, e)
    
    def extract_sync(self, file_path: Path, encoding: Optional[str] = None) -> ExtractedContent:
        """Synchroniczna ekstrakcja XML."""
        try:
            encoding = encoding or 'utf-8'
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            
            return ExtractedContent(
                text=content,
                source_path=file_path,
                encoding=encoding,
                mime_type='application/xml'
            )
        except Exception as e:
            logger.error(f"Błąd ekstrakcji {file_path}: {e}")
            return self._create_error_result(file_path, e)


class YAMLExtractor(BaseExtractor):
    """Ekstraktor plików YAML."""
    
    SUPPORTED_EXTENSIONS = {'.yaml', '.yml'}
    
    async def extract_async(self, file_path: Path, encoding: Optional[str] = None) -> ExtractedContent:
        """Asynchroniczna ekstrakcja YAML."""
        try:
            encoding = encoding or 'utf-8'
            async with aiofiles.open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = await f.read()
            
            return ExtractedContent(
                text=content,
                source_path=file_path,
                encoding=encoding,
                mime_type='application/x-yaml'
            )
        except Exception as e:
            logger.error(f"Błąd ekstrakcji {file_path}: {e}")
            return self._create_error_result(file_path, e)
    
    def extract_sync(self, file_path: Path, encoding: Optional[str] = None) -> ExtractedContent:
        """Synchroniczna ekstrakcja YAML."""
        try:
            encoding = encoding or 'utf-8'
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            
            return ExtractedContent(
                text=content,
                source_path=file_path,
                encoding=encoding,
                mime_type='application/x-yaml'
            )
        except Exception as e:
            logger.error(f"Błąd ekstrakcji {file_path}: {e}")
            return self._create_error_result(file_path, e)


class PDFExtractor(BaseExtractor):
    """Ekstraktor plików PDF (opcjonalny - wymaga PyPDF2)."""
    
    SUPPORTED_EXTENSIONS = {'.pdf'}
    
    def __init__(self):
        self._has_pypdf = False
        try:
            import PyPDF2
            self._has_pypdf = True
        except ImportError:
            logger.warning("PyPDF2 nie jest zainstalowany. Ekstrakcja PDF będzie niedostępna.")
    
    async def extract_async(self, file_path: Path, encoding: Optional[str] = None) -> ExtractedContent:
        """Asynchroniczna ekstrakcja PDF."""
        if not self._has_pypdf:
            return self._create_error_result(
                file_path, 
                Exception("PyPDF2 nie jest zainstalowany. Zainstaluj: pip install PyPDF2")
            )
        
        try:
            # PDF extraction wymaga synchronicznego kodu
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.extract_sync, file_path, encoding)
        except Exception as e:
            logger.error(f"Błąd ekstrakcji PDF {file_path}: {e}")
            return self._create_error_result(file_path, e)
    
    def extract_sync(self, file_path: Path, encoding: Optional[str] = None) -> ExtractedContent:
        """Synchroniczna ekstrakcja PDF."""
        if not self._has_pypdf:
            return self._create_error_result(
                file_path,
                Exception("PyPDF2 nie jest zainstalowany. Zainstaluj: pip install PyPDF2")
            )
        
        try:
            import PyPDF2
            
            text_parts = []
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                num_pages = len(reader.pages)
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                    except Exception as e:
                        text_parts.append(f"--- Page {page_num + 1} [Error: {e}] ---")
            
            content = '\n\n'.join(text_parts)
            
            return ExtractedContent(
                text=content,
                source_path=file_path,
                encoding='utf-8',
                mime_type='application/pdf',
                metadata={'pages': num_pages}
            )
        except Exception as e:
            logger.error(f"Błąd ekstrakcji PDF {file_path}: {e}")
            return self._create_error_result(file_path, e)


class DocxExtractor(BaseExtractor):
    """Ekstraktor plików DOCX (opcjonalny - wymaga python-docx)."""
    
    SUPPORTED_EXTENSIONS = {'.docx'}
    
    def __init__(self):
        self._has_docx = False
        try:
            import docx
            self._has_docx = True
        except ImportError:
            logger.warning("python-docx nie jest zainstalowany. Ekstrakcja DOCX będzie niedostępna.")
    
    async def extract_async(self, file_path: Path, encoding: Optional[str] = None) -> ExtractedContent:
        """Asynchroniczna ekstrakcja DOCX."""
        if not self._has_docx:
            return self._create_error_result(
                file_path,
                Exception("python-docx nie jest zainstalowany. Zainstaluj: pip install python-docx")
            )
        
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.extract_sync, file_path, encoding)
        except Exception as e:
            logger.error(f"Błąd ekstrakcji DOCX {file_path}: {e}")
            return self._create_error_result(file_path, e)
    
    def extract_sync(self, file_path: Path, encoding: Optional[str] = None) -> ExtractedContent:
        """Synchroniczna ekstrakcja DOCX."""
        if not self._has_docx:
            return self._create_error_result(
                file_path,
                Exception("python-docx nie jest zainstalowany. Zainstaluj: pip install python-docx")
            )
        
        try:
            import docx
            
            document = docx.Document(file_path)
            paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
            
            content = '\n\n'.join(paragraphs)
            
            return ExtractedContent(
                text=content,
                source_path=file_path,
                encoding='utf-8',
                mime_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                metadata={'paragraphs': len(paragraphs)}
            )
        except Exception as e:
            logger.error(f"Błąd ekstrakcji DOCX {file_path}: {e}")
            return self._create_error_result(file_path, e)


class ContentExtractor:
    """Główna klasa ekstraktora - zarządza wszystkimi ekstraktorami."""
    
    def __init__(self):
        self._extractors: List[BaseExtractor] = [
            TextExtractor(),
            CodeExtractor(),
            JSONExtractor(),
            CSVExtractor(),
            XMLExtractor(),
            YAMLExtractor(),
            PDFExtractor(),
            DocxExtractor(),
        ]
        self._extension_map: Dict[str, BaseExtractor] = {}
        self._build_extension_map()
    
    def _build_extension_map(self):
        """Zbuduj mapę rozszerzeń do ekstraktorów."""
        for extractor in self._extractors:
            for ext in extractor.SUPPORTED_EXTENSIONS:
                self._extension_map[ext.lower()] = extractor
    
    def register_extractor(self, extractor: BaseExtractor):
        """Zarejestruj niestandardowy ekstraktor."""
        self._extractors.append(extractor)
        for ext in extractor.SUPPORTED_EXTENSIONS:
            self._extension_map[ext.lower()] = extractor
    
    def get_extractor(self, file_path: Path) -> Optional[BaseExtractor]:
        """Pobierz ekstraktor dla danego pliku."""
        ext = file_path.suffix.lower()
        return self._extension_map.get(ext)
    
    def can_extract(self, file_path: Path) -> bool:
        """Sprawdź czy można ekstrahować dany plik."""
        return self.get_extractor(file_path) is not None
    
    async def extract_async(
        self, 
        file_path: Path, 
        encoding: Optional[str] = None
    ) -> ExtractedContent:
        """Asynchroniczna ekstrakcja pliku."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return ExtractedContent(
                text="",
                source_path=file_path,
                success=False,
                error_message=f"Plik nie istnieje: {file_path}"
            )
        
        extractor = self.get_extractor(file_path)
        
        if extractor is None:
            # Fallback - spróbuj jako tekst
            return await TextExtractor().extract_async(file_path, encoding)
        
        return await extractor.extract_async(file_path, encoding)
    
    def extract_sync(
        self, 
        file_path: Path, 
        encoding: Optional[str] = None
    ) -> ExtractedContent:
        """Synchroniczna ekstrakcja pliku."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return ExtractedContent(
                text="",
                source_path=file_path,
                success=False,
                error_message=f"Plik nie istnieje: {file_path}"
            )
        
        extractor = self.get_extractor(file_path)
        
        if extractor is None:
            return TextExtractor().extract_sync(file_path, encoding)
        
        return extractor.extract_sync(file_path, encoding)
    
    async def extract_batch_async(
        self,
        file_paths: List[Path],
        encoding: Optional[str] = None,
        max_concurrent: int = 10
    ) -> AsyncIterator[ExtractedContent]:
        """Asynchroniczna ekstrakcja batcha plików."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def extract_with_limit(path: Path) -> ExtractedContent:
            async with semaphore:
                return await self.extract_async(path, encoding)
        
        tasks = [extract_with_limit(path) for path in file_paths]
        
        for completed in asyncio.as_completed(tasks):
            result = await completed
            yield result
    
    def get_supported_extensions(self) -> List[str]:
        """Zwróć listę wspieranych rozszerzeń."""
        return sorted(self._extension_map.keys())
