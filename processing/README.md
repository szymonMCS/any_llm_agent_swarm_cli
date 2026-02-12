# AgentSwarm Processing Module

Wydajny system przetwarzania plików dla aplikacji AgentSwarm z obsługą async/await, batch processing, checkpointów i śledzenia postępu.

## Funkcjonalności

- **FileScanner** - Skanowanie katalogów z filtrowaniem (glob patterns), wykrywanie kodowania
- **ContentExtractor** - Ekstrakcja tekstu z różnych formatów (.txt, .py, .js, .md, .json, .csv, .pdf, .docx)
- **CheckpointManager** - Zapisywanie postępu, wznawianie przerwanych zadań
- **ProgressTracker** - Raportowanie postępu z ETA i statystykami
- **BatchProcessor** - Przetwarzanie batchy z async/await i limitem równoległych zadań

## Wymagania

```bash
pip install aiofiles
```

Opcjonalne (dla dodatkowych formatów):
```bash
pip install PyPDF2       # dla PDF
pip install python-docx  # dla DOCX
```

## Szybki start

### Podstawowe przetwarzanie

```python
import asyncio
from pathlib import Path
from agentswarm.processing import quick_process
from agentswarm.processing.content_extractor import ExtractedContent

async def main():
    async def my_processor(content: ExtractedContent):
        return {
            'file': str(content.source_path),
            'lines': len(content.text.splitlines())
        }
    
    result = await quick_process(
        directory=Path("./src"),
        processor=my_processor,
        patterns=["*.py", "*.js"],
        max_concurrent=10
    )
    
    print(f"Przetworzono: {result.processed_files} plików")
    print(f"Czas: {result.duration_seconds:.2f}s")

asyncio.run(main())
```

### Przetwarzanie z checkpointami

```python
from agentswarm.processing import process_with_checkpoints, resume_processing

# Pierwsze uruchomienie
result = await process_with_checkpoints(
    directory=Path("./src"),
    processor=my_processor,
    checkpoint_dir=Path("./checkpoints"),
    patterns=["*.py"],
    job_name="my_analysis"
)

print(f"Job ID: {result.job_id}")  # Zapisz ID do wznowienia

# Wznowienie (jeśli przerwane)
result = await resume_processing(
    checkpoint_dir=Path("./checkpoints"),
    job_id="abc123",
    processor=my_processor
)
```

### Zaawansowana konfiguracja

```python
from agentswarm.processing import FileProcessor, ProcessingConfig

config = ProcessingConfig(
    include_patterns=["*.py", "*.md"],
    exclude_patterns=["*_test.py"],
    exclude_dirs=['.git', '__pycache__', 'tests'],
    max_file_size=50 * 1024 * 1024,  # 50 MB
    batch_size=50,
    max_concurrent=8,
    max_retries=3,
    use_checkpoints=True,
    checkpoint_dir=Path("./checkpoints"),
    enable_progress=True
)

processor = FileProcessor(config)

result = await processor.process_directory(
    directory=Path("./project"),
    processor=my_processor,
    job_name="full_analysis"
)
```

### Pipeline wieloetapowy

```python
from agentswarm.processing import ProcessingPipeline

pipeline = ProcessingPipeline()

# Etap 1: Ekstrakcja metadanych
async def stage1_metadata(content):
    return {'size': len(content.text), 'lines': len(content.text.splitlines())}

# Etap 2: Analiza zawartości
async def stage2_analysis(content):
    return {'todos': content.text.count('TODO'), 'functions': content.text.count('def ')}

# Etap 3: Generowanie raportu
async def stage3_report(content):
    return {'processed': True, 'timestamp': time.time()}

pipeline.add_stage("metadata", stage1_metadata)
pipeline.add_stage("analysis", stage2_analysis)
pipeline.add_stage("report", stage3_report)

result = await pipeline.process(
    directory=Path("./src"),
    patterns=["*.py"]
)
```

### Streaming wyników

```python
async def process_streaming():
    from agentswarm.processing import FileScanner, FileProcessor, ProcessingConfig
    
    config = ProcessingConfig(enable_progress=False)
    processor = FileProcessor(config)
    scanner = FileScanner()
    
    files = scanner.scan(Path("./src"))
    
    async for result in processor.process_streaming(files, my_processor):
        if result.success:
            print(f"✓ {result.file_path.name}")
        else:
            print(f"✗ {result.file_path.name}: {result.error}")
```

## API Reference

### FileScanner

```python
from agentswarm.processing import FileScanner, ScanConfig

config = ScanConfig(
    include_patterns=["*.py"],
    exclude_patterns=["test_*.py"],
    exclude_dirs=['.git', '__pycache__'],
    max_file_size=100 * 1024 * 1024,
    detect_encoding=True
)

scanner = FileScanner(config)
files = scanner.scan(Path("./src"))  # List[FileInfo]
```

### ContentExtractor

```python
from agentswarm.processing import ContentExtractor

extractor = ContentExtractor()

# Synchronicznie
result = extractor.extract_sync(Path("file.py"))

# Asynchronicznie
result = await extractor.extract_async(Path("file.py"))

# Batch
async for result in extractor.extract_batch_async(file_paths, max_concurrent=10):
    print(result.text)
```

### CheckpointManager

```python
from agentswarm.processing import CheckpointManager

manager = CheckpointManager(Path("./checkpoints"))

# Utwórz job
checkpoint = await manager.create_job("my_job", file_paths)

# Aktualizuj status
await manager.update_file_status(job_id, file_path, 'completed')

# Wznów
resumable_files = await manager.get_resumable_files(job_id)

# Statystyki
stats = manager.get_job_stats(job_id)
```

### ProgressTracker

```python
from agentswarm.processing import ProgressTracker, ConsoleProgressReporter

tracker = ProgressTracker(total_files=100)
tracker.add_callback(ConsoleProgressReporter())

await tracker.start()
await tracker.file_started(file_path)
await tracker.file_completed(file_path)
await tracker.stop()

stats = tracker.get_current_stats()
print(f"Postęp: {stats.progress_percentage}%")
```

### BatchProcessor

```python
from agentswarm.processing import BatchProcessor, BatchConfig

config = BatchConfig(
    batch_size=100,
    max_concurrent=10,
    max_retries=3,
    retry_delay=1.0
)

processor = BatchProcessor(config)
results = await processor.process_all(files, my_processor, "job_name")
```

## Wspierane formaty plików

| Format | Rozszerzenia | Wymagania |
|--------|--------------|-----------|
| Tekst | .txt, .md, .rst, .log | - |
| Kod | .py, .js, .ts, .jsx, .tsx, .html, .css, .java, .go, .rs, .cpp, .c, .h | - |
| Dane | .json, .csv, .tsv, .xml, .yaml, .yml | - |
| PDF | .pdf | PyPDF2 |
| Word | .docx | python-docx |

## Architektura

```
┌─────────────────────────────────────────────────────────────┐
│                    FileProcessor (Orchestrator)              │
├─────────────────────────────────────────────────────────────┤
│  FileScanner → ContentExtractor → BatchProcessor            │
│       ↓                ↓                ↓                   │
│  ScanConfig      ExtractedContent   BatchConfig             │
│       ↓                ↓                ↓                   │
│  FileInfo         Metadata         ProcessingResult         │
├─────────────────────────────────────────────────────────────┤
│  CheckpointManager ← → ProgressTracker                      │
│  JobCheckpoint          ProgressStats                       │
│  FileCheckpoint           FileProgress                      │
└─────────────────────────────────────────────────────────────┘
```

## Wydajność

- **Async/await** - Efektywne operacje I/O
- **Batch processing** - Konfigurowalny rozmiar batcha
- **Semafor** - Limit równoległych zadań
- **Streaming** - Memory-efficient dla dużych plików
- **Checkpoint** - Wznawianie bez ponownego przetwarzania

## Testy

```bash
python -m agentswarm.processing.test_processing
```

## Przykłady

Zobacz `examples.py` dla więcej przykładów użycia.

## Licencja

MIT
