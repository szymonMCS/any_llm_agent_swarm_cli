"""
Przykłady użycia modułu przetwarzania AgentSwarm.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, List
import json

from agentswarm.processing import (
    FileProcessor,
    ProcessingConfig,
    ProcessingPipeline,
    PipelineStage,
    FileScanner,
    ScanConfig,
    ContentExtractor,
    CheckpointManager,
    ProgressTracker,
    BatchProcessor,
    BatchConfig,
    quick_process,
    process_with_checkpoints,
    resume_processing,
)
from agentswarm.processing.content_extractor import ExtractedContent


# ============== PRZYKŁAD 1: Podstawowe przetwarzanie ==============

async def example_basic_processing():
    """Podstawowe przetwarzanie plików."""
    
    # Prosty processor - zliczanie linii kodu
    async def count_lines(content: ExtractedContent) -> Dict[str, Any]:
        lines = content.text.splitlines()
        return {
            'file': str(content.source_path),
            'lines': len(lines),
            'chars': len(content.text),
            'language': content.metadata.get('language', 'unknown')
        }
    
    # Przetwórz katalog
    directory = Path("./src")
    
    result = await quick_process(
        directory=directory,
        processor=count_lines,
        patterns=["*.py", "*.js"],
        max_concurrent=5,
        batch_size=50
    )
    
    print(f"Przetworzono: {result.processed_files} plików")
    print(f"Błędy: {result.failed_files}")
    print(f"Czas: {result.duration_seconds:.2f}s")
    
    return result


# ============== PRZYKŁAD 2: Przetwarzanie z checkpointami ==============

async def example_with_checkpoints():
    """Przetwarzanie z checkpointami i wznawianiem."""
    
    async def analyze_code(content: ExtractedContent) -> Dict[str, Any]:
        """Analiza kodu - zliczanie funkcji, klas, itp."""
        text = content.text
        
        # Prosta heurystyka
        functions = text.count('def ') + text.count('function ')
        classes = text.count('class ')
        comments = text.count('#') + text.count('//')
        
        return {
            'file': str(content.source_path),
            'functions': functions,
            'classes': classes,
            'comments': comments,
            'size': len(text)
        }
    
    checkpoint_dir = Path("./checkpoints")
    
    # Pierwsze uruchomienie
    result = await process_with_checkpoints(
        directory=Path("./src"),
        processor=analyze_code,
        checkpoint_dir=checkpoint_dir,
        patterns=["*.py"],
        job_name="code_analysis"
    )
    
    print(f"Job ID: {result.job_id}")
    
    # Wznowienie (jeśli przerwane)
    # result = await resume_processing(
    #     checkpoint_dir=checkpoint_dir,
    #     job_id=result.job_id,
    #     processor=analyze_code
    # )
    
    return result


# ============== PRZYKŁAD 3: Konfiguracja zaawansowana ==============

async def example_advanced_config():
    """Zaawansowana konfiguracja przetwarzania."""
    
    config = ProcessingConfig(
        # Skanowanie
        include_patterns=["*.py", "*.md", "*.txt"],
        exclude_patterns=["*_test.py", "test_*.py"],
        exclude_hidden=True,
        exclude_dirs=['.git', '__pycache__', 'node_modules', '.venv', 'tests'],
        max_file_size=50 * 1024 * 1024,  # 50 MB
        
        # Batch processing
        batch_size=25,
        max_concurrent=8,
        max_retries=5,
        retry_delay=2.0,
        continue_on_error=True,
        
        # Checkpoints
        use_checkpoints=True,
        checkpoint_dir=Path("./my_checkpoints"),
        checkpoint_save_interval=25,
        
        # Progress tracking
        enable_progress=True,
        progress_interval=0.5,
        log_file=Path("./processing.log"),
        
        # Streaming
        use_streaming=True,
        streaming_threshold=5 * 1024 * 1024  # 5 MB
    )
    
    processor = FileProcessor(config)
    
    async def extract_metadata(content: ExtractedContent) -> Dict[str, Any]:
        return {
            'path': str(content.source_path),
            'mime': content.mime_type,
            'encoding': content.encoding,
            'size': len(content.text),
            'metadata': content.metadata
        }
    
    result = await processor.process_directory(
        directory=Path("./project"),
        processor=extract_metadata,
        job_name="metadata_extraction"
    )
    
    return result


# ============== PRZYKŁAD 4: Pipeline wieloetapowy ==============

async def example_pipeline():
    """Pipeline wieloetapowego przetwarzania."""
    
    # Etap 1: Ekstrakcja metadanych
    async def stage1_metadata(content: ExtractedContent) -> Dict[str, Any]:
        return {
            'file_size': len(content.text),
            'line_count': len(content.text.splitlines()),
            'mime_type': content.mime_type
        }
    
    # Etap 2: Analiza zawartości
    async def stage2_analysis(content: ExtractedContent) -> Dict[str, Any]:
        text = content.text.lower()
        return {
            'word_count': len(text.split()),
            'has_todos': 'todo' in text,
            'has_fixmes': 'fixme' in text,
            'imports': len([l for l in content.text.splitlines() if l.strip().startswith('import ')])
        }
    
    # Etap 3: Generowanie raportu
    async def stage3_report(content: ExtractedContent) -> Dict[str, Any]:
        return {
            'source': str(content.source_path),
            'timestamp': asyncio.get_event_loop().time(),
            'processed': True
        }
    
    # Utwórz pipeline
    pipeline = ProcessingPipeline()
    pipeline.add_stage("metadata", stage1_metadata)
    pipeline.add_stage("analysis", stage2_analysis)
    pipeline.add_stage("report", stage3_report)
    
    # Przetwórz
    result = await pipeline.process(
        directory=Path("./src"),
        patterns=["*.py"],
        job_name="pipeline_processing"
    )
    
    return result


# ============== PRZYKŁAD 5: Streaming wyników ==============

async def example_streaming():
    """Przetwarzanie w trybie streaming."""
    
    config = ProcessingConfig(
        batch_size=10,
        max_concurrent=5,
        enable_progress=False  # Wyłączamy dla streaming
    )
    
    processor = FileProcessor(config)
    scanner = FileScanner()
    
    # Zeskanuj pliki
    files = scanner.scan(Path("./src"))
    
    async def simple_processor(content: ExtractedContent) -> Dict[str, Any]:
        await asyncio.sleep(0.01)  # Symulacja pracy
        return {'file': str(content.source_path), 'length': len(content.text)}
    
    # Przetwarzaj w trybie streaming
    count = 0
    async for result in processor.process_streaming(files, simple_processor):
        if result.success:
            count += 1
            print(f"[{count}] {result.file_path.name}: OK")
        else:
            print(f"[{count}] {result.file_path.name}: ERROR - {result.error}")
    
    print(f"\nPrzetworzono {count} plików")


# ============== PRZYKŁAD 6: Zarządzanie jobami ==============

async def example_job_management():
    """Zarządzanie jobami i checkpointami."""
    
    checkpoint_dir = Path("./checkpoints")
    
    # Utwórz processor z checkpointami
    config = ProcessingConfig(
        use_checkpoints=True,
        checkpoint_dir=checkpoint_dir,
        enable_progress=True
    )
    
    processor = FileProcessor(config)
    
    async def dummy_processor(content: ExtractedContent) -> str:
        await asyncio.sleep(0.001)
        return "processed"
    
    # Uruchom przetwarzanie
    result = await processor.process_directory(
        directory=Path("./src"),
        processor=dummy_processor,
        patterns=["*.py"],
        job_name="managed_job"
    )
    
    print(f"Job ID: {result.job_id}")
    
    # Pobierz statystyki
    stats = await processor.get_job_stats(result.job_id)
    print(f"Stats: {json.dumps(stats, indent=2)}")
    
    # Lista wszystkich jobów
    jobs = await processor.list_jobs()
    print(f"\nWszystkie joby ({len(jobs)}):")
    for job in jobs:
        print(f"  - {job['job_id']}: {job['job_name']} ({job['status']})")
    
    return result


# ============== PRZYKŁAD 7: Custom scanner ==============

async def example_custom_scanner():
    """Niestandardowa konfiguracja skanera."""
    
    scan_config = ScanConfig(
        include_patterns=["*.py"],
        exclude_patterns=["test_*.py", "*_test.py"],
        exclude_hidden=True,
        exclude_dirs=['.git', '__pycache__', 'tests', 'docs'],
        max_file_size=10 * 1024 * 1024,  # 10 MB
        min_file_size=100,  # Min 100 bajtów
        follow_symlinks=False,
        detect_encoding=True,
        detect_binary=True,
        custom_filter=lambda p: not p.name.startswith('_')  # Dodatkowy filtr
    )
    
    scanner = FileScanner(scan_config)
    
    # Skanuj
    files = scanner.scan(Path("./src"))
    
    print(f"Znaleziono {len(files)} plików:")
    for file_info in files[:10]:  # Pokaż pierwsze 10
        print(f"  - {file_info.path.name} ({file_info.size} bajtów, {file_info.encoding})")
    
    return files


# ============== PRZYKŁAD 8: Wielokrotne katalogi ==============

async def example_multi_root():
    """Skanowanie wielu katalogów jednocześnie."""
    
    from agentswarm.processing import MultiRootScanner
    
    scan_config = ScanConfig(
        include_patterns=["*.py", "*.js", "*.ts"],
        exclude_dirs=['node_modules', '__pycache__']
    )
    
    multi_scanner = MultiRootScanner(scan_config)
    
    roots = [
        Path("./backend/src"),
        Path("./frontend/src"),
        Path("./shared")
    ]
    
    # Synchronicznie
    files = list(multi_scanner.scan_multiple(roots))
    
    print(f"Znaleziono {len(files)} plików w {len(roots)} katalogach")
    
    # Grupuj po rozszerzeniu
    by_ext = {}
    for f in files:
        ext = f.extension
        by_ext[ext] = by_ext.get(ext, 0) + 1
    
    print("\nPodział po rozszerzeniach:")
    for ext, count in sorted(by_ext.items()):
        print(f"  {ext}: {count}")
    
    return files


# ============== PRZYKŁAD 9: Retry i error handling ==============

async def example_retry_handling():
    """Obsługa błędów i retry."""
    
    from agentswarm.processing.batch_processor import retry_on_error
    
    attempt_count = {}
    
    @retry_on_error(max_retries=3, retry_delay=0.5)
    async def flaky_processor(content: ExtractedContent) -> Dict[str, Any]:
        path = str(content.source_path)
        attempt_count[path] = attempt_count.get(path, 0) + 1
        
        # Symulacja losowych błędów
        import random
        if random.random() < 0.3 and attempt_count[path] < 3:
            raise Exception(f"Losowy błąd (próba {attempt_count[path]})")
        
        return {
            'file': path,
            'attempts': attempt_count[path],
            'success': True
        }
    
    config = ProcessingConfig(
        max_retries=3,
        retry_delay=0.5,
        continue_on_error=True
    )
    
    processor = FileProcessor(config)
    
    result = await processor.process_directory(
        directory=Path("./src"),
        processor=flaky_processor,
        patterns=["*.py"]
    )
    
    print(f"Przetworzono: {result.processed_files}")
    print(f"Błędy: {result.failed_files}")
    
    return result


# ============== PRZYKŁAD 10: Eksport wyników ==============

async def example_export_results():
    """Eksport wyników do różnych formatów."""
    
    results = []
    
    async def collecting_processor(content: ExtractedContent) -> Dict[str, Any]:
        result = {
            'file': str(content.source_path),
            'size': len(content.text),
            'lines': len(content.text.splitlines()),
            'mime': content.mime_type,
            'encoding': content.encoding
        }
        results.append(result)
        return result
    
    result = await quick_process(
        directory=Path("./src"),
        processor=collecting_processor,
        patterns=["*.py"]
    )
    
    # Eksport do JSON
    json_file = Path("./results.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': result.to_dict(),
            'files': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Zapisano wyniki do {json_file}")
    
    # Eksport do CSV
    import csv
    csv_file = Path("./results.csv")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    print(f"Zapisano wyniki do {csv_file}")
    
    return results


# ============== GŁÓWNA FUNKCJA ==============

async def run_all_examples():
    """Uruchom wszystkie przykłady."""
    
    examples = [
        ("Basic Processing", example_basic_processing),
        ("Custom Scanner", example_custom_scanner),
        ("Multi Root", example_multi_root),
    ]
    
    for name, example_func in examples:
        print(f"\n{'='*60}")
        print(f"PRZYKŁAD: {name}")
        print('='*60)
        
        try:
            await example_func()
        except Exception as e:
            print(f"Błąd w przykładzie {name}: {e}")


if __name__ == "__main__":
    # Uruchom przykłady
    asyncio.run(run_all_examples())
