"""
Testy modułu przetwarzania AgentSwarm.
"""

import asyncio
import tempfile
import json
from pathlib import Path
from typing import List
import unittest

from agentswarm.processing import (
    FileScanner,
    ScanConfig,
    FileInfo,
    ContentExtractor,
    ExtractedContent,
    CheckpointManager,
    JobCheckpoint,
    JobStatus,
    ProgressTracker,
    ProgressStats,
    BatchProcessor,
    BatchConfig,
    ProcessingResult,
    FileProcessor,
    ProcessingConfig,
)
from agentswarm.processing.file_scanner import EncodingDetector


class TestEncodingDetector(unittest.TestCase):
    """Testy wykrywania kodowania."""
    
    def test_detect_utf8(self):
        """Test wykrywania UTF-8."""
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False) as f:
            f.write("Hello World! Zażółć gęślą jaźń")
            temp_path = Path(f.name)
        
        try:
            encoding = EncodingDetector.detect_by_content(temp_path)
            self.assertEqual(encoding, 'utf-8')
        finally:
            temp_path.unlink()
    
    def test_detect_utf8_with_bom(self):
        """Test wykrywania UTF-8 z BOM."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as f:
            f.write(b'\xef\xbb\xbfHello World!')
            temp_path = Path(f.name)
        
        try:
            encoding = EncodingDetector.detect_by_content(temp_path)
            self.assertEqual(encoding, 'utf-8-sig')
        finally:
            temp_path.unlink()


class TestFileScanner(unittest.TestCase):
    """Testy skanera plików."""
    
    def setUp(self):
        """Przygotuj testowe pliki."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Utwórz strukturę katalogów
        (self.temp_path / "src").mkdir()
        (self.temp_path / "tests").mkdir()
        (self.temp_path / "docs").mkdir()
        (self.temp_path / ".hidden").mkdir()
        
        # Utwórz pliki
        (self.temp_path / "src" / "main.py").write_text("print('hello')", encoding='utf-8')
        (self.temp_path / "src" / "utils.py").write_text("def helper(): pass", encoding='utf-8')
        (self.temp_path / "tests" / "test_main.py").write_text("def test(): pass", encoding='utf-8')
        (self.temp_path / "docs" / "readme.md").write_text("# README", encoding='utf-8')
        (self.temp_path / ".hidden" / "secret.txt").write_text("secret", encoding='utf-8')
    
    def tearDown(self):
        """Posprzątaj po testach."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_scan_all_files(self):
        """Test skanowania wszystkich plików."""
        scanner = FileScanner()
        files = scanner.scan(self.temp_path)
        
        # Powinno znaleźć 4 pliki (bez ukrytego)
        self.assertEqual(len(files), 4)
        
        # Sprawdź nazwy
        names = {f.name for f in files}
        self.assertEqual(names, {'main.py', 'utils.py', 'test_main.py', 'readme.md'})
    
    def test_scan_with_pattern(self):
        """Test skanowania z patternem."""
        config = ScanConfig(include_patterns=["*.py"])
        scanner = FileScanner(config)
        files = scanner.scan(self.temp_path)
        
        # Powinno znaleźć 3 pliki .py
        self.assertEqual(len(files), 3)
        
        for f in files:
            self.assertEqual(f.extension, '.py')
    
    def test_scan_exclude_dirs(self):
        """Test wykluczania katalogów."""
        config = ScanConfig(exclude_dirs=['tests'])
        scanner = FileScanner(config)
        files = scanner.scan(self.temp_path)
        
        # Nie powinno być plików z tests/
        for f in files:
            self.assertNotIn('tests', f.path.parts)
    
    def test_scan_include_hidden(self):
        """Test włączania ukrytych plików."""
        config = ScanConfig(exclude_hidden=False)
        scanner = FileScanner(config)
        files = scanner.scan(self.temp_path)
        
        # Powinno znaleźć też ukryte pliki
        names = {f.name for f in files}
        self.assertIn('secret.txt', names)


class TestContentExtractor(unittest.TestCase):
    """Testy ekstraktora treści."""
    
    def setUp(self):
        """Przygotuj testowe pliki."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Utwórz pliki testowe
        (self.temp_path / "test.txt").write_text("Hello World!", encoding='utf-8')
        (self.temp_path / "test.py").write_text("def hello():\n    print('world')", encoding='utf-8')
        (self.temp_path / "test.json").write_text('{"key": "value"}', encoding='utf-8')
        (self.temp_path / "test.csv").write_text("a,b,c\n1,2,3", encoding='utf-8')
        (self.temp_path / "test.md").write_text("# Title\n\nContent", encoding='utf-8')
    
    def tearDown(self):
        """Posprzątaj po testach."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_extract_text(self):
        """Test ekstrakcji pliku tekstowego."""
        extractor = ContentExtractor()
        result = extractor.extract_sync(self.temp_path / "test.txt")
        
        self.assertTrue(result.success)
        self.assertEqual(result.text, "Hello World!")
        self.assertEqual(result.mime_type, 'text/plain')
    
    def test_extract_python(self):
        """Test ekstrakcji pliku Python."""
        extractor = ContentExtractor()
        result = extractor.extract_sync(self.temp_path / "test.py")
        
        self.assertTrue(result.success)
        self.assertIn('def hello()', result.text)
        self.assertEqual(result.metadata.get('language'), 'py')
    
    def test_extract_json(self):
        """Test ekstrakcji JSON."""
        extractor = ContentExtractor()
        result = extractor.extract_sync(self.temp_path / "test.json")
        
        self.assertTrue(result.success)
        self.assertIn('"key": "value"', result.text)
        self.assertEqual(result.mime_type, 'application/json')
    
    def test_extract_csv(self):
        """Test ekstrakcji CSV."""
        extractor = ContentExtractor()
        result = extractor.extract_sync(self.temp_path / "test.csv")
        
        self.assertTrue(result.success)
        self.assertEqual(result.metadata.get('rows'), 2)
        self.assertEqual(result.metadata.get('columns'), 3)
    
    def test_get_supported_extensions(self):
        """Test pobierania wspieranych rozszerzeń."""
        extractor = ContentExtractor()
        extensions = extractor.get_supported_extensions()
        
        self.assertIn('.txt', extensions)
        self.assertIn('.py', extensions)
        self.assertIn('.json', extensions)
        self.assertIn('.csv', extensions)


class TestCheckpointManager(unittest.TestCase):
    """Testy manager checkpointów."""
    
    def setUp(self):
        """Przygotuj tymczasowy katalog."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        self.manager = CheckpointManager(self.checkpoint_dir)
    
    def tearDown(self):
        """Posprzątaj po testach."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_create_job(self):
        """Test tworzenia jobu."""
        file_paths = [Path("/tmp/file1.txt"), Path("/tmp/file2.txt")]
        
        checkpoint = self.manager.create_job_sync("test_job", file_paths)
        
        self.assertEqual(checkpoint.job_name, "test_job")
        self.assertEqual(checkpoint.total_files, 2)
        self.assertEqual(len(checkpoint.file_checkpoints), 2)
    
    def test_save_and_load_checkpoint(self):
        """Test zapisywania i wczytywania checkpointu."""
        file_paths = [Path("/tmp/file1.txt")]
        
        # Utwórz i zapisz
        checkpoint = self.manager.create_job_sync("test_job", file_paths)
        job_id = checkpoint.job_id
        
        # Wyczyść z pamięci
        del self.manager._current_jobs[job_id]
        
        # Wczytaj
        loaded = self.manager.load_checkpoint_sync(job_id)
        
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.job_name, "test_job")
        self.assertEqual(loaded.total_files, 1)
    
    def test_update_file_status(self):
        """Test aktualizacji statusu pliku."""
        file_paths = [Path("/tmp/file1.txt")]
        checkpoint = self.manager.create_job_sync("test_job", file_paths)
        job_id = checkpoint.job_id
        
        # Aktualizuj status
        asyncio.run(self.manager.update_file_status(
            job_id, Path("/tmp/file1.txt"), 'completed'
        ))
        
        # Sprawdź
        updated = self.manager.load_checkpoint_sync(job_id)
        self.assertEqual(updated.processed_files, 1)
    
    def test_get_job_stats(self):
        """Test pobierania statystyk jobu."""
        file_paths = [Path("/tmp/file1.txt"), Path("/tmp/file2.txt")]
        checkpoint = self.manager.create_job_sync("test_job", file_paths)
        
        stats = self.manager.get_job_stats(checkpoint.job_id)
        
        self.assertIsNotNone(stats)
        self.assertEqual(stats['total_files'], 2)
        self.assertEqual(stats['progress_percentage'], 0.0)


class TestProgressTracker(unittest.TestCase):
    """Testy trackera postępu."""
    
    def test_progress_stats(self):
        """Test statystyk postępu."""
        stats = ProgressStats(total_files=100, total_bytes=10000)
        
        self.assertEqual(stats.progress_percentage, 0.0)
        
        # Symuluj postęp
        stats.processed_files = 50
        self.assertEqual(stats.progress_percentage, 50.0)
        
        stats.processed_files = 100
        self.assertEqual(stats.progress_percentage, 100.0)
    
    def test_progress_stats_formatting(self):
        """Test formatowania czasu."""
        stats = ProgressStats()
        
        # Test formatowania
        self.assertEqual(stats._format_duration(30), "30.0s")
        self.assertEqual(stats._format_duration(90), "1m 30s")
        self.assertEqual(stats._format_duration(3660), "1h 1m")
    
    def test_to_dict(self):
        """Test konwersji do słownika."""
        stats = ProgressStats(total_files=100, processed_files=50)
        
        d = stats.to_dict()
        
        self.assertEqual(d['total_files'], 100)
        self.assertEqual(d['processed_files'], 50)
        self.assertEqual(d['progress_percentage'], 50.0)


class TestBatchProcessor(unittest.TestCase):
    """Testy procesora batchy."""
    
    def setUp(self):
        """Przygotuj testowe pliki."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Utwórz pliki testowe
        for i in range(5):
            (self.temp_path / f"file{i}.txt").write_text(f"Content {i}", encoding='utf-8')
    
    def tearDown(self):
        """Posprzątaj po testach."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    async def _test_processor(self):
        """Test przetwarzania batcha."""
        from agentswarm.processing.file_scanner import FileScanner, FileInfo
        
        # Zeskanuj pliki
        scanner = FileScanner()
        files = scanner.scan(self.temp_path)
        
        # Prosty processor
        async def processor(content):
            return {'length': len(content.text)}
        
        # Przetwórz
        config = BatchConfig(batch_size=2, max_concurrent=2)
        batch_processor = BatchProcessor(config)
        
        results = await batch_processor.process_all(files, processor, "test_job")
        
        self.assertEqual(len(results), 3)  # 5 plików / batch_size 2 = 3 batch'e
        
        total_processed = sum(len(r.successful) for r in results)
        self.assertEqual(total_processed, 5)
    
    def test_batch_processing(self):
        """Uruchom test async."""
        asyncio.run(self._test_processor())


class TestFileProcessor(unittest.TestCase):
    """Testy głównego procesora plików."""
    
    def setUp(self):
        """Przygotuj testowe pliki."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Utwórz strukturę
        (self.temp_path / "src").mkdir()
        (self.temp_path / "src" / "main.py").write_text("print('hello')", encoding='utf-8')
        (self.temp_path / "src" / "utils.py").write_text("def helper(): pass", encoding='utf-8')
    
    def tearDown(self):
        """Posprzątaj po testach."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    async def _test_process_directory(self):
        """Test przetwarzania katalogu."""
        config = ProcessingConfig(
            use_checkpoints=False,
            enable_progress=False
        )
        
        processor = FileProcessor(config)
        
        async def simple_processor(content):
            return {'size': len(content.text)}
        
        result = await processor.process_directory(
            self.temp_path / "src",
            simple_processor,
            patterns=["*.py"]
        )
        
        self.assertEqual(result.total_files, 2)
        self.assertEqual(result.processed_files, 2)
    
    def test_process_directory(self):
        """Uruchom test async."""
        asyncio.run(self._test_process_directory())


class TestIntegration(unittest.TestCase):
    """Testy integracyjne."""
    
    def setUp(self):
        """Przygotuj testowe środowisko."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Utwórz strukturę projektu
        (self.temp_path / "src").mkdir()
        (self.temp_path / "tests").mkdir()
        (self.temp_path / "docs").mkdir()
        
        # Pliki Python
        (self.temp_path / "src" / "main.py").write_text("""
def main():
    print("Hello World")
    # TODO: add more features
    
if __name__ == "__main__":
    main()
""", encoding='utf-8')
        
        (self.temp_path / "src" / "utils.py").write_text("""
def helper():
    pass

def another_function():
    # FIXME: optimize this
    pass
""", encoding='utf-8')
        
        # Pliki testowe
        (self.temp_path / "tests" / "test_main.py").write_text("""
def test_main():
    assert True
""", encoding='utf-8')
        
        # Dokumentacja
        (self.temp_path / "docs" / "readme.md").write_text("""# Project

This is a test project.
""", encoding='utf-8')
    
    def tearDown(self):
        """Posprzątaj po testach."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    async def _test_full_workflow(self):
        """Test pełnego workflow."""
        from agentswarm.processing import quick_process
        
        results = []
        
        async def analyze_code(content):
            text = content.text
            result = {
                'file': str(content.source_path),
                'lines': len(text.splitlines()),
                'functions': text.count('def '),
                'todos': text.count('TODO'),
                'fixmes': text.count('FIXME')
            }
            results.append(result)
            return result
        
        summary = await quick_process(
            directory=self.temp_path / "src",
            processor=analyze_code,
            patterns=["*.py"],
            max_concurrent=2,
            batch_size=10
        )
        
        # Sprawdź wyniki
        self.assertEqual(summary.total_files, 2)
        self.assertEqual(summary.processed_files, 2)
        
        # Sprawdź zebrane dane
        self.assertEqual(len(results), 2)
        
        # Sprawdź czy policzono funkcje
        total_functions = sum(r['functions'] for r in results)
        self.assertEqual(total_functions, 3)  # main, helper, another_function
        
        # Sprawdź TODOs i FIXMEs
        total_todos = sum(r['todos'] for r in results)
        total_fixmes = sum(r['fixmes'] for r in results)
        self.assertEqual(total_todos, 1)
        self.assertEqual(total_fixmes, 1)
    
    def test_full_workflow(self):
        """Uruchom test integracyjny."""
        asyncio.run(self._test_full_workflow())


def run_tests():
    """Uruchom wszystkie testy."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Dodaj testy
    suite.addTests(loader.loadTestsFromTestCase(TestEncodingDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestFileScanner))
    suite.addTests(loader.loadTestsFromTestCase(TestContentExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestCheckpointManager))
    suite.addTests(loader.loadTestsFromTestCase(TestProgressTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestBatchProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestFileProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Uruchom
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
