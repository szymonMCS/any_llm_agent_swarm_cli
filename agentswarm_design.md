# AgentSwarm - Architektura Aplikacji CLI

## Spis treści
1. [Przegląd projektu](#przegląd-projektu)
2. [Struktura katalogów](#struktura-katalogów)
3. [Diagram klas](#diagram-klas)
4. [Komponenty i interfejsy](#komponenty-i-interfejsy)
5. [Przepływ danych](#przepływ-danych)
6. [Instalacja i użycie](#instalacja-i-użycie)

---

## Przegląd projektu

**AgentSwarm** to modułowa aplikacja CLI w Pythonie umożliwiająca równoległe przetwarzanie dużych zbiorów plików przy użyciu architektury agent swarm z dowolnym dostawcą LLM.

### Kluczowe cechy
- **Modularność**: Łatwe dodawanie nowych dostawców LLM przez wzorzec Factory
- **Bezpieczeństwo**: Bezpieczne przechowywanie kluczy API (keyring + szyfrowanie)
- **Skalowalność**: Async/multiprocessing dla przetwarzania batchowego
- **Elastyczność**: Wsparcie dla 7+ dostawców LLM (OpenAI, Anthropic, Google, Cohere, Mistral, Ollama, Azure)

---

## Struktura katalogów

```
agentswarm/
├── pyproject.toml              # Konfiguracja pakietu (PEP 621)
├── README.md
├── LICENSE
├── .gitignore
│
├── src/
│   └── agentswarm/
│       ├── __init__.py         # Wersja i eksporty główne
│       ├── __main__.py         # Entry point: python -m agentswarm
│       │
│       ├── cli/                # Interfejs wiersza poleceń
│       │   ├── __init__.py
│       │   ├── main.py         # Główna grupa CLI (Typer/Click)
│       │   ├── commands/
│       │   │   ├── __init__.py
│       │   │   ├── init.py     # Komenda: agentswarm init
│       │   │   ├── config.py   # Komenda: agentswarm config
│       │   │   ├── run.py      # Komenda: agentswarm run
│       │   │   └── status.py   # Komenda: agentswarm status
│       │   └── utils.py        # Pomocniki CLI (formatowanie, walidacja)
│       │
│       ├── core/               # Rdzeń aplikacji
│       │   ├── __init__.py
│       │   ├── config_manager.py    # Zarządzanie konfiguracją
│       │   ├── security_manager.py  # Bezpieczne przechowywanie kluczy
│       │   └── exceptions.py        # Własne wyjątki
│       │
│       ├── providers/          # Dostawcy LLM (wzorzec Strategy + Factory)
│       │   ├── __init__.py
│       │   ├── base.py         # Abstract Base Class dla providerów
│       │   ├── factory.py      # LLMProviderFactory
│       │   ├── registry.py     # Rejestr dostawców
│       │   └── implementations/
│       │       ├── __init__.py
│       │       ├── openai_provider.py
│       │       ├── anthropic_provider.py
│       │       ├── google_provider.py
│       │       ├── cohere_provider.py
│       │       ├── mistral_provider.py
│       │       ├── ollama_provider.py
│       │       └── azure_provider.py
│       │
│       ├── swarm/              # Architektura Agent Swarm
│       │   ├── __init__.py
│       │   ├── coordinator.py  # Koordynator zadań
│       │   ├── worker.py       # Worker agent
│       │   ├── task_queue.py   # Kolejka zadań (asyncio.Queue)
│       │   ├── result_collector.py  # Agregator wyników
│       │   └── models.py       # Pydantic models (Task, Result, AgentConfig)
│       │
│       ├── processing/         # Przetwarzanie plików
│       │   ├── __init__.py
│       │   ├── file_scanner.py      # Skanowanie katalogów
│       │   ├── batch_processor.py   # Batch processing z async/await
│       │   ├── file_handlers.py     # Handlery różnych typów plików
│       │   └── progress_tracker.py  # Śledzenie postępu (rich/tqdm)
│       │
│       └── utils/              # Narzędzia pomocnicze
│           ├── __init__.py
│           ├── logging_config.py    # Konfiguracja logowania
│           ├── validators.py        # Walidatory
│           └── async_utils.py       # Narzędzia async
│
├── tests/                      # Testy
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_providers.py
│       │   ├── test_swarm.py
│       │   ├── test_processing.py
│       │   └── test_config.py
│       └── integration/
│           └── test_end_to_end.py
│
├── docs/                       # Dokumentacja
│   ├── architecture.md
│   ├── providers.md
│   └── examples/
│
└── examples/                   # Przykłady użycia
    ├── batch_code_review.py
    ├── document_translation.py
    └── data_extraction.py
```

---

## Diagram klas

### 1. Hierarchia Providerów LLM (Strategy Pattern)

```
┌─────────────────────────────────────────────────────────────────┐
│                    <<abstract>>                                 │
│                 BaseLLMProvider                                 │
├─────────────────────────────────────────────────────────────────┤
│ - config: ProviderConfig                                        │
│ - client: Any                                                   │
├─────────────────────────────────────────────────────────────────┤
│ + __init__(config: ProviderConfig)                              │
│ + @abstractmethod async generate(prompt: str) -> str            │
│ + @abstractmethod async generate_batch(prompts: List[str])      │
│                      -> List[str]                               │
│ + @abstractmethod validate_config() -> bool                     │
│ + @abstractmethod get_model_list() -> List[str]                 │
│ + @property @abstractmethod name() -> str                       │
│ + @property @abstractmethod max_tokens() -> int                 │
│ + @property @abstractmethod supports_streaming() -> bool        │
└─────────────────────────────────────────────────────────────────┘
                              △
                              │ implements
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────┴───────┐    ┌────────┴────────┐   ┌──────┴──────┐
│OpenAIProvider │    │AnthropicProvider│   │GoogleProvider│
├───────────────┤    ├─────────────────┤   ├─────────────┤
│- openai_client│    │- anthropic_client│  │- genai_client│
├───────────────┤    ├─────────────────┤   ├─────────────┤
│+ generate()   │    │+ generate()     │   │+ generate() │
│+ generate_batch│   │+ generate_batch()│  │+ generate_batch()│
│+ validate()   │    │+ validate()     │   │+ validate() │
└───────────────┘    └─────────────────┘   └─────────────┘
        │
        │     ┌─────────────────┐   ┌─────────────┐   ┌─────────────┐
        └────►│  CohereProvider │   │MistralProvider│  │OllamaProvider│
              ├─────────────────┤   ├─────────────┤   ├─────────────┤
              │+ generate()     │   │+ generate() │   │+ generate() │
              └─────────────────┘   └─────────────┘   └─────────────┘
```

### 2. Factory Pattern dla Providerów

```
┌─────────────────────────────────────────────────────────────────┐
│                 LLMProviderFactory                               │
├─────────────────────────────────────────────────────────────────┤
│ - _registry: Dict[str, Type[BaseLLMProvider]]                   │
│ - _lock: asyncio.Lock                                           │
├─────────────────────────────────────────────────────────────────┤
│ + register_provider(name: str, provider_class: Type)            │
│ + create_provider(name: str, config: ProviderConfig)            │
│                      -> BaseLLMProvider                         │
│ + list_available_providers() -> List[str]                       │
│ + get_provider_info(name: str) -> ProviderInfo                  │
│ + @classmethod get_instance() -> LLMProviderFactory             │
└─────────────────────────────────────────────────────────────────┘
                              │ uses
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 ProviderRegistry (singleton)                     │
├─────────────────────────────────────────────────────────────────┤
│ - _instance: ProviderRegistry                                   │
│ - _providers: Dict[str, ProviderEntry]                          │
├─────────────────────────────────────────────────────────────────┤
│ + auto_discover_providers()                                     │
│ + get_provider_class(name: str) -> Type[BaseLLMProvider]        │
│ + register_builtin_providers()                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Agent Swarm Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SwarmCoordinator                              │
├─────────────────────────────────────────────────────────────────┤
│ - config: SwarmConfig                                           │
│ - provider: BaseLLMProvider                                     │
│ - task_queue: TaskQueue                                         │
│ - workers: List[SwarmWorker]                                    │
│ - result_collector: ResultCollector                             │
│ - _semaphore: asyncio.Semaphore                                 │
│ - _shutdown_event: asyncio.Event                                │
├─────────────────────────────────────────────────────────────────┤
│ + __init__(config: SwarmConfig, provider: BaseLLMProvider)      │
│ + async initialize_workers(count: int)                          │
│ + async submit_task(task: Task) -> TaskId                       │
│ + async submit_batch(tasks: List[Task]) -> List[TaskId]         │
│ + async run_until_complete() -> SwarmResult                     │
│ + async get_status() -> SwarmStatus                             │
│ + async shutdown(graceful: bool = True)                         │
│ + async pause() / resume()                                      │
└─────────────────────────────────────────────────────────────────┘
                              │ manages
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SwarmWorker                                   │
├─────────────────────────────────────────────────────────────────┤
│ - worker_id: str                                                │
│ - provider: BaseLLMProvider                                     │
│ - task_queue: TaskQueue                                         │
│ - result_collector: ResultCollector                             │
│ - _current_task: Optional[Task]                                 │
│ - _task_count: int                                              │
│ - _error_count: int                                             │
├─────────────────────────────────────────────────────────────────┤
│ + async run()  # Main worker loop                               │
│ + async process_task(task: Task) -> Result                      │
│ + get_stats() -> WorkerStats                                    │
│ + is_busy() -> bool                                             │
└─────────────────────────────────────────────────────────────────┘
                              │ produces/consumes
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TaskQueue (asyncio.Queue wrapper)             │
├─────────────────────────────────────────────────────────────────┤
│ - _queue: asyncio.PriorityQueue[Task]                           │
│ - _task_map: Dict[TaskId, Task]                                 │
│ - _lock: asyncio.Lock                                           │
├─────────────────────────────────────────────────────────────────┤
│ + async put(task: Task, priority: int = 5)                      │
│ + async get() -> Task                                           │
│ + async mark_done(task_id: TaskId)                              │
│ + get_pending_count() -> int                                    │
│ + get_task_status(task_id: TaskId) -> TaskStatus                │
└─────────────────────────────────────────────────────────────────┘
```

### 4. System Konfiguracji

```
┌─────────────────────────────────────────────────────────────────┐
│                 ConfigManager                                    │
├─────────────────────────────────────────────────────────────────┤
│ - _config_path: Path                                            │
│ - _config: GlobalConfig                                         │
│ - _security_manager: SecurityManager                            │
│ - _lock: threading.RLock                                        │
├─────────────────────────────────────────────────────────────────┤
│ + load_config() -> GlobalConfig                                 │
│ + save_config()                                                 │
│ + get_provider_config(name: str) -> ProviderConfig              │
│ + set_provider_config(name: str, config: ProviderConfig)        │
│ + get_global_setting(key: str) -> Any                           │
│ + set_global_setting(key: str, value: Any)                      │
│ + validate_all_configs() -> ValidationResult                    │
│ + @property config_dir() -> Path                                │
└─────────────────────────────────────────────────────────────────┘
                              │ uses
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 SecurityManager                                  │
├─────────────────────────────────────────────────────────────────┤
│ - _keyring_backend: Any                                         │
│ - _encryption_key: bytes                                        │
│ - _service_name: str = "agentswarm"                             │
├─────────────────────────────────────────────────────────────────┤
│ + store_api_key(provider: str, api_key: str)                    │
│ + retrieve_api_key(provider: str) -> Optional[str]              │
│ + delete_api_key(provider: str)                                 │
│ + rotate_encryption_key()                                       │
│ + @staticmethod is_keyring_available() -> bool                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5. Przetwarzanie Plików

```
┌─────────────────────────────────────────────────────────────────┐
│                 BatchProcessor                                   │
├─────────────────────────────────────────────────────────────────┤
│ - coordinator: SwarmCoordinator                                 │
│ - file_scanner: FileScanner                                     │
│ - progress_tracker: ProgressTracker                             │
│ - max_batch_size: int                                           │
│ - max_concurrent_batches: int                                   │
├─────────────────────────────────────────────────────────────────┤
│ + async process_directory(                                      │
│       path: Path,                                               │
│       file_pattern: str,                                        │
│       prompt_template: str,                                     │
│       output_handler: OutputHandler                             │
│   ) -> ProcessingResult                                         │
│ + async process_file_list(                                      │
│       files: List[Path],                                        │
│       prompt_template: str                                      │
│   ) -> ProcessingResult                                         │
│ + pause() / resume() / cancel()                                 │
└─────────────────────────────────────────────────────────────────┘
                              │ uses
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 FileScanner                                      │
├─────────────────────────────────────────────────────────────────┤
│ - include_patterns: List[str]                                   │
│ - exclude_patterns: List[str]                                   │
│ - max_file_size: int                                            │
│ - follow_symlinks: bool                                         │
├─────────────────────────────────────────────────────────────────┤
│ + scan(path: Path) -> AsyncGenerator[FileInfo, None]            │
│ + scan_batch(path: Path, batch_size: int)                       │
│       -> AsyncGenerator[List[FileInfo], None]                   │
│ + estimate_total_size(path: Path) -> Tuple[int, int]            │
│ + validate_file(file_path: Path) -> ValidationResult            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Komponenty i interfejsy

### 1. Interfejs Providera LLM (BaseLLMProvider)

```python
# src/agentswarm/providers/base.py

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, AsyncIterator
from dataclasses import dataclass
from pydantic import BaseModel


class ProviderConfig(BaseModel):
    """Konfiguracja dostawcy LLM"""
    api_key: Optional[str] = None
    api_key_env_var: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "default"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60
    retry_attempts: int = 3
    retry_delay: float = 1.0
    extra_params: Dict[str, Any] = {}


class GenerationResult(BaseModel):
    """Wynik generacji"""
    content: str
    tokens_used: int
    tokens_prompt: int
    tokens_completion: int
    finish_reason: str
    model: str
    latency_ms: float
    metadata: Dict[str, Any] = {}


class BaseLLMProvider(ABC):
    """Abstrakcyjna klasa bazowa dla wszystkich providerów LLM"""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self._client = None
        self._initialized = False
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Nazwa dostawcy (np. 'openai', 'anthropic')"""
        pass
    
    @property
    @abstractmethod
    def display_name(self) -> str:
        """Czytelna nazwa dostawcy (np. 'OpenAI')"""
        pass
    
    @property
    @abstractmethod
    def supported_models(self) -> List[str]:
        """Lista obsługiwanych modeli"""
        pass
    
    @property
    @abstractmethod
    def max_context_length(self) -> int:
        """Maksymalna długość kontekstu w tokenach"""
        pass
    
    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Czy dostawca wspiera streaming"""
        pass
    
    @property
    @abstractmethod
    def supports_batching(self) -> bool:
        """Czy dostawca wspiera batch API"""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Inicjalizacja klienta API"""
        pass
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """Generuje odpowiedź dla pojedynczego promptu"""
        pass
    
    @abstractmethod
    async def generate_batch(
        self, 
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[GenerationResult]:
        """Generuje odpowiedzi dla batcha promptów"""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Streamuje odpowiedź token po tokenie"""
        pass
    
    @abstractmethod
    async def validate_config(self) -> tuple[bool, Optional[str]]:
        """Waliduje konfigurację i testuje połączenie"""
        pass
    
    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Zlicza tokeny w tekście"""
        pass
    
    async def close(self) -> None:
        """Zamyka połączenia i zwalnia zasoby"""
        if self._client:
            await self._client.close()
```

### 2. Interfejs Koordynatora Swarm

```python
# src/agentswarm/swarm/coordinator.py

from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel
import asyncio
from datetime import datetime


class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class Task(BaseModel):
    """Reprezentacja zadania do wykonania"""
    id: str
    prompt: str
    system_prompt: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_worker: Optional[str] = None
    max_retries: int = 3
    retry_count: int = 0
    timeout_seconds: int = 120
    dependencies: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Result(BaseModel):
    """Wynik wykonania zadania"""
    task_id: str
    success: bool
    content: Optional[str] = None
    error_message: Optional[str] = None
    tokens_used: int = 0
    latency_ms: float = 0.0
    worker_id: Optional[str] = None
    retry_count: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class SwarmConfig:
    """Konfiguracja swarm"""
    worker_count: int = 4
    max_queue_size: int = 1000
    task_timeout: int = 120
    retry_failed_tasks: bool = True
    max_retries: int = 3
    batch_size: int = 10
    enable_progress_tracking: bool = True
    save_partial_results: bool = True
    partial_results_interval: int = 10


@dataclass
class SwarmStatus:
    """Status swarm"""
    is_running: bool
    is_paused: bool
    workers_total: int
    workers_active: int
    workers_idle: int
    queue_pending: int
    tasks_total: int
    tasks_completed: int
    tasks_failed: int
    tasks_in_progress: int
    avg_latency_ms: float
    tokens_per_second: float


class SwarmCoordinator:
    """Koordynator zarządzający agent swarm"""
    
    def __init__(
        self, 
        config: SwarmConfig,
        provider_factory: Callable[[], BaseLLMProvider]
    ):
        self.config = config
        self._provider_factory = provider_factory
        self._task_queue: Optional[TaskQueue] = None
        self._result_collector: Optional[ResultCollector] = None
        self._workers: List[SwarmWorker] = []
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._shutdown_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._status = SwarmStatus(...)
    
    async def initialize(self) -> None:
        """Inicjalizuje wszystkie komponenty swarm"""
        pass
    
    async def start(self) -> None:
        """Uruchamia koordynator i workery"""
        pass
    
    async def submit_task(self, task: Task) -> str:
        """Dodaje pojedyncze zadanie do kolejki"""
        pass
    
    async def submit_batch(
        self, 
        tasks: List[Task],
        callback: Optional[Callable[[Result], None]] = None
    ) -> List[str]:
        """Dodaje batch zadań do kolejki"""
        pass
    
    async def get_result(self, task_id: str, timeout: Optional[float] = None) -> Result:
        """Pobiera wynik zadania (opcjonalnie czeka)"""
        pass
    
    async def get_all_results(self) -> List[Result]:
        """Pobiera wszystkie wyniki"""
        pass
    
    async def get_status(self) -> SwarmStatus:
        """Zwraca aktualny status swarm"""
        pass
    
    async def pause(self) -> None:
        """Wstrzymuje przetwarzanie nowych zadań"""
        pass
    
    async def resume(self) -> None:
        """Wznawia przetwarzanie"""
        pass
    
    async def cancel_task(self, task_id: str) -> bool:
        """Anuluje zadanie"""
        pass
    
    async def cancel_all(self) -> None:
        """Anuluje wszystkie oczekujące zadania"""
        pass
    
    async def shutdown(self, graceful: bool = True, timeout: float = 30.0) -> None:
        """Zamyka koordynator i workery"""
        pass
```

### 3. Interfejs CLI

```python
# src/agentswarm/cli/main.py

import typer
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="agentswarm",
    help="Agent Swarm CLI - Przetwarzaj duże zbiory plików z LLM",
    rich_markup_mode="rich"
)
console = Console()


@app.command()
def init(
    config_dir: Optional[Path] = typer.Option(
        None, 
        "--config-dir", "-c",
        help="Katalog konfiguracji (domyślnie: ~/.agentswarm)"
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Nadpisz istniejącą konfigurację"
    )
):
    """
    Inicjalizuje konfigurację AgentSwarm.
    
    Tworzy katalog konfiguracji i ustawia domyślne wartości.
    """
    pass


@app.command()
def config(
    action: str = typer.Argument(
        ...,
        help="Akcja: set, get, list, remove, test"
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider", "-p",
        help="Nazwa dostawcy LLM"
    ),
    key: Optional[str] = typer.Option(
        None,
        "--key", "-k",
        help="Klucz konfiguracji"
    ),
    value: Optional[str] = typer.Option(
        None,
        "--value", "-v",
        help="Wartość do ustawienia"
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive", "-i",
        help="Tryb interaktywny"
    )
):
    """
    Zarządza konfiguracją dostawców LLM.
    
    Examples:
        agentswarm config list
        agentswarm config set --provider openai --key api_key
        agentswarm config test --provider anthropic
    """
    pass


@app.command()
def run(
    prompt: str = typer.Argument(
        ...,
        help="Prompt lub ścieżka do pliku z promptem"
    ),
    input_path: Path = typer.Option(
        ...,
        "--input", "-i",
        help="Ścieżka do katalogu lub pliku wejściowego"
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Ścieżka do pliku wyjściowego"
    ),
    provider: str = typer.Option(
        "openai",
        "--provider", "-p",
        help="Dostawca LLM do użycia"
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="Model do użycia"
    ),
    workers: int = typer.Option(
        4,
        "--workers", "-w",
        help="Liczba workerów"
    ),
    pattern: str = typer.Option(
        "*",
        "--pattern",
        help="Wzorzec plików (glob)"
    ),
    batch_size: int = typer.Option(
        10,
        "--batch-size", "-b",
        help="Rozmiar batcha"
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        help="Przeszukuj podkatalogi"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Symulacja bez wykonywania"
    ),
    continue_from: Optional[str] = typer.Option(
        None,
        "--continue",
        help="Kontynuuj od checkpointu"
    )
):
    """
    Uruchamia agent swarm na zbiorze plików.
    
    Examples:
        agentswarm run "Przeanalizuj kod:" --input ./src --pattern "*.py"
        agentswarm run @prompt.txt --input ./docs --output ./results.json
    """
    pass


@app.command()
def status(
    watch: bool = typer.Option(
        False,
        "--watch", "-w",
        help="Tryb podglądu na żywo"
    ),
    refresh_interval: int = typer.Option(
        2,
        "--refresh",
        help="Interwał odświeżania (sekundy)"
    )
):
    """
    Wyświetla status działającego swarm.
    """
    pass


@app.command()
def providers(
    list_all: bool = typer.Option(
        False,
        "--list", "-l",
        help="Lista wszystkich dostawców"
    ),
    test: Optional[str] = typer.Option(
        None,
        "--test",
        help="Przetestuj konkretnego dostawcę"
    )
):
    """
    Zarządza dostawcami LLM.
    """
    pass
```

### 4. Modele danych (Pydantic)

```python
# src/agentswarm/swarm/models.py

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pathlib import Path


class FileInfo(BaseModel):
    """Informacje o pliku"""
    path: Path
    size_bytes: int
    modified_at: datetime
    checksum: Optional[str] = None
    mime_type: Optional[str] = None
    encoding: str = "utf-8"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessingConfig(BaseModel):
    """Konfiguracja przetwarzania"""
    input_path: Path
    output_path: Optional[Path] = None
    file_pattern: str = "*"
    exclude_patterns: List[str] = Field(default_factory=lambda: [
        "*.tmp", "*.log", ".git", "__pycache__", ".env"
    ])
    recursive: bool = True
    follow_symlinks: bool = False
    max_file_size_mb: int = 100
    max_files: Optional[int] = None
    encoding: str = "utf-8"
    chunk_size: int = 8192
    
    @validator('max_file_size_mb')
    def validate_max_size(cls, v):
        if v < 0 or v > 1000:
            raise ValueError('max_file_size_mb musi być między 0 a 1000')
        return v


class GlobalConfig(BaseModel):
    """Globalna konfiguracja aplikacji"""
    version: str = "1.0.0"
    default_provider: str = "openai"
    default_model: Optional[str] = None
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    config_dir: Path = Field(default_factory=lambda: Path.home() / ".agentswarm")
    checkpoint_interval: int = 10
    max_concurrent_requests: int = 10
    request_timeout: int = 60
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict)
    
    class Config:
        env_prefix = "AGENTSWARM_"


class Checkpoint(BaseModel):
    """Punkt kontrolny dla wznawiania przetwarzania"""
    id: str
    created_at: datetime
    total_files: int
    processed_files: int
    failed_files: List[str]
    results: List[Result]
    config: ProcessingConfig
    provider_config: ProviderConfig
```

---

## Przepływ danych

### 1. Przepływ inicjalizacji

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   User       │────►│  agentswarm init │────►│ ConfigManager   │
│   CLI        │     │                  │     │                 │
└──────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                              ┌────────────────────────┼────────────────────────┐
                              │                        │                        │
                              ▼                        ▼                        ▼
                       ┌─────────────┐         ┌──────────────┐        ┌─────────────┐
                       │ Create dirs │         │ Setup logging│        │ Init config │
                       │ ~/.agentswarm│        │              │        │   files     │
                       └─────────────┘         └──────────────┘        └─────────────┘
```

### 2. Przepływ konfiguracji providera

```
┌──────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│   User       │────►│ agentswarm config   │────►│  Interactive     │
│   CLI        │     │   set --provider X  │     │  Configuration   │
└──────────────┘     └─────────────────────┘     └────────┬─────────┘
                                                          │
                              ┌───────────────────────────┼────────────────┐
                              │                           │                │
                              ▼                           ▼                ▼
                       ┌─────────────┐           ┌───────────────┐  ┌──────────────┐
                       │ Prompt for  │           │ Validate      │  │ Store in     │
                       │ API key     │           │ connection    │  │ keyring      │
                       └─────────────┘           └───────────────┘  └──────────────┘
```

### 3. Przepływ przetwarzania (run)

```
┌─────────────┐
│ agentswarm  │
│    run      │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INITIALIZATION                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Load Config  │─►│ Init Provider│─►│Create Task   │─►│Init Coordinator│   │
│  │              │  │   Factory    │  │   Queue      │  │   & Workers   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FILE DISCOVERY                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Scan Dir     │─►│ Apply Filters│─►│ Group into   │─►│ Create File  │   │
│  │ (async)      │  │ (pattern)    │  │ batches      │  │   Tasks      │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SWARM EXECUTION                                    │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    TaskQueue (Priority Queue)                        │   │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │   │
│   │  │ Task 1  │  │ Task 2  │  │ Task 3  │  │ Task 4  │  │  ...    │   │   │
│   │  │(high)   │  │(normal) │  │(normal) │  │(low)    │  │         │   │   │
│   │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘   │   │
│   └───────┼────────────┼────────────┼────────────┼────────────┼────────┘   │
│           │            │            │            │            │             │
│           ▼            ▼            ▼            ▼            ▼             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        SwarmWorkers                                  │   │
│   │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐          │   │
│   │  │ Worker 1│    │ Worker 2│    │ Worker 3│    │ Worker 4│  ...     │   │
│   │  │[LLM]    │    │[LLM]    │    │[LLM]    │    │[LLM]    │          │   │
│   │  │Provider │    │Provider │    │Provider │    │Provider │          │   │
│   │  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘          │   │
│   └───────┼──────────────┼──────────────┼──────────────┼───────────────┘   │
│           │              │              │              │                    │
│           └──────────────┴──────────────┴──────────────┘                    │
│                          │                                                  │
│                          ▼                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     ResultCollector                                  │   │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │   │
│   │  │ Result 1│  │ Result 2│  │ Result 3│  │ Result 4│  │  ...    │   │   │
│   │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Aggregate    │─►│ Format Output│─►│ Write to File│─►│ Save Checkpt │   │
│  │ Results      │  │ (json/csv)   │  │              │  │              │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4. Przepływ wykonania zadania (Worker)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Worker Task Execution                                │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────┐
    │  START   │
    └────┬─────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Fetch Task from │────►│ Validate Task   │────►│ Check Dependencies│
│     Queue       │     │                 │     │                  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                              ┌─────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Load File       │
                    │ Content         │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Build Prompt    │
                    │ (template +     │
                    │  context)       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐     ┌─────────────────┐
                    │ Call LLM        │────►│ Retry Logic     │
                    │ Provider        │     │ (exponential    │
                    │ (async)         │◄────│  backoff)       │
                    └────────┬────────┘     └─────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
       ┌────────────┐ ┌────────────┐ ┌────────────┐
       │  Success   │ │   Retry    │ │   Fail     │
       │            │ │  (n<max)   │ │ (n>=max)   │
       └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
             │              │              │
             ▼              └──────────────┘
    ┌─────────────────┐                    │
    │ Create Result   │                    ▼
    │ Object          │           ┌─────────────────┐
    │                 │           │ Create Error    │
    │ - content       │           │ Result          │
    │ - tokens_used   │           │                 │
    │ - latency       │           │ - error_message │
    │ - metadata      │           │ - retry_count   │
    └────────┬────────┘           └────────┬────────┘
             │                             │
             └─────────────┬───────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Send to Result  │
                  │   Collector     │
                  └────────┬────────┘
                           │
                           ▼
                    ┌────────────┐
                    │   LOOP     │
                    │  (next task)│
                    └────────────┘
```

---

## Instalacja i użycie

### Instalacja

```bash
# Instalacja z PyPI
pip install agentswarm

# Instalacja z konkretnymi providerami
pip install agentswarm[openai,anthropic]
pip install agentswarm[all]  # Wszyscy dostawcy

# Instalacja deweloperska
git clone https://github.com/user/agentswarm.git
cd agentswarm
pip install -e ".[dev]"
```

### Szybki start

```bash
# 1. Inicjalizacja
agentswarm init

# 2. Konfiguracja OpenAI
agentswarm config set --provider openai --key api_key
# Wprowadź swój klucz API gdy zostaniesz poproszony

# 3. Test połączenia
agentswarm config test --provider openai

# 4. Uruchomienie przetwarzania
agentswarm run "Przeanalizuj ten kod i znajdź błędy:" \
    --input ./src \
    --pattern "*.py" \
    --output ./analysis.json \
    --workers 8
```

### Przykłady użycia

```bash
# Analiza kodu z wieloma workerami
agentswarm run @code_review_prompt.txt \
    --input ./project \
    --pattern "*.py" \
    --exclude "test_*,*_test.py" \
    --workers 10 \
    --batch-size 20 \
    --output ./review_results.json

# Tłumaczenie dokumentów
agentswarm run "Przetłumacz na polski:" \
    --input ./docs_en \
    --pattern "*.md" \
    --output ./docs_pl \
    --provider anthropic \
    --model claude-3-opus-20240229

# Ekstrakcja danych z checkpointiem
agentswarm run @extract_entities.txt \
    --input ./data \
    --pattern "*.txt" \
    --output ./entities.json \
    --checkpoint-interval 50

# Wznawianie przerwanego zadania
agentswarm run @extract_entities.txt \
    --input ./data \
    --continue checkpoint_12345.json

# Podgląd statusu w czasie rzeczywistym
agentswarm status --watch
```

---

## Dodawanie nowego dostawcy LLM

Aby dodać nowego dostawcę LLM, wystarczy:

1. **Utworzyć klasę providera** w `src/agentswarm/providers/implementations/`:

```python
# src/agentswarm/providers/implementations/new_provider.py

from ..base import BaseLLMProvider, ProviderConfig, GenerationResult

class NewProvider(BaseLLMProvider):
    @property
    def name(self) -> str:
        return "newprovider"
    
    @property
    def display_name(self) -> str:
        return "New Provider"
    
    async def initialize(self) -> None:
        # Inicjalizacja klienta
        self._client = NewProviderClient(api_key=self.config.api_key)
        self._initialized = True
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> GenerationResult:
        # Implementacja generacji
        pass
    
    # ... pozostałe metody abstrakcyjne
```

2. **Zarejestrować providera** w `src/agentswarm/providers/registry.py`:

```python
from .implementations.new_provider import NewProvider

def register_builtin_providers(factory: LLMProviderFactory):
    factory.register_provider("newprovider", NewProvider)
```

3. **Dodać zależność** do `pyproject.toml`:

```toml
[project.optional-dependencies]
newprovider = ["newprovider-sdk>=1.0.0"]
all = ["newprovider-sdk>=1.0.0", ...]
```

---

## Podsumowanie architektury

| Komponent | Wzorzec | Odpowiedzialność |
|-----------|---------|------------------|
| BaseLLMProvider | Strategy | Abstrakcja dla wszystkich LLM |
| LLMProviderFactory | Factory | Tworzenie instancji providerów |
| SwarmCoordinator | Mediator | Koordynacja workerów |
| SwarmWorker | Worker | Wykonywanie zadań |
| TaskQueue | Queue | Zarządzanie kolejką zadań |
| ConfigManager | Singleton | Zarządzanie konfiguracją |
| SecurityManager | Singleton | Bezpieczne przechowywanie kluczy |
| BatchProcessor | Facade | Uproszczony interfejs przetwarzania |

---

*Dokument wygenerowany: 2024*
*Wersja: 1.0.0*
