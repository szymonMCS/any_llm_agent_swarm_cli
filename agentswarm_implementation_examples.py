"""
Przykładowe implementacje kluczowych klas dla AgentSwarm
To są fragmenty kodu pokazujące implementację głównych komponentów.
"""

# ============================================================================
# 1. BAZOWY PROVIDER LLM (src/agentswarm/providers/base.py)
# ============================================================================

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, AsyncIterator
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import asyncio


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
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class GenerationResult(BaseModel):
    """Wynik generacji"""
    content: str
    tokens_used: int = 0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    finish_reason: str = "stop"
    model: str = ""
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
    async def validate_config(self) -> tuple[bool, Optional[str]]:
        """Waliduje konfigurację i testuje połączenie"""
        pass
    
    async def close(self) -> None:
        """Zamyka połączenia i zwalnia zasoby"""
        if self._client:
            await self._client.close()


# ============================================================================
# 2. FACTORY I REJESTR PROVIDERÓW (src/agentswarm/providers/factory.py)
# ============================================================================

from typing import Type, Dict
import asyncio


class LLMProviderFactory:
    """Factory do tworzenia instancji providerów LLM"""
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry: Dict[str, Type[BaseLLMProvider]] = {}
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> 'LLMProviderFactory':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def register_provider(self, name: str, provider_class: Type[BaseLLMProvider]) -> None:
        """Rejestruje nowego dostawcę"""
        if not issubclass(provider_class, BaseLLMProvider):
            raise ValueError(f"Provider class must inherit from BaseLLMProvider")
        self._registry[name.lower()] = provider_class
    
    def create_provider(self, name: str, config: ProviderConfig) -> BaseLLMProvider:
        """Tworzy instancję providera"""
        name = name.lower()
        if name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise ValueError(f"Unknown provider '{name}'. Available: {available}")
        
        provider_class = self._registry[name]
        return provider_class(config)
    
    def list_available_providers(self) -> List[str]:
        """Zwraca listę dostępnych providerów"""
        return list(self._registry.keys())
    
    def is_provider_available(self, name: str) -> bool:
        """Sprawdza czy provider jest dostępny"""
        return name.lower() in self._registry


# ============================================================================
# 3. PRZYKŁADOWY PROVIDER - OpenAI (src/agentswarm/providers/implementations/openai_provider.py)
# ============================================================================

class OpenAIProvider(BaseLLMProvider):
    """Implementacja providera OpenAI"""
    
    DEFAULT_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo"
    ]
    
    @property
    def name(self) -> str:
        return "openai"
    
    @property
    def display_name(self) -> str:
        return "OpenAI"
    
    @property
    def supported_models(self) -> List[str]:
        return self.DEFAULT_MODELS
    
    @property
    def max_context_length(self) -> int:
        model_limits = {
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16385
        }
        return model_limits.get(self.config.model, 4096)
    
    @property
    def supports_streaming(self) -> bool:
        return True
    
    @property
    def supports_batching(self) -> bool:
        return True
    
    async def initialize(self) -> None:
        try:
            import openai
            api_key = self.config.api_key or self._get_api_key_from_env()
            self._client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
            self._initialized = True
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    def _get_api_key_from_env(self) -> str:
        import os
        env_var = self.config.api_key_env_var or "OPENAI_API_KEY"
        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(f"OpenAI API key not found. Set {env_var} environment variable.")
        return api_key
    
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        if not self._initialized:
            await self.initialize()
        
        import time
        start_time = time.time()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self._client.chat.completions.create(
                model=self.config.model or "gpt-4o-mini",
                messages=messages,
                temperature=kwargs.get('temperature', self.config.temperature),
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                **self.config.extra_params
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            return GenerationResult(
                content=response.choices[0].message.content,
                tokens_used=response.usage.total_tokens,
                tokens_prompt=response.usage.prompt_tokens,
                tokens_completion=response.usage.completion_tokens,
                finish_reason=response.choices[0].finish_reason,
                model=response.model,
                latency_ms=latency_ms
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")
    
    async def generate_batch(
        self, 
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[GenerationResult]:
        """Generuje odpowiedzi dla batcha promptów równolegle"""
        semaphore = asyncio.Semaphore(kwargs.get('max_concurrent', 5))
        
        async def generate_with_limit(prompt: str) -> GenerationResult:
            async with semaphore:
                return await self.generate(prompt, system_prompt, **kwargs)
        
        tasks = [generate_with_limit(p) for p in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def validate_config(self) -> tuple[bool, Optional[str]]:
        try:
            if not self._initialized:
                await self.initialize()
            
            # Testowe wywołanie API
            response = await self._client.chat.completions.create(
                model=self.config.model or "gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return True, None
        except Exception as e:
            return False, str(e)


# ============================================================================
# 4. MODELE SWARM (src/agentswarm/swarm/models.py)
# ============================================================================

from enum import Enum
from pathlib import Path


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
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
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
class WorkerStats:
    """Statystyki workera"""
    worker_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_tokens: int = 0
    avg_latency_ms: float = 0.0
    current_task: Optional[str] = None
    is_active: bool = False


import uuid


# ============================================================================
# 5. KOLEJKA ZADAŃ (src/agentswarm/swarm/task_queue.py)
# ============================================================================

class TaskQueue:
    """Asynchroniczna kolejka zadań z priorytetami"""
    
    def __init__(self, max_size: int = 1000):
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_size)
        self._task_map: Dict[str, Task] = {}
        self._lock = asyncio.Lock()
        self._task_counter = 0
    
    async def put(self, task: Task, priority: Optional[int] = None) -> None:
        """Dodaje zadanie do kolejki"""
        if priority is not None:
            task.priority = TaskPriority(priority)
        
        async with self._lock:
            self._task_counter += 1
            # (priority, counter, task) - counter zapewnia FIFO dla tego samego priorytetu
            await self._queue.put((task.priority.value, self._task_counter, task))
            self._task_map[task.id] = task
            task.status = TaskStatus.PENDING
    
    async def get(self) -> Task:
        """Pobiera zadanie z kolejki"""
        _, _, task = await self._queue.get()
        async with self._lock:
            task.status = TaskStatus.ASSIGNED
        return task
    
    def task_done(self) -> None:
        """Oznacza zadanie jako zakończone"""
        self._queue.task_done()
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Zwraca status zadania"""
        async with self._lock:
            task = self._task_map.get(task_id)
            return task.status if task else None
    
    def get_pending_count(self) -> int:
        """Zwraca liczbę oczekujących zadań"""
        return self._queue.qsize()
    
    async def cancel_task(self, task_id: str) -> bool:
        """Anuluje zadanie (tylko jeśli jeszcze nie jest przetwarzane)"""
        async with self._lock:
            task = self._task_map.get(task_id)
            if task and task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                return True
            return False


# ============================================================================
# 6. WORKER (src/agentswarm/swarm/worker.py)
# ============================================================================

import logging

logger = logging.getLogger(__name__)


class SwarmWorker:
    """Worker wykonujący zadania z kolejki"""
    
    def __init__(
        self,
        worker_id: str,
        provider: BaseLLMProvider,
        task_queue: TaskQueue,
        result_callback: callable
    ):
        self.worker_id = worker_id
        self.provider = provider
        self.task_queue = task_queue
        self.result_callback = result_callback
        self._stats = WorkerStats(worker_id=worker_id)
        self._current_task: Optional[Task] = None
        self._shutdown_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Domyślnie nie wstrzymany
    
    async def run(self) -> None:
        """Główna pętla workera"""
        logger.info(f"Worker {self.worker_id} started")
        
        while not self._shutdown_event.is_set():
            # Czekaj jeśli wstrzymany
            await self._pause_event.wait()
            
            try:
                # Pobierz zadanie z kolejki (z timeoutem)
                task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )
                
                # Przetwórz zadanie
                result = await self.process_task(task)
                
                # Wyślij wynik
                await self.result_callback(result)
                
                # Oznacz jako zakończone
                self.task_queue.task_done()
                
            except asyncio.TimeoutError:
                # Brak zadań w kolejce, kontynuuj
                continue
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
        
        logger.info(f"Worker {self.worker_id} stopped")
    
    async def process_task(self, task: Task) -> Result:
        """Przetwarza pojedyncze zadanie"""
        self._current_task = task
        task.assigned_worker = self.worker_id
        task.started_at = datetime.utcnow()
        task.status = TaskStatus.PROCESSING
        
        try:
            # Wywołaj LLM
            generation_result = await asyncio.wait_for(
                self.provider.generate(
                    prompt=task.prompt,
                    system_prompt=task.system_prompt
                ),
                timeout=task.timeout_seconds
            )
            
            # Utwórz wynik
            result = Result(
                task_id=task.id,
                success=True,
                content=generation_result.content,
                tokens_used=generation_result.tokens_used,
                latency_ms=generation_result.latency_ms,
                worker_id=self.worker_id,
                retry_count=task.retry_count
            )
            
            # Aktualizuj statystyki
            self._stats.tasks_completed += 1
            self._stats.total_tokens += generation_result.tokens_used
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            
        except asyncio.TimeoutError:
            result = Result(
                task_id=task.id,
                success=False,
                error_message="Task timeout",
                worker_id=self.worker_id,
                retry_count=task.retry_count
            )
            task.status = TaskStatus.TIMEOUT
            self._stats.tasks_failed += 1
            
        except Exception as e:
            result = Result(
                task_id=task.id,
                success=False,
                error_message=str(e),
                worker_id=self.worker_id,
                retry_count=task.retry_count
            )
            task.status = TaskStatus.FAILED
            self._stats.tasks_failed += 1
        
        finally:
            self._current_task = None
        
        return result
    
    def pause(self) -> None:
        """Wstrzymuje workera"""
        self._pause_event.clear()
    
    def resume(self) -> None:
        """Wznawia workera"""
        self._pause_event.set()
    
    def shutdown(self) -> None:
        """Zamyka workera"""
        self._shutdown_event.set()
    
    def get_stats(self) -> WorkerStats:
        """Zwraca statystyki workera"""
        return self._stats
    
    def is_busy(self) -> bool:
        """Czy worker aktualnie przetwarza zadanie"""
        return self._current_task is not None


# ============================================================================
# 7. KOORDYNATOR (src/agentswarm/swarm/coordinator.py)
# ============================================================================

from typing import Callable


class ResultCollector:
    """Kolekcjoner wyników"""
    
    def __init__(self):
        self._results: List[Result] = []
        self._lock = asyncio.Lock()
        self._result_event = asyncio.Event()
    
    async def add_result(self, result: Result) -> None:
        """Dodaje wynik"""
        async with self._lock:
            self._results.append(result)
            self._result_event.set()
    
    async def get_results(self) -> List[Result]:
        """Zwraca wszystkie wyniki"""
        async with self._lock:
            return self._results.copy()
    
    async def wait_for_results(self, count: int, timeout: Optional[float] = None) -> List[Result]:
        """Czeka na określoną liczbę wyników"""
        async with self._lock:
            if len(self._results) >= count:
                return self._results[:count]
        
        try:
            await asyncio.wait_for(
                self._wait_for_count(count),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            pass
        
        async with self._lock:
            return self._results.copy()
    
    async def _wait_for_count(self, count: int) -> None:
        while True:
            await self._result_event.wait()
            async with self._lock:
                if len(self._results) >= count:
                    return
            self._result_event.clear()


class SwarmCoordinator:
    """Koordynator zarządzający agent swarm"""
    
    def __init__(
        self, 
        config: SwarmConfig,
        provider: BaseLLMProvider
    ):
        self.config = config
        self.provider = provider
        self._task_queue = TaskQueue(max_size=config.max_queue_size)
        self._result_collector = ResultCollector()
        self._workers: List[SwarmWorker] = []
        self._worker_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._is_running = False
    
    async def initialize(self) -> None:
        """Inicjalizuje wszystkie komponenty swarm"""
        # Inicjalizuj providera
        await self.provider.initialize()
        
        # Utwórz workery
        for i in range(self.config.worker_count):
            worker = SwarmWorker(
                worker_id=f"worker-{i+1}",
                provider=self.provider,  # W produkcji: kopia lub osobna instancja
                task_queue=self._task_queue,
                result_callback=self._result_collector.add_result
            )
            self._workers.append(worker)
    
    async def start(self) -> None:
        """Uruchamia koordynator i workery"""
        if self._is_running:
            raise RuntimeError("Swarm is already running")
        
        self._is_running = True
        
        # Uruchom workery jako osobne taski
        for worker in self._workers:
            task = asyncio.create_task(worker.run())
            self._worker_tasks.append(task)
        
        logger.info(f"Swarm started with {len(self._workers)} workers")
    
    async def submit_task(self, task: Task) -> str:
        """Dodaje pojedyncze zadanie do kolejki"""
        await self._task_queue.put(task)
        return task.id
    
    async def submit_batch(self, tasks: List[Task]) -> List[str]:
        """Dodaje batch zadań do kolejki"""
        task_ids = []
        for task in tasks:
            task_id = await self.submit_task(task)
            task_ids.append(task_id)
        return task_ids
    
    async def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[Result]:
        """Pobiera wynik zadania"""
        # Czekaj na wynik
        start_time = asyncio.get_event_loop().time()
        
        while True:
            results = await self._result_collector.get_results()
            for result in results:
                if result.task_id == task_id:
                    return result
            
            if timeout:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    return None
            
            await asyncio.sleep(0.1)
    
    async def get_all_results(self) -> List[Result]:
        """Pobiera wszystkie wyniki"""
        return await self._result_collector.get_results()
    
    async def wait_for_completion(self) -> List[Result]:
        """Czeka na zakończenie wszystkich zadań"""
        # Czekaj na opróżnienie kolejki
        await self._task_queue._queue.join()
        
        # Poczekaj chwilę na przetworzenie ostatnich wyników
        await asyncio.sleep(0.5)
        
        return await self.get_all_results()
    
    async def pause(self) -> None:
        """Wstrzymuje przetwarzanie nowych zadań"""
        self._pause_event.clear()
        for worker in self._workers:
            worker.pause()
        logger.info("Swarm paused")
    
    async def resume(self) -> None:
        """Wznawia przetwarzanie"""
        self._pause_event.set()
        for worker in self._workers:
            worker.resume()
        logger.info("Swarm resumed")
    
    async def shutdown(self, graceful: bool = True, timeout: float = 30.0) -> None:
        """Zamyka koordynator i workery"""
        if not self._is_running:
            return
        
        logger.info("Shutting down swarm...")
        
        if graceful:
            # Czekaj na zakończenie oczekujących zadań
            try:
                await asyncio.wait_for(
                    self._task_queue._queue.join(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning("Graceful shutdown timeout, forcing...")
        
        # Zamknij workery
        for worker in self._workers:
            worker.shutdown()
        
        # Anuluj taski workerów
        for task in self._worker_tasks:
            task.cancel()
        
        # Poczekaj na zakończenie
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        # Zamknij providera
        await self.provider.close()
        
        self._is_running = False
        logger.info("Swarm shutdown complete")


# ============================================================================
# 8. PRZYKŁAD UŻYCIA
# ============================================================================

async def example_usage():
    """Przykład użycia AgentSwarm"""
    
    # 1. Konfiguracja providera
    config = ProviderConfig(
        api_key="your-api-key-here",
        model="gpt-4o-mini",
        temperature=0.7
    )
    
    # 2. Utworzenie providera przez factory
    factory = LLMProviderFactory.get_instance()
    factory.register_provider("openai", OpenAIProvider)
    
    provider = factory.create_provider("openai", config)
    
    # 3. Konfiguracja swarm
    swarm_config = SwarmConfig(
        worker_count=4,
        max_queue_size=100,
        task_timeout=60
    )
    
    # 4. Utworzenie i uruchomienie koordynatora
    coordinator = SwarmCoordinator(swarm_config, provider)
    await coordinator.initialize()
    await coordinator.start()
    
    # 5. Dodanie zadań
    tasks = [
        Task(
            prompt=f"Przeanalizuj ten tekst: Przykład {i}",
            system_prompt="Jesteś pomocnym asystentem.",
            priority=TaskPriority.NORMAL
        )
        for i in range(20)
    ]
    
    task_ids = await coordinator.submit_batch(tasks)
    print(f"Submitted {len(task_ids)} tasks")
    
    # 6. Czekanie na wyniki
    results = await coordinator.wait_for_completion()
    print(f"Completed {len(results)} tasks")
    
    # 7. Wyświetlenie wyników
    for result in results:
        status = "✓" if result.success else "✗"
        print(f"{status} Task {result.task_id}: {result.content[:50]}...")
    
    # 8. Zamknięcie
    await coordinator.shutdown()


# Uruchomienie przykładu
if __name__ == "__main__":
    asyncio.run(example_usage())
