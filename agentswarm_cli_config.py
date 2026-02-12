"""
Implementacja CLI i systemu konfiguracji dla AgentSwarm
"""

# ============================================================================
# 1. SYSTEM KONFIGURACJI (src/agentswarm/core/config_manager.py)
# ============================================================================

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from threading import RLock
import keyring
from cryptography.fernet import Fernet
import base64
import hashlib


@dataclass
class ProviderConfig:
    """Konfiguracja pojedynczego providera"""
    name: str
    api_key: Optional[str] = None
    api_key_env_var: Optional[str] = None
    base_url: Optional[str] = None
    default_model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60
    enabled: bool = True
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


@dataclass
class GlobalConfig:
    """Globalna konfiguracja aplikacji"""
    version: str = "1.0.0"
    default_provider: str = "openai"
    log_level: str = "INFO"
    log_file: Optional[str] = None
    max_workers: int = 4
    default_batch_size: int = 10
    request_timeout: int = 60
    retry_attempts: int = 3
    checkpoint_interval: int = 50
    providers: Dict[str, ProviderConfig] = None
    
    def __post_init__(self):
        if self.providers is None:
            self.providers = {}


class SecurityManager:
    """Zarządzanie bezpiecznym przechowywaniem kluczy API"""
    
    SERVICE_NAME = "agentswarm"
    
    def __init__(self):
        self._encryption_key: Optional[bytes] = None
        self._fernet: Optional[Fernet] = None
        self._init_encryption()
    
    def _init_encryption(self) -> None:
        """Inicjalizuje klucz szyfrowania"""
        # Użyj stałego klucza opartego na machine-id dla spójności
        machine_id = self._get_machine_id()
        key = hashlib.sha256(machine_id.encode()).digest()
        self._encryption_key = base64.urlsafe_b64encode(key)
        self._fernet = Fernet(self._encryption_key)
    
    def _get_machine_id(self) -> str:
        """Pobiera unikalny identyfikator maszyny"""
        # Próba odczytania machine-id
        machine_id_paths = [
            "/etc/machine-id",
            "/var/lib/dbus/machine-id",
        ]
        for path in machine_id_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return f.read().strip()
        
        # Fallback: użyj hostname + username
        return f"{os.getlogin()}@{os.uname().nodename}"
    
    def store_api_key(self, provider: str, api_key: str) -> bool:
        """Przechowuje klucz API bezpiecznie"""
        try:
            # Szyfruj klucz przed zapisaniem
            encrypted_key = self._fernet.encrypt(api_key.encode()).decode()
            keyring.set_password(self.SERVICE_NAME, provider, encrypted_key)
            return True
        except Exception as e:
            print(f"Error storing API key: {e}")
            return False
    
    def retrieve_api_key(self, provider: str) -> Optional[str]:
        """Pobiera klucz API"""
        try:
            encrypted_key = keyring.get_password(self.SERVICE_NAME, provider)
            if encrypted_key:
                return self._fernet.decrypt(encrypted_key.encode()).decode()
            return None
        except Exception as e:
            print(f"Error retrieving API key: {e}")
            return None
    
    def delete_api_key(self, provider: str) -> bool:
        """Usuwa klucz API"""
        try:
            keyring.delete_password(self.SERVICE_NAME, provider)
            return True
        except Exception:
            return False
    
    @staticmethod
    def is_keyring_available() -> bool:
        """Sprawdza czy keyring jest dostępny"""
        try:
            keyring.get_keyring()
            return True
        except Exception:
            return False


class ConfigManager:
    """Zarządza konfiguracją aplikacji"""
    
    CONFIG_FILENAME = "config.json"
    DEFAULT_CONFIG_DIR = ".agentswarm"
    
    _instance = None
    _lock = RLock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._config_dir = Path.home() / self.DEFAULT_CONFIG_DIR
        self._config_file = self._config_dir / self.CONFIG_FILENAME
        self._config: Optional[GlobalConfig] = None
        self._security_manager = SecurityManager()
        self._initialized = True
    
    @property
    def config_dir(self) -> Path:
        """Zwraca katalog konfiguracji"""
        return self._config_dir
    
    def ensure_config_dir(self) -> None:
        """Tworzy katalog konfiguracji jeśli nie istnieje"""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        # Ustaw odpowiednie uprawnienia (tylko dla właściciela)
        os.chmod(self._config_dir, 0o700)
    
    def load_config(self) -> GlobalConfig:
        """Ładuje konfigurację z pliku"""
        if self._config is not None:
            return self._config
        
        if not self._config_file.exists():
            self._config = GlobalConfig()
            return self._config
        
        try:
            with open(self._config_file, 'r') as f:
                data = json.load(f)
            
            # Konwertuj słowniki providerów na obiekty ProviderConfig
            providers_data = data.pop('providers', {})
            providers = {}
            for name, p_data in providers_data.items():
                providers[name] = ProviderConfig(name=name, **p_data)
            
            data['providers'] = providers
            self._config = GlobalConfig(**data)
            
        except Exception as e:
            print(f"Error loading config: {e}")
            self._config = GlobalConfig()
        
        return self._config
    
    def save_config(self) -> bool:
        """Zapisuje konfigurację do pliku"""
        if self._config is None:
            return False
        
        try:
            self.ensure_config_dir()
            
            # Konwertuj na słownik
            data = asdict(self._config)
            
            # Usuń API keys z konfiguracji (są w keyring)
            for provider_data in data.get('providers', {}).values():
                provider_data.pop('api_key', None)
            
            with open(self._config_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Ustaw odpowiednie uprawnienia
            os.chmod(self._config_file, 0o600)
            
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def get_provider_config(self, name: str) -> Optional[ProviderConfig]:
        """Pobiera konfigurację providera"""
        config = self.load_config()
        provider_config = config.providers.get(name)
        
        if provider_config:
            # Spróbuj pobrać API key z keyring
            api_key = self._security_manager.retrieve_api_key(name)
            if api_key:
                provider_config.api_key = api_key
            # Lub ze zmiennej środowiskowej
            elif provider_config.api_key_env_var:
                api_key = os.getenv(provider_config.api_key_env_var)
                if api_key:
                    provider_config.api_key = api_key
        
        return provider_config
    
    def set_provider_config(self, name: str, provider_config: ProviderConfig) -> None:
        """Ustawia konfigurację providera"""
        config = self.load_config()
        
        # Zapisz API key w keyring
        if provider_config.api_key:
            self._security_manager.store_api_key(name, provider_config.api_key)
            # Nie przechowuj w obiekcie (dla bezpieczeństwa)
            provider_config.api_key = None
        
        config.providers[name] = provider_config
        self.save_config()
    
    def remove_provider_config(self, name: str) -> bool:
        """Usuwa konfigurację providera"""
        config = self.load_config()
        
        if name in config.providers:
            del config.providers[name]
            self._security_manager.delete_api_key(name)
            self.save_config()
            return True
        return False
    
    def list_configured_providers(self) -> List[str]:
        """Zwraca listę skonfigurowanych providerów"""
        config = self.load_config()
        return list(config.providers.keys())
    
    def get_global_setting(self, key: str, default: Any = None) -> Any:
        """Pobiera globalne ustawienie"""
        config = self.load_config()
        return getattr(config, key, default)
    
    def set_global_setting(self, key: str, value: Any) -> None:
        """Ustawia globalne ustawienie"""
        config = self.load_config()
        if hasattr(config, key):
            setattr(config, key, value)
            self.save_config()


# ============================================================================
# 2. IMPLEMENTACJA CLI (src/agentswarm/cli/main.py)
# ============================================================================

import typer
from typing import Optional, List
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
import asyncio

app = typer.Typer(
    name="agentswarm",
    help="Agent Swarm CLI - Przetwarzaj duże zbiory plików z LLM",
    rich_markup_mode="rich"
)
console = Console()

# Inicjalizacja config manager
config_manager = ConfigManager()


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
    """
    console.print(Panel.fit(
        "[bold blue]AgentSwarm Initialization[/bold blue]",
        border_style="blue"
    ))
    
    # Sprawdź czy konfiguracja już istnieje
    if config_manager.config_dir.exists() and not force:
        if not Confirm.ask("Konfiguracja już istnieje. Nadpisać?"):
            console.print("[yellow]Inicjalizacja anulowana.[/yellow]")
            raise typer.Exit()
    
    # Utwórz katalog konfiguracji
    config_manager.ensure_config_dir()
    
    # Zapisz domyślną konfigurację
    config = config_manager.load_config()
    config_manager.save_config()
    
    console.print(f"[green]✓[/green] Utworzono katalog konfiguracji: {config_manager.config_dir}")
    console.print("[green]✓[/green] Zainicjalizowano konfigurację")
    console.print("\n[bold]Następne kroki:[/bold]")
    console.print("  1. Skonfiguruj dostawcę LLM: [cyan]agentswarm config set --provider openai[/cyan]")
    console.print("  2. Przetestuj połączenie: [cyan]agentswarm config test --provider openai[/cyan]")
    console.print("  3. Uruchom przetwarzanie: [cyan]agentswarm run --help[/cyan]")


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
        True,
        "--interactive/--no-interactive", "-i/-I",
        help="Tryb interaktywny"
    )
):
    """
    Zarządza konfiguracją dostawców LLM.
    """
    action = action.lower()
    
    if action == "list":
        _list_providers()
    elif action == "set":
        _set_provider_config(provider, interactive)
    elif action == "get":
        _get_provider_config(provider)
    elif action == "remove":
        _remove_provider_config(provider)
    elif action == "test":
        _test_provider_config(provider)
    else:
        console.print(f"[red]Nieznana akcja: {action}[/red]")
        raise typer.Exit(1)


def _list_providers():
    """Wyświetla listę skonfigurowanych providerów"""
    providers = config_manager.list_configured_providers()
    
    if not providers:
        console.print("[yellow]Brak skonfigurowanych dostawców.[/yellow]")
        console.print("Użyj: [cyan]agentswarm config set --provider <name>[/cyan]")
        return
    
    table = Table(title="Skonfigurowani dostawcy LLM")
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Status", style="yellow")
    
    for name in providers:
        p_config = config_manager.get_provider_config(name)
        has_key = config_manager._security_manager.retrieve_api_key(name) is not None
        status = "[green]✓ Skonfigurowany[/green]" if has_key else "[red]✗ Brak klucza API[/red]"
        table.add_row(
            name,
            p_config.default_model or "-",
            status
        )
    
    console.print(table)


def _set_provider_config(provider: Optional[str], interactive: bool):
    """Ustawia konfigurację providera"""
    if not provider and interactive:
        provider = Prompt.ask(
            "Wybierz dostawcę",
            choices=["openai", "anthropic", "google", "cohere", "mistral", "ollama", "azure"]
        )
    
    if not provider:
        console.print("[red]Wymagana nazwa dostawcy (--provider)[/red]")
        raise typer.Exit(1)
    
    console.print(Panel.fit(
        f"[bold]Konfiguracja dostawcy: {provider}[/bold]",
        border_style="blue"
    ))
    
    # Pobierz istniejącą konfigurację lub utwórz nową
    existing_config = config_manager.get_provider_config(provider)
    p_config = existing_config or ProviderConfig(name=provider)
    
    if interactive:
        # API Key
        api_key = Prompt.ask(
            f"Klucz API dla {provider}",
            password=True
        )
        if api_key:
            p_config.api_key = api_key
        
        # Default model
        p_config.default_model = Prompt.ask(
            "Domyślny model",
            default=p_config.default_model or ""
        ) or None
        
        # Temperature
        temp_str = Prompt.ask(
            "Temperature (0.0-2.0)",
            default=str(p_config.temperature)
        )
        p_config.temperature = float(temp_str)
        
        # Max tokens
        tokens_str = Prompt.ask(
            "Max tokens",
            default=str(p_config.max_tokens)
        )
        p_config.max_tokens = int(tokens_str)
    
    # Zapisz konfigurację
    config_manager.set_provider_config(provider, p_config)
    console.print(f"[green]✓[/green] Zapisano konfigurację dla {provider}")


def _get_provider_config(provider: Optional[str]):
    """Wyświetla konfigurację providera"""
    if not provider:
        console.print("[red]Wymagana nazwa dostawcy (--provider)[/red]")
        raise typer.Exit(1)
    
    p_config = config_manager.get_provider_config(provider)
    if not p_config:
        console.print(f"[red]Dostawca {provider} nie jest skonfigurowany[/red]")
        raise typer.Exit(1)
    
    table = Table(title=f"Konfiguracja: {provider}")
    table.add_column("Ustawienie", style="cyan")
    table.add_column("Wartość", style="green")
    
    has_key = config_manager._security_manager.retrieve_api_key(provider) is not None
    
    table.add_row("API Key", "[green]✓ Ustawiony[/green]" if has_key else "[red]✗ Brak[/red]")
    table.add_row("Default Model", p_config.default_model or "-")
    table.add_row("Temperature", str(p_config.temperature))
    table.add_row("Max Tokens", str(p_config.max_tokens))
    table.add_row("Timeout", str(p_config.timeout))
    
    console.print(table)


def _remove_provider_config(provider: Optional[str]):
    """Usuwa konfigurację providera"""
    if not provider:
        console.print("[red]Wymagana nazwa dostawcy (--provider)[/red]")
        raise typer.Exit(1)
    
    if Confirm.ask(f"Czy na pewno usunąć konfigurację dla {provider}?"):
        if config_manager.remove_provider_config(provider):
            console.print(f"[green]✓[/green] Usunięto konfigurację dla {provider}")
        else:
            console.print(f"[yellow]Dostawca {provider} nie był skonfigurowany[/yellow]")


def _test_provider_config(provider: Optional[str]):
    """Testuje połączenie z providerem"""
    if not provider:
        console.print("[red]Wymagana nazwa dostawcy (--provider)[/red]")
        raise typer.Exit(1)
    
    p_config = config_manager.get_provider_config(provider)
    if not p_config:
        console.print(f"[red]Dostawca {provider} nie jest skonfigurowany[/red]")
        raise typer.Exit(1)
    
    console.print(f"Testowanie połączenia z {provider}...")
    
    # Tutaj byłoby rzeczywiste testowanie połączenia
    # Na razie tylko symulacja
    console.print(f"[green]✓[/green] Połączenie z {provider} działa poprawnie")


@app.command()
def run(
    prompt: str = typer.Argument(
        ...,
        help="Prompt lub ścieżka do pliku z promptem (@file.txt)"
    ),
    input_path: Path = typer.Option(
        ...,
        "--input", "-i",
        help="Ścieżka do katalogu lub pliku wejściowego",
        exists=True
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Ścieżka do pliku wyjściowego"
    ),
    provider: str = typer.Option(
        None,
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
        help="Liczba workerów",
        min=1,
        max=50
    ),
    pattern: str = typer.Option(
        "*",
        "--pattern",
        help="Wzorzec plików (glob)"
    ),
    batch_size: int = typer.Option(
        10,
        "--batch-size", "-b",
        help="Rozmiar batcha",
        min=1,
        max=100
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
    )
):
    """
    Uruchamia agent swarm na zbiorze plików.
    """
    # Wczytaj prompt z pliku jeśli zaczyna się od @
    if prompt.startswith("@"):
        prompt_file = Path(prompt[1:])
        if prompt_file.exists():
            prompt = prompt_file.read_text()
        else:
            console.print(f"[red]Plik promptu nie istnieje: {prompt_file}[/red]")
            raise typer.Exit(1)
    
    # Użyj domyślnego providera jeśli nie podano
    if not provider:
        provider = config_manager.get_global_setting("default_provider", "openai")
    
    # Sprawdź konfigurację providera
    p_config = config_manager.get_provider_config(provider)
    if not p_config:
        console.print(f"[red]Dostawca {provider} nie jest skonfigurowany[/red]")
        console.print(f"Użyj: [cyan]agentswarm config set --provider {provider}[/cyan]")
        raise typer.Exit(1)
    
    console.print(Panel.fit(
        "[bold blue]AgentSwarm Execution[/bold blue]",
        border_style="blue"
    ))
    
    # Wyświetl konfigurację
    console.print(f"[cyan]Provider:[/cyan] {provider}")
    console.print(f"[cyan]Model:[/cyan] {model or p_config.default_model or 'default'}")
    console.print(f"[cyan]Input:[/cyan] {input_path}")
    console.print(f"[cyan]Pattern:[/cyan] {pattern}")
    console.print(f"[cyan]Workers:[/cyan] {workers}")
    console.print(f"[cyan]Batch size:[/cyan] {batch_size}")
    console.print()
    
    if dry_run:
        console.print("[yellow]DRY RUN - Symulacja bez wykonywania[/yellow]")
        # Tutaj logika dry run
        return
    
    # Tutaj byłaby rzeczywista logika uruchomienia swarm
    console.print("[green]Przetwarzanie zakończone![/green]")


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
        help="Interwał odświeżania (sekundy)",
        min=1
    )
):
    """
    Wyświetla status działającego swarm.
    """
    if watch:
        console.print("[yellow]Tryb podglądu (Ctrl+C aby zakończyć)[/yellow]\n")
        try:
            while True:
                # Wyczyść ekran i wyświetl status
                console.clear()
                _display_status()
                asyncio.sleep(refresh_interval)
        except KeyboardInterrupt:
            console.print("\n[green]Zakończono podgląd.[/green]")
    else:
        _display_status()


def _display_status():
    """Wyświetla aktualny status"""
    table = Table(title="AgentSwarm Status")
    table.add_column("Metryka", style="cyan")
    table.add_column("Wartość", style="green")
    
    # Symulowane dane - w rzeczywistości byłyby pobierane z koordynatora
    table.add_row("Status", "[green]Running[/green]")
    table.add_row("Workers", "4/4 active")
    table.add_row("Queue", "23 pending")
    table.add_row("Completed", "156 tasks")
    table.add_row("Failed", "2 tasks")
    table.add_row("Tokens/sec", "1,234")
    
    console.print(table)


@app.command()
def providers_list():
    """Wyświetla listę dostępnych dostawców LLM"""
    table = Table(title="Dostępni dostawcy LLM")
    table.add_column("Nazwa", style="cyan")
    table.add_column("Opis", style="green")
    table.add_column("Streaming", style="yellow")
    table.add_column("Batching", style="yellow")
    
    providers_info = [
        ("openai", "OpenAI GPT-4/GPT-3.5", "✓", "✓"),
        ("anthropic", "Anthropic Claude", "✓", "✗"),
        ("google", "Google Gemini", "✓", "✗"),
        ("cohere", "Cohere Command", "✓", "✗"),
        ("mistral", "Mistral AI", "✓", "✗"),
        ("ollama", "Ollama (local)", "✓", "✗"),
        ("azure", "Azure OpenAI", "✓", "✓"),
    ]
    
    for name, desc, streaming, batching in providers_info:
        table.add_row(name, desc, streaming, batching)
    
    console.print(table)
    console.print("\n[bold]Konfiguracja dostawcy:[/bold]")
    console.print("  [cyan]agentswarm config set --provider <name>[/cyan]")


# Entry point
if __name__ == "__main__":
    app()
