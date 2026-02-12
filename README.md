# AgentSwarm 🐝

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**AgentSwarm** to modułowa aplikacja CLI w Pythonie umożliwiająca równoległe przetwarzanie dużych zbiorów plików przy użyciu architektury agent swarm z dowolnym dostawcą LLM.

## 🚀 Szybki start

```bash
# Instalacja
pip install agentswarm

# Inicjalizacja
agentswarm init

# Konfiguracja OpenAI
agentswarm config set --provider openai

# Uruchomienie
agentswarm run "Przeanalizuj kod:" --input ./src --pattern "*.py" --output ./analysis.json
```

## ✨ Kluczowe cechy

- **🔌 Modularność**: Łatwe dodawanie nowych dostawców LLM przez wzorzec Factory
- **🔒 Bezpieczeństwo**: Bezpieczne przechowywanie kluczy API (keyring + szyfrowanie)
- **⚡ Skalowalność**: Async/multiprocessing dla przetwarzania batchowego
- **🎯 Elastyczność**: Wsparcie dla 7+ dostawców LLM
- **📊 Monitoring**: Podgląd statusu w czasie rzeczywistym
- **🔄 Wznawianie**: Checkpointy dla długich zadań

## 🏗️ Architektura

```
┌─────────────────────────────────────────────────────────────────┐
│                         AgentSwarm                               │
├─────────────────────────────────────────────────────────────────┤
│  CLI Layer          │  Core Layer         │  Provider Layer     │
│  ─────────          │  ─────────          │  ────────────       │
│  • init             │  • ConfigManager    │  • OpenAI           │
│  • config           │  • SecurityManager  │  • Anthropic        │
│  • run              │  • Exceptions       │  • Google           │
│  • status           │                     │  • Cohere           │
│                     │                     │  • Mistral          │
│                     │                     │  • Ollama           │
│                     │                     │  • Azure            │
├─────────────────────────────────────────────────────────────────┤
│  Swarm Layer        │  Processing Layer                         │
│  ───────────        │  ──────────────                           │
│  • Coordinator      │  • FileScanner                            │
│  • Workers          │  • BatchProcessor                         │
│  • TaskQueue        │  • ProgressTracker                        │
│  • ResultCollector  │  • FileHandlers                           │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Instalacja

### Podstawowa instalacja

```bash
pip install agentswarm
```

### Z konkretnymi dostawcami

```bash
# Tylko OpenAI
pip install agentswarm[openai]

# OpenAI i Anthropic
pip install agentswarm[openai,anthropic]

# Wszyscy dostawcy
pip install agentswarm[all]
```

### Instalacja deweloperska

```bash
git clone https://github.com/agentswarm/agentswarm.git
cd agentswarm
pip install -e ".[dev]"
```

## 🛠️ Konfiguracja

### Inicjalizacja

```bash
agentswarm init
```

Tworzy katalog konfiguracji w `~/.agentswarm/`.

### Konfiguracja dostawcy

```bash
# Interaktywna konfiguracja
agentswarm config set --provider openai

# Lista skonfigurowanych dostawców
agentswarm config list

# Test połączenia
agentswarm config test --provider openai

# Usunięcie konfiguracji
agentswarm config remove --provider openai
```

### Zmienne środowiskowe

Możesz również użyć zmiennych środowiskowych:

```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

## 🚀 Użycie

### Podstawowe użycie

```bash
agentswarm run "Przeanalizuj ten kod:" \
    --input ./src \
    --pattern "*.py" \
    --output ./analysis.json
```

### Zaawansowane opcje

```bash
agentswarm run @prompt.txt \
    --input ./project \
    --pattern "*.py" \
    --exclude "test_*,*_test.py" \
    --output ./results.json \
    --provider anthropic \
    --model claude-3-opus-20240229 \
    --workers 10 \
    --batch-size 20 \
    --recursive
```

### Prompt z pliku

Użyj `@` aby wczytać prompt z pliku:

```bash
agentswarm run @code_review_prompt.txt --input ./src --pattern "*.py"
```

### Podgląd statusu

```bash
# Jednorazowy status
agentswarm status

# Podgląd na żywo
agentswarm status --watch
```

## 📋 Przykłady użycia

### Analiza kodu

```bash
agentswarm run "Znajdź potencjalne błędy i zaproponuj poprawki:" \
    --input ./src \
    --pattern "*.py" \
    --workers 8 \
    --output ./code_review.json
```

### Tłumaczenie dokumentów

```bash
agentswarm run "Przetłumacz na polski zachowując formatowanie markdown:" \
    --input ./docs_en \
    --pattern "*.md" \
    --output ./docs_pl \
    --provider anthropic
```

### Ekstrakcja danych

```bash
agentswarm run @extract_entities.txt \
    --input ./data \
    --pattern "*.txt" \
    --output ./entities.json \
    --checkpoint-interval 50
```

### Wznawianie przerwanego zadania

```bash
agentswarm run @extract_entities.txt \
    --input ./data \
    --continue checkpoint_12345.json
```

## 🔌 Wspierani dostawcy LLM

| Dostawca | Streaming | Batch API | Lokalny |
|----------|-----------|-----------|---------|
| OpenAI | ✅ | ✅ | ❌ |
| Anthropic | ✅ | ❌ | ❌ |
| Google (Gemini) | ✅ | ❌ | ❌ |
| Cohere | ✅ | ❌ | ❌ |
| Mistral | ✅ | ❌ | ❌ |
| Ollama | ✅ | ❌ | ✅ |
| Azure OpenAI | ✅ | ✅ | ❌ |

## 🏗️ Dodawanie nowego dostawcy

1. Utwórz klasę providera:

```python
# src/agentswarm/providers/implementations/my_provider.py

from ..base import BaseLLMProvider, ProviderConfig, GenerationResult

class MyProvider(BaseLLMProvider):
    @property
    def name(self) -> str:
        return "myprovider"
    
    async def generate(self, prompt: str, **kwargs) -> GenerationResult:
        # Implementacja
        pass
    
    # ... pozostałe metody
```

2. Zarejestruj w factory:

```python
from agentswarm.providers import LLMProviderFactory
from .my_provider import MyProvider

factory = LLMProviderFactory.get_instance()
factory.register_provider("myprovider", MyProvider)
```

## 📊 Architektura Swarm

```
┌─────────────────────────────────────────────────────────────────┐
│                    SwarmCoordinator                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   TaskQueue (Priority)        Workers (Async)                   │
│   ┌─────────┐ ┌─────────┐     ┌─────────┐ ┌─────────┐          │
│   │ Task 1  │ │ Task 2  │────►│ Worker 1│ │ Worker 2│          │
│   │ (high)  │ │ (norm)  │     │ [LLM]   │ │ [LLM]   │          │
│   └─────────┘ └─────────┘     └────┬────┘ └────┬────┘          │
│                                    │           │                │
│                                    └─────┬─────┘                │
│                                          │                      │
│                                          ▼                      │
│                               ┌─────────────────┐              │
│                               │ ResultCollector │              │
│                               └─────────────────┘              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## ⚙️ Konfiguracja zaawansowana

### Plik konfiguracyjny

```json
{
  "version": "1.0.0",
  "default_provider": "openai",
  "max_workers": 4,
  "default_batch_size": 10,
  "request_timeout": 60,
  "retry_attempts": 3,
  "providers": {
    "openai": {
      "name": "openai",
      "default_model": "gpt-4o-mini",
      "temperature": 0.7,
      "max_tokens": 4096
    }
  }
}
```

### Zmienne środowiskowe

| Zmienna | Opis | Domyślna |
|---------|------|----------|
| `AGENTSWARM_LOG_LEVEL` | Poziom logowania | INFO |
| `AGENTSWARM_CONFIG_DIR` | Katalog konfiguracji | ~/.agentswarm |
| `AGENTSWARM_MAX_WORKERS` | Maksymalna liczba workerów | 4 |

## 🧪 Testowanie

```bash
# Uruchom wszystkie testy
pytest

# Testy jednostkowe
pytest tests/unit

# Testy integracyjne
pytest tests/integration

# Z pokryciem kodu
pytest --cov=agentswarm --cov-report=html
```

## 🤝 Wkład w projekt

1. Fork repozytorium
2. Utwórz branch (`git checkout -b feature/amazing-feature`)
3. Commit zmiany (`git commit -m 'Add amazing feature'`)
4. Push do brancha (`git push origin feature/amazing-feature`)
5. Otwórz Pull Request

## 📄 Licencja

Projekt jest dostępny na licencji MIT. Zobacz [LICENSE](LICENSE) dla szczegółów.

## 🙏 Podziękowania

- [OpenAI](https://openai.com/) za API GPT
- [Anthropic](https://anthropic.com/) za Claude
- [Typer](https://typer.tiangolo.com/) za framework CLI
- [Rich](https://rich.readthedocs.io/) za piękne wyjście terminala

## 📞 Wsparcie

- 📧 Email: support@agentswarm.dev
- 💬 Discord: [AgentSwarm Community](https://discord.gg/agentswarm)
- 🐛 Issues: [GitHub Issues](https://github.com/agentswarm/agentswarm/issues)

---

<p align="center">
  Made with ❤️ by the AgentSwarm Team
</p>
