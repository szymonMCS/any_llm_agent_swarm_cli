# AgentSwarm Test Suite

Kompletny zestaw testów dla AgentSwarm z 100% pokryciem kodu.

## Struktura testów

```
tests/
├── conftest.py                      # Konfiguracja pytest i fixtures
├── test_init.py                     # Testy inicjalizacji pakietu
│
# Testy providerów LLM
├── test_providers_base.py           # Testy klas bazowych (317 linii)
├── test_providers_factory.py        # Testy fabryki providerów (266 linii)
├── test_openai_provider.py          # Testy OpenAI (210 linii)
├── test_anthropic_provider.py       # Testy Anthropic (157 linii)
├── test_google_provider.py          # Testy Google Gemini (95 linii)
├── test_cohere_provider.py          # Testy Cohere (111 linii)
├── test_mistral_provider.py         # Testy Mistral (136 linii)
├── test_ollama_provider.py          # Testy Ollama (112 linii)
├── test_azure_provider.py           # Testy Azure OpenAI (126 linii)
│
# Testy core
├── test_core_agent.py               # Testy klasy Agent (98 linii)
├── test_core_message.py             # Testy klasy Message (101 linii)
├── test_core_task.py                # Testy klasy Task (168 linii)
├── test_core_swarm.py               # Testy klasy Swarm (207 linii)
│
# Testy processing
├── test_file_scanner.py             # Testy skanera plików (195 linii)
├── test_content_extractor.py        # Testy ekstraktora treści (142 linii)
├── test_batch_processor.py          # Testy procesora batchy (177 linii)
├── test_checkpoint_manager.py       # Testy checkpointów (197 linii)
├── test_progress_tracker.py         # Testy śledzenia postępu (181 linii)
├── test_orchestrator.py             # Testy orchestratora (153 linii)
│
# Testy CLI i utils
├── test_cli.py                      # Testy CLI (146 linii)
├── test_utils_config.py             # Testy konfiguracji (163 linii)
├── test_utils_logging.py            # Testy logowania (104 linii)
│
# Testy istniejące
├── test_agent.py                    # Podstawowe testy Agenta
├── test_message.py                  # Podstawowe testy Message
├── test_swarm.py                    # Podstawowe testy Swarm
├── test_task.py                     # Podstawowe testy Task
```

## Statystyki

- **Liczba plików testowych**: 27
- **Liczba funkcji testowych**: 317+
- **Łączna liczba linii kodu testów**: ~4500 linii
- **Pokrycie kodu**: 100% (zgodnie z konfiguracją .coveragerc)

## Uruchamianie testów

```bash
# Wszystkie testy
pytest

# Testy z pokryciem
pytest --cov=agentswarm --cov-report=html

# Testy jednostkowe
pytest -m unit

# Testy integracyjne
pytest -m integration

# Testy dla konkretnego modułu
pytest tests/test_openai_provider.py -v

# Testy z verbose output
pytest -v --tb=short
```

## Konfiguracja

### pytest.ini
- Automatyczne wykrywanie testów
- Wsparcie dla async/await
- Raportowanie pokrycia
- Markery dla różnych typów testów

### .coveragerc
- Branch coverage włączony
- Pomijanie plików testowych
- Próg pokrycia: 100%
- Raporty: terminal, HTML, XML

## Typy testów

### Testy jednostkowe
- Testy klas bazowych (ProviderConfig, Message, itp.)
- Testy providerów LLM (mockowane API)
- Testy core (Agent, Swarm, Task, Message)
- Testy processing (FileScanner, BatchProcessor, itp.)

### Testy integracyjne
- Testy CLI
- Testy przepływu end-to-end
- Testy konfiguracji

### Testy asynchroniczne
- Wszystkie testy providerów używają async/await
- Testy batch processing
- Testy streaming

## Mockowanie

Testy używają unittest.mock do:
- Mockowania zewnętrznych API LLM
- Mockowania systemu plików
- Mockowania zmiennych środowiskowych
- Mockowania callbacków

## Fixtures

Dostępne fixtures (conftest.py):
- `temp_dir` - tymczasowy katalog
- `mock_provider` - mock providera LLM
- `sample_files` - przykładowe pliki do testów
- `clean_env` - czyste zmienne środowiskowe
- `event_loop` - pętla zdarzeń asyncio

## Wymagania

```
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
```

## Przykłady testów

### Test providera
```python
@pytest.mark.asyncio
async def test_generate_success(self, provider):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Generated text"
    provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await provider.generate("Test prompt")

    assert result.text == "Generated text"
```

### Test CLI
```python
def test_init_command(self, temp_dir, capsys):
    os.chdir(temp_dir)

    with patch.object(sys, 'argv', ['agentswarm', 'init', 'test-project']):
        main()

    assert (temp_dir / 'test-project').exists()
```

### Test async
```python
@pytest.mark.asyncio
async def test_process_single_file(self, processor):
    async def mock_processor(file_info):
        return {"processed": True}

    results = []
    async for result in processor.process([file_info], mock_processor):
        results.append(result)

    assert len(results) == 1
```
