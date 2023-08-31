# presidio

To use Korean model, you have to call config file to NlpEngineProvider and pass it to Analyzerengine.

Example:
```python
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider

LANGUAGES_CONFIG_FILE = "./docs/analyzer/languages-config.yml"

provider = NlpEngineProvider(conf_file=LANGUAGES_CONFIG_FILE)
nlp_engine_with_korean = provider.create_engine()

analyzer = AnalyzerEngine(
  nlp_engine = nlp_engine_with_korean,
  supported_languages=["en", "ko"]
)

results_korean = analyzer.analyze(text="내 이름은 데이비드야", language="ko")
print(results_korean)

results_english = analyzer.analyze(text="My name is David", language="en")
print(results_english)
```
