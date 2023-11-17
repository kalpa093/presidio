from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

configuration = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "ko", "model_name": "ko_core_news_lg"},
               {"lang_code": "en", "model_name": "en_core_web_lg"},],
}

provider = NlpEngineProvider(nlp_configuration=configuration)
nlp_engine = provider.create_engine()

analyzer = AnalyzerEngine(
    nlp_engine=nlp_engine,
    supported_languages=["ko", "en"]
)
anonymizer = AnonymizerEngine()

text = "홍길동의 주민등록번호는 900101-1234567입니다. 전화번호는 010-1234-1234, 유선전화는 02-123-1234입니다. 신용카드번호는 5389-2000-5326-8619입니다. 건강보험번호는 5-1017360347입니다."

results = analyzer.analyze(text=text, language="ko")
print(text)
for result in results:
    print(result)

# 익명화
anonymized_text = anonymizer.anonymize(text, results)

# 익명화된 텍스트 출력
print("Anonymized Text:", anonymized_text.text)
