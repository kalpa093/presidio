from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
import csv

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

label = []
data = []
with open('fake_addresses_label.csv', 'r', encoding='cp949', errors='ignore') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        data.append(row[0])
        label.append(row[3])

count = 0
tp = 0
fp = 0
fn = 0
tn = 0
for i in range(len(data)):
    results = analyzer.analyze(text=data[i], language="ko")
    print(data[i])
    for result in results:
        print(result)
        if result.entity_type == "주소":
            count = count + 1
            if label[i] == "true": #진짜 데이터를 TRUE라고 한것.
                tp = tp + 1
            else:                   #가짜 데이터를 TRUE라고 한것.
                tn = tn + 1
    if len(results) < 1:
        if label[i] == "true": #진짜 데이터를 FALSE라고 한것.
            fp = fp + 1
        else:                   #가짜 데이터를 FALSE라고 한것.
            fn = fn + 1

print(tp/(tp+fp+0.01))
print(tp/(tp+fn+0.01))


#Precision = 0.6335667833916958
#Recall = 0.9052086125256857
#Accuracy = 0.93079