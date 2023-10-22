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
with open('address.csv', 'r', encoding='cp949', errors='ignore') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        data.append(row[0])
        label.append(row[1])

count = 0
tp = 0
fp = 0
fn = 0
tn = 0
true_data = []
true_label = []
false_data = []
false_label = []

for i in range(len(data)):
    if label[i] == "TRUE":
        true_data.append(data[i])
        true_label.append("TRUE")
    elif label[i] == "FALSE":
        false_data.append(data[i])
        false_label.append("FALSE")

for i in range(len(true_data)):
    results = analyzer.analyze(text=true_data[i], language="ko")
    print(true_data[i])
    print(results)
    for result in results:
        if result.entity_type == "주소":
            count = count + 1
            if true_label[i] == "TRUE":
                tp = tp + 1
            else:
                fp = fp + 1
    if len(results) < 1:
        if true_label[i] == "TRUE":
            fn = fn + 1
        else:
            tn = tn + 1


for i in range(len(false_data)):
    results = analyzer.analyze(text=false_data[i], language="ko")
    print(false_data[i])
    print(results)
    for result in results:
        if result.entity_type == "주소":
            count = count + 1
            if false_label[i] == "TRUE":
                tp = tp + 1
            else:
                fp = fp + 1
    if len(results) < 1:
        if false_label[i] == "TRUE":
            fn = fn + 1
        else:
            tn = tn + 1

print(tp)
print(tn)