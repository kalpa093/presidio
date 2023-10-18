from google.cloud import vision
from google.oauth2 import service_account
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider


#Presidio 사용을 위한 환경설정, nlp engine 불러오기
configuration = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "ko", "model_name": "ko_core_news_lg"},],
}

provider = NlpEngineProvider(nlp_configuration=configuration)
nlp_engine = provider.create_engine()

analyzer = AnalyzerEngine(
    nlp_engine=nlp_engine,
    supported_languages=["ko"]
)


# 서비스 계정 키 파일의 경로를 설정
credentials = service_account.Credentials.from_service_account_file('visionapi.json')

# 클라이언트를 생성
client = vision.ImageAnnotatorClient(credentials=credentials)

# 이미지 파일의 경로를 설정
image_path = '옥션_1.png'

# 이미지를 열고 바이너리 데이터로 읽어
with open(image_path, 'rb') as image_file:
    content = image_file.read()

# Vision API를 사용하여 텍스트를 읽기
image = vision.Image(content=content)
response = client.text_detection(image=image)

# 결과에서 텍스트 추출 및 Presidio 분석
texts = response.text_annotations
#f = open("옥션_1.txt", 'w', encoding='utf-8')
for text in texts:
    print(text.description)
    #f.write(text.description)
    results = analyzer.analyze(text=text.description, language="ko")
    for result in results:
        print(result)
#f.close()
