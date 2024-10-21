import openai
import json
import os

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "api_key"

# gpt-4에 요청하여 PII 데이터가 포함된 문장과 BIO 태깅을 생성하는 함수
def generate_tagged_sentences(n):
    results = []
    for _ in range(n):
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "데이터셋의 생성의 전문가가 되어줘"},
                {"role": "user", "content": """
                여러가지 다양한 상황에서의 자연스러운 PII 데이터를 포함하고 있는 한국어 문장의 대화를 생성해줘.
                PII 데이터에는 이름, 전화번호, 주민등록번호, 이메일, 주소, 운전면허번호, 건강보험번호, 여권번호, 신용카드번호, 운송장번호가 있어.
                그리고 각 문장의 단어에 대해 PII데이터에 대한 BIO tagging을 해줘.
                출력은 JSON 형식으로 해줘.

                예시 문장 형식:
                A: 안녕하세요, 김민수 씨. 오늘 회의는 오후 3시에 시작됩니다. 혹시 급한 문의사항이 있으면 제 이메일로 보내주세요. 제 이메일은 kimminsu@example.com입니다.
                B: 네, 박수진 씨. 알려주셔서 감사합니다. 혹시 모임 장소를 잊어버리면 어디로 연락드리면 될까요?
                A: 제 전화번호로 연락하시면 됩니다. 010-1234-5678이에요.
                B: 알겠습니다. 그리고 제 여권번호가 필요한 서류 작성이 있어서 미리 알려드릴게요. 여권번호는 M12345678입니다.
                A: 감사합니다, 민수 씨. 그럼 오후 3시에 뵙겠습니다.

                예시 출력 형식:
                [
                  {
                    "sentence": "김민수의 주민등록번호는 801010-1234567이고, 사는 곳은 서울특별시 강남구 테헤란로 123입니다.",
                    "labels": [
                        {"word": "김민수", "label": "B-NAME"},
                        {"word": "의", "label": "O"},
                        {"word": "주민등록번호는", "label": "O"},
                        {"word": "801010-1234567", "label": "B-RRN"},
                        {"word": "이고", "label": "O"},
                        {"word": ",", "label": "O"},
                        {"word": "사는", "label": "O"},
                        {"word": "곳은", "label": "O"},
                        {"word": "서울특별시", "label": "B-ADDRESS"},
                        {"word": "강남구", "label": "I-ADDRESS"},
                        {"word": "테헤란로", "label": "I-ADDRESS"},
                        {"word": "123", "label": "I-ADDRESS"},
                        {"word": "입니다", "label": "O"},
                        {"word": ".", "label": "O"}
                    ]
                  },
                  {
                    "sentence": "이수진의 이메일은 sujin.lee@example.com입니다. 그리고 운전면허번호는 12-34-567890입니다.",
                    "labels": [
                        {"word": "이수진", "label": "B-NAME"},
                        {"word": "의", "label": "O"},
                        {"word": "이메일은", "label": "O"},
                        {"word": "sujin.lee@example.com", "label": "B-EMAIL"},
                        {"word": "입니다", "label": "O"},
                        {"word": ".", "label": "O"},
                        {"word": "그리고", "label": "O"},
                        {"word": "운전면허번호는", "label": "O"},
                        {"word": "12-34-567890", "label": "B-DRIVERS_LICENSE"},
                        {"word": "입니다", "label": "O"},
                        {"word": ".", "label": "O"}
                    ]
                  }
                ]
                """
                }
            ]
        )

        response_data = response.choices[0].message.content
        print(response_data)
        try:
            sentence_data = json.loads(response_data)
            results.append(sentence_data)
        except json.JSONDecodeError:
            print(f"Failed to decode JSON: {response_data}")
            continue
    return results

N = 1000
# N개의 문장을 생성하고 태깅
tagged_sentences = generate_tagged_sentences(N)

# JSON 파일로 저장
with open('tagged_sentences.json', 'w', encoding='utf-8') as f:
    json.dump(tagged_sentences, f, ensure_ascii=False, indent=4)

print("JSON 파일이 성공적으로 생성되었습니다.")