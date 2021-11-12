# 📋 KorSci-Electra 기반 개체명 인식 모델 개발 연구  (21. 11. 1. ~)
- 소개 : 과학기술분야 데이터를 활용한 개채명 인식 KorSci-ELECTRA 모델 개발
- 맡은 역할 : 학습 데이터(NTIS) 전처리, 모델 개발

# NER(NTIS)

NTIS 데이터
- 연구보고서 2014~2017년
- 사용한 필드 영역
  - 연구내용요약, 키워드
  - 연구내용요약 : 총 188,214건
  - 키워드 : 총 832,141개
- 분류 수 : 323
  - ex) EA01, HB01, NC01 ...

데이터 전처리
- 연구내용요약
  - 길이가 20 초과인 요약만 사용
  - ‘보안상 생략’, ‘세부내용 생략’, ‘세부사항 생략’, ‘보안과제임으로 비공개’, ‘보안과제로 생략’이라는 문장이 포함된 요약은 연구내용에 대한 내용이 존재하지 않아서 제외
  - 요약 내용에 한글이나 영어가 포함되어 있지 않다면 제외
  - 123,646건

- 키워드
  - 길이가 20 미만인 키워드만 사용
  - 중복 제거 (232,410개)
  - 일반적인 단어 정제
    - 키워드에 매칭되는 분류 코드의 수가 5개 초과라면 제외
    - 키워드에 매칭되는 분류 코드의 1순위 빈도 수가 
      - 100개 이상이라면 1순위와 2순위의 빈도 수 차이가 1순위 빈도 수의 10% 이상 차이가 난다면 제외
      - 100개 미만이라면 1순위와 2순위의 빈도 수 차이가 3개 이상 차이가 난다면 제외
  - TFIDF를 활용하여 정제 중

사전학습 모델
- Electra 
  - Base 모델
- 데이터
  - 학술 논문, NTIS 연구 보고서, 특허, 한국어 위키, 네이버 IT 뉴스
  - 141.58GB
  - Mecab 적용
- 사전 크기
  - 16200
<!--   
실험 결과
|정제작업|토크나이저|필드 영역|Macro F-1|Micro F-1|
|---|---|---|---|---|
|Case 1|WordPiece|발명의명칭, 청구항, 요약서, 배경기술, 기술분야, 과제의해결수단, 발명의상세한설명|0.62|0.72|
|Case 2|WordPiece|발명의명칭, 청구항, 요약서, 배경기술, 기술분야, 과제의해결수단, 발명의상세한설명|0.63|0.72|
|Case 3|WordPiece|발명의명칭, 청구항, 요약서, 배경기술, 기술분야, 과제의해결수단, 발명의상세한설명|0.63|0.73|
|Case 2|WordPiece|발명의명칭, 청구항, 요약서|0.63|0.73|
|Case 3|WordPiece|발명의명칭, 청구항, 요약서|0.62|0.71|

대부분의 정제 작업은 유사한 학습결과를 보여주었으며 Case 3가 가장 높은 성능을 보여주었다. 필드 영역을 [발명의명칭, 청구항, 요약서]로 변경하여 실험한 결과 모든 필드 영역을 사용했을 때보다 Case 2는 약간의 성능 향상이 있었으나 Case 3는 오히려 성능이 하락하는 결과가 나타났다. Case 2는 제외된 [배경기술, 기술분야, 과제의해결수단, 발명의상세한설명]의 정제작업이 적절하지 않았기 때문에 성능이 향상된 것으로 보이고 Case 2보다 Case 3에서 적절한 정제작업이 이루어진 필드 영역이 제외되었기 때문에 성능이 하락한 것으로 보인다.

오류 분석  
본 연구에서는 특허 코드를 자동으로 분류하기 위한 모델을 언어 모델을 학습하였다. 학습 데이터로 쓰인 특허 코드는 세분화된 분류가 포함된 4자리 코드만을 사용하여 종류가 많아 포함되는 내용이 포괄적이기 때문에 분류하는 데에 있어서 오류를 일으켰다.  
(B29C 31/00, B29C 31/02, B29C 31/04 ... 같은 코드들이 B29C에 함께 포함됨)  
분류 코드를 소분류까지 분류한다면 Target label이 너무 많아지고 이번 연구처럼 코드를 4자리까지만 포함시키면 다양한 내용과 단어를 포함하여 해당 분류와 관련이 적은 텍스트를 모두 학습 데이터로 사용하게 되는 한계가 있었다. 이를 극복하기 위해 텍스트 내에서 분류 코드에 해당되는 키워드를 파악할 수 있는 개체명 인식기를 개발한다면 분류에 필요한 키워드를 추출하여 적절한 텍스트 데이터를 구축할 수 있을 것이다. 결과적으로는 해당 도메인의 분류 성능에 도움을 줄 수 있을 것이다. -->
<!-- 
