# Discussion Content Review

### 고정된 discussion
대회의 목적: LLM이 학생들로 하여금 cheating, 표절, 그리고 글쓰기 능력이 저하되는 것을 우려한다. 또한, 학생들이 쓰는 경우에는 그들 스스로의 아이디어가 아니라 LLM의 아이디어들을 차용하여 충분한 노력없이 리서치 에세이들을 작성할 수 있는 부분을 우려하고 있다. 이렇게 된다면, 학생들은 제공되는 다른 리소스들을 읽어보지 않고도, LLM을 이용하여 비교적 그럴 듯한 에세이를 짧은 시간 안에 만들어 낼 수 있기 때문에, 학습 효과에 심각한 영향을 준다. 그래서, 이 대회는 LLM이 생성한 건지, 실제 사람이 작성한 문서인지 확인할 수 있는 모델을 개발하는 것이다. 

대회는 총 2가지 track으로 진행된다: 전통적인 정확도 베이스와, 효율성 측면에서 검증하는 것. 효율성 측면은, GPU 가 없는 환경에서 CPU 만으로 돌릴 수 있는 환경에서 정확도를 얼마나 뽑을 수 있는가가 관건


출처: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/452097
(번외)
대회의 공식적인 디스코드 채널:  discord.gg/kaggle 
저 채널 주소는 캐글 공식 디스코드 채널인데, 해당 대회는  public 안에서 찾아서 따로 그 안에서 관련 대회들 얘기를 할 수 있는 형태로 구성되어 있다고 하니 참고하면 될 거 같다. 

LLM-generated-text-detection github link: https://github.com/NLP2CT/LLM-generated-Text-Detection
출처: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/453481


Discussion에서 나온 질문 중에, 70B 써본 사람 없냐고 물어보는 게시글에 다른 LLM 관련 캐글에서 상위 점수를 받은 팀이 70B를 써서 점수 올린 걸 보면, 확실히 parameter 큰 모델을 쓰면 Leaderboard 상위권 먹을 확률이 높을 듯 싶다. 

https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/453418 -> LLM Science exam 대회에서 platypus2-70B + Wikipedia RAG 써서 private score 0.91, public score 0.909 나온 케이스 

출처: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/453418

LLM-generated text detection 관련 논문: https://arxiv.org/abs/2303.07205

OpenAI 가 이전에 AI generated text classifier 를 출시한 적이 있었는데, 그게 2023-07-20 을 기점으로 낮은 예측률로 인해서 서비스를 종료하게 됨. 

그러면서, 본인들의 한계점에 대해서 정리한 것이 있어서 그것도 추가로 정리해보면: 
 1. classifier 가 짧은 텍스트에 대해서 정확도가 매우 낮음(1000 character 미만), 그리고 가끔 긴 텍스트에 대해서도 제대로 분류하지 못하는 케이스들이 있었음.
 2. 종종 사람이 적은 글들을 굉장히 높은 확률로 ai가 생성한 글이라는 식으로 분류하는 경우가 있음.
 3. 영어 이외 다른 언어들의 데이터 같은 경우는 굉장히 정확도가 낮음
 4. 확률적으로 다음 단어를 유추하기 쉬운 경우는, ai가 쓴 건지 사람이 쓴 건지 구분을 하더라도, 그 결과를 믿기 힘듬. 예를 들어, 1000개의 소수를 가장 작은 소수부터 뽑는다고 했을 때, 그 결과값은 사람이 생성하든 인공지능이 생성하든 동일하기 때문에 이런 경우에는 인공지능이 생성한 것인지 사람이 생성한 것인지 구분이 힘듬
 5. classifier가 있다고 하더라도, 그 classifier를 우회할 수 있는 툴들이 나올 것이기 때문에 classifier가 long-term으로 봤을 때의 효용성이 떨어짐
 6. neural network으로 훈련된 classifier 같은 경우, training data 밖의 데이터를 구분할 때 굉장히 낮은 정확도가 나옴. 트레이닝 데이터에 없는 input을 넣었을 때 굉장히 자신감 있게 예측하는데 틀리는 경우가 종종 있었음.

 OpenAI가 classifier를 만들게 된 과정: 하나의 주제를 가지고 human-written data와 AI-generated data를 훈련시킴. 데이터 수집은 굉장히 다양한 분야에서 뽑았고, 사람이 썼을 거 같은 데이터들을 뽑았다. 그리고 나서, 데이터를 prompt와 response로 나누고 하나의 prompt를 fine-tuning된 다양한 LM에 각각 넣어서 뽑고 그 데이터들을 통해서 text classifier를 운영했다. Text Classifier 웹 앱에서는, confidence threshold를 조정해서 false positive 지수를 낮게 잡음. 다른 말로 말하면, ai-written으로 분류할 때는 모델 정확도가 굉장히 높다고 뜰 때만 ai-written text로 분류했음.

https://www.kaggle.com/code/defdet/llama-2-13b-on-tpu-training/notebook -> 라마 13B quantization 없이 TPU로 돌릴 수 있는 노트북
https://www.kaggle.com/code/sandiago21/llm-science-exam-llms-training-tpu ->  quantization 하긴 해야 하지만, 라마 30B/70B TPU로 돌릴 수 있는 노트북(상세한 내용은 노트북을 자세히 들여다보진 않아서 들어가서 봐야 할 듯)



대회에서, 7개의 주제로 이루어진 prompt에 대해서 관련 에세이가 사람이 쓴 건지 LLM이 쓴 건지 구분해야 하는 대회인데, 여러 가지 테스트를 해보면서 대회에서 쓴 7가지 prompt들을 추측하기엔: 
1. "Car-free Cities"
2. "Does the electoral college work?"
3. "Exploring Venus"
4. "The Face on Mars"
5. "Facial action coding system"
6. "Seeking multiple opinions"
7. "Phones and driving"

계속 시도해보니, test data 에 있지 않는 다른 prompt들로 에세이를 만들어서 모델을 훈련시키면, leaderboard 점수가 떨어진다고 하고, test data에 있는 prompt들로 에세이를 만들어서 훈련시킨 모델은 leaderboard 점수가 올라가는 현상이 있었다. 

그래서 데이터를 스스로 만들어서 공유하니, 참고하면 될 거 같다. 
데이터셋 설명: 13,712개의 human dataset(PERSUADE 2.0 corpus에서 가져옴) 와 1165 AI-LLM generated text 는 다른 두 명의 사람에게서 가져옴. AI-LLM generated 데이터 같은 경우는, 7 prompts를 이용하여 만들어진 데이터라고 함. 

추가적으로, persuade corpus 2.0에서 뽑은 13 개의 prompt들도 정리합니다(prompt 옆 숫자들은 그 prompt로 만들어진 에세이 개수들입니다):
1. Facial action coding system	2167
2. Distance learning	2157
3. Does the electoral college work?	2046
4. Car-free cities	1959
5. Driverless cars	1886
6. Exploring Venus	1862
7. Summer projects	1750
8. Mandatory extracurricular activities	1670
9. Cell phones at school	1656
10. Grades for extracurricular activities	1626
11. The Face on Mars	1583
12. Seeking multiple opinions	1552
13. Community service	1542
14. "A Cowboy Who Rode the Waves"	1372
15. Phones and driving	1168

"7 Prompts" training dataset discussion 적은 사람이 조사해봤을 때는 위 15개의 prompt들을 테스트해보니, 결국 7개의 prompt로 만들어진 에세이로 모델 평가를 했을 때 Leaderboard 점수가 올랐고 나머지 Prompt들은 점수가 떨어졌다고 하니, 7개의 Prompt 위주로 essay를 생성해야 할 거 같습니다. 



출처: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/453410

다른 글들을 봐도, 확실히 AI-generated Data에 노이즈(철자 틀림, 문법적 오류)를 주최 측에서 고의로 넣은 듯한 느낌이 있다고 하니 노이즈 제거도 생각해야 할 듯 싶습니다. 

출처: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/453283


Deepfake Text Datasets 관련 논문 주소: https://arxiv.org/pdf/2305.13242.pdf
Deepfake Text Datasets github: https://github.com/yafuly/DeepfakeTextDetect

데이터 노이즈 중 문법이 틀린 경우 detect 해주는 코드: https://www.kaggle.com/code/defdet/grammar-correction-detect-ai





