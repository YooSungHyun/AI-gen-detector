# Discussion Content Review

시간순 정렬 후 여기 이전 까지를 정리함: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/453481

### 고정된 discussion

대회의 목적: LLM이 학생들로 하여금 cheating, 표절, 그리고 글쓰기 능력이 저하되는 것을 우려한다. 또한, 학생들이 쓰는 경우에는 그들 스스로의 아이디어가 아니라 LLM의 아이디어들을 차용하여 충분한 노력없이 리서치 에세이들을 작성할 수 있는 부분을 우려하고 있다. 이렇게 된다면, 학생들은 제공되는 다른 리소스들을 읽어보지 않고도, LLM을 이용하여 비교적 그럴 듯한 에세이를 짧은 시간 안에 만들어 낼 수 있기 때문에, 학습 효과에 심각한 영향을 준다. 그래서, 이 대회는 LLM이 생성한 건지, 실제 사람이 작성한 문서인지 확인할 수 있는 모델을 개발하는 것이다. 

대회는 총 2가지 track으로 진행된다: 전통적인 정확도 베이스와, 효율성 측면에서 검증하는 것. 효율성 측면은, GPU 가 없는 환경에서 CPU 만으로 돌릴 수 있는 환경에서 정확도를 얼마나 뽑을 수 있는가가 관건


(출처: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/452097)
(번외)
대회의 공식적인 디스코드 채널:  discord.gg/kaggle 
저 채널 주소는 캐글 공식 디스코드 채널인데, 해당 대회는  public 안에서 찾아서 따로 그 안에서 관련 대회들 얘기를 할 수 있는 형태로 구성되어 있다고 하니 참고하면 될 거 같다. 


-------------------------------------------------------------------------------------------------------------------------------------------

### Open Discussion

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

1. Facial action coding system  2167
2. Distance learning    2157
3. Does the electoral college work? 2046
4. Car-free cities  1959
5. Driverless cars  1886
6. Exploring Venus  1862
7. Summer projects  1750
8. Mandatory extracurricular activities 1670
9. Cell phones at school    1656
10. Grades for extracurricular activities   1626
11. The Face on Mars    1583
12. Seeking multiple opinions   1552
13. Community service   1542
14. "A Cowboy Who Rode the Waves"   1372
15. Phones and driving  1168

"7 Prompts" training dataset discussion 적은 사람이 조사해봤을 때는 위 15개의 prompt들을 테스트해보니, 결국 7개의 prompt로 만들어진 에세이로 모델 평가를 했을 때 Leaderboard 점수가 올랐고 나머지 Prompt들은 점수가 떨어졌다고 하니, 7개의 Prompt 위주로 essay를 생성해야 할 거 같습니다. 



출처: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/453410

다른 글들을 봐도, 확실히 AI-generated Data에 노이즈(철자 틀림, 문법적 오류)를 주최 측에서 고의로 넣은 듯한 느낌이 있다고 하니 노이즈 제거도 생각해야 할 듯 싶습니다. 

출처: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/453283


Deepfake Text Datasets 관련 논문 주소: https://arxiv.org/pdf/2305.13242.pdf
Deepfake Text Datasets github: https://github.com/yafuly/DeepfakeTextDetect

데이터 노이즈 중 문법이 틀린 경우 detect 해주는 코드: https://www.kaggle.com/code/defdet/grammar-correction-detect-ai

<2페이지>

LLMDet: Third Party LLM generated text detection Tool 
    gitub: https://github.com/TrustedLLM/LLMDet#-llmdeta-third-party-large-language-models-generated-text-detection-tool--paper
    paper: https://arxiv.org/abs/2305.15004

DetectGPT medium post: https://medium.com/@TheHaseebHassan/detectgpt-detecting-ai-generated-text-a0284f1d05de

Defending Against Neural Fake News by Allen Institute(University of Washington)
    paper: https://dl.acm.org/doi/pdf/10.5555/3454287.3455099
    간단 설명: 가짜뉴스 판별기

GLTR: Statistical Detection and Visualization of Generated Text
    paper: https://aclanthology.org/P19-3019.pdf

BERTology Meets Biology: Interpreting Attention in Protein Language Models
    간단 설명: attention 원리를 이해하기 위해 language model에 단백질 구조를 훈련시켜 biological data에서 ai-generated text의 흔적들을 찾은 논문

Here are the roc_auc scores for each of the generated datasets/models that were shared:

discussion에 공개된 데이터들을 통해서 훈련한 모델들의 roc-auc 점수들
    chat_gpt_moth: 1.0
    falcon_180b_v1: 1.0
    radek_500: 1.0
    llammistral7binstruct: 0.9999401125883339
    llama2_chat: 0.9999701795192938
    llama_70b_v1: 0.9996283909327388
    학습 데이터: DAIGT Proper Train Dataset

출처: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/452536


chatgpt 에게 에세이 쓰라고 하는 프롬프트가 정리된 노트북: https://www.kaggle.com/code/mozattt/daigt-script-example-to-generate-ai-essay

출처: https://www.kaggle.com/code/mozattt/daigt-script-example-to-generate-ai-essay

SeqXGPT: Sentence-Level AI-Generated Text Detection
    document level detection. 

Can AI-Generated Text be Reliabily Detected?
    paper link:  https://arxiv.org/pdf/2303.11156.pdf
    code: https://github.com/vinusankars/Reliability-of-AI-text-detectors

The Science of Detecting LLM-Generated Texts
    paper link: https://arxiv.org/pdf/2303.07205.pdf

The Imitation Game: Detecting Human and AI-Generated Texts in the Era of Large Language Models
    paper: https://arxiv.org/pdf/2307.12166.pdf
    dataset link inside of the paper:  https://github.com/sakibsh/LLM

Detecting ChatGPT: A Survey of the State of Detecting ChatGPT-Generated Text
    paper: https://arxiv.org/pdf/2309.07689.pdf

## 어느 수준의 학생이 쓴 에세이를 대상으로 하였는가?

https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/452372

개요에 중학생, 고등학생 수준으로 나와있다.

## 미스트랄 7B를 이용한 0.854 베이스라인

https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/452362

-   train
    -   https://www.kaggle.com/code/hotchpotch/train-llm-detect-ai-comp-mistral-7b
-   infer
    -   https://www.kaggle.com/code/hotchpotch/infer-llm-detect-ai-comp-mistral-7b

캐글 커널 기준 P100 4시간 소요. 예측 3시간 소요

DAIGT 데이터셋 사용, 코드는 아래를 참고해서 만들었음.

-   https://www.kaggle.com/datasets/alejopaullier/daigt-external-dataset/data
-   https://www.kaggle.com/code/alejopaullier/daigt-deberta-text-classification-train

## LLM으로 에세이를 만들때 유의할 점

https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/452262

1.   평가된 녀석이 어떤 LLM으로 생성됐을지는 알 수 없음
2.   gpt-3.5-turbo만 사용하면 편향이 발생할 수 있음
3.   LLM마다 글쓰기 스타일이 다름. 여러가지 범용모델을 사용하는것이 좋을듯
4.   템퍼러쳐등 옵션을 다양하게 조절해볼 것

## AUC 쓸때 조심할 점 (양성 비율과 객체 순서에 따른 미묘한 점수 차이 발생)

https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/452259

AUC의 계산 공식 상 나타날 수 있는 특징을 설명해놓은 글로, 기본적으로 default가 제공하는 평균이 제일 신뢰할 만 하지만, 그렇지 않을 경우도 있음을 유의하라고 함. (개인적으로 나는 근데 default가 average인데는 다 이유가 있을거라고 생각함...)

## ROC에 대한 기본 개념

https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/452236

영어로 잘 설명되어있긴한데, 그냥 한글 블로그 찾아보는게 더 빠를지도...

## GAN을 써볼 수 있지 않을지?

https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/452217

텍스트가 사람이 썼는지보다, 어떤 사람이 쓴것인지 필체를 분간하는 것이, 오히려 사람이 썼는지를 분간하는것보다 성능이 좋을 수 있다.

다만 이마저도 힘들 수 있는게, 문체라는게 시간이 흐르면서 바뀔수도 있기 때문이다. (https://news.ycombinator.com/item?id=33755016)

## 뭐가 됐던 파라미터 큰 놈이 잘하더라.

## train_prompts.csv는 정확히 PERSUADE 코퍼스에서 비롯되었다.

PERSUADE 코퍼스에서 하이픈이나 후행공백(trailing whitespace)이 제거하고 몇가지 더 수행하면 거의 비슷해질 것 같다.

PERSUADE에 있었지만, train.csv에는 없었던 녀석들이 이렇게 있다.

```python
{'\x94', ']', 'á', '¹', '`', 'å', '~', '}', 'Ö', '\\', '=', '\x97', '(', '©', '²', ')', '\x91', '>', '®', ';', '<', '£', '+', '#', '¶', '\xa0', '{', '^', '\x80', '[', '|', '\x93', '-', '\x85', 'Ó', '*', '/', '$', 'é', 'ó', '\x99'}
```

프롬프트는 15개를 찾을 수 있었으며, 그중 2번과 12번이 트레이닝 세트에 들어있다.

>   해당 15개의 방법론은, Few Shot처럼, 예시를 주어주고, 해당 소스를 통해 뭔가를 작성하기 위한 의도이므로, 내용을 그대로 가져다 쓰기보단, 에세이를 잘 쓰기위한 조건절만 활용하는게 좋을듯 하다.

1.  Today the majority of humans own and operate cell phones on a daily basis. In essay form, explain if drivers should or should not be able to use cell phones in any capacity while operating a vehicle.
2.  Write an explanatory essay to inform fellow citizens about the advantages of limiting car usage. Your essay must be based on ideas and information that can be found in the passage set. Manage your time carefully so that you can read the passages; plan your response; write your response; and revise and edit your response. Be sure to use evidence from multiple sources; and avoid overly relying on one source. Your response should be in the form of a multiparagraph essay. Write your essay in the space provided.
3.  Some schools require students to complete summer projects to assure they continue learning during their break. Should these summer projects be teacher-designed or student-designed? Take a position on this question. Support your response with reasons and specific examples.
4.  You have just read the article, 'A Cowboy Who Rode the Waves.' Luke's participation in the Seagoing Cowboys program allowed him to experience adventures and visit many unique places. Using information from the article, write an argument from Luke's point of view convincing others to participate in the Seagoing Cowboys program. Be sure to include: reasons to join the program; details from the article to support Luke's claims; an introduction, a body, and a conclusion to your essay.
5.  Your principal has decided that all students must participate in at least one extracurricular activity. For example, students could participate in sports, work on the yearbook, or serve on the student council. Do you agree or disagree with this decision? Use specific details and examples to convince others to support your position.
6.  In "The Challenge of Exploring Venus," the author suggests studying Venus is a worthy pursuit despite the dangers it presents. Using details from the article, write an essay evaluating how well the author supports this idea. Be sure to include: a claim that evaluates how well the author supports the idea that studying Venus is a worthy pursuit despite the dangers; an explanation of the evidence from the article that supports your claim; an introduction, a body, and a conclusion to your essay.
7.  In the article "Making Mona Lisa Smile," the author describes how a new technology called the Facial Action Coding System enables computers to identify human emotions. Using details from the article, write an essay arguing whether the use of this technology to read the emotional expressions of students in a classroom is valuable.
8.  You have read the article 'Unmasking the Face on Mars.' Imagine you are a scientist at NASA discussing the Face with someone who thinks it was created by aliens. Using information in the article, write an argumentative essay to convince someone that the Face is just a natural landform.Be sure to include: claims to support your argument that the Face is a natural landform; evidence from the article to support your claims; an introduction, a body, and a conclusion to your argumentative essay.
9.  Some of your friends perform community service. For example, some tutor elementary school children and others clean up litter. They think helping the community is very important. But other friends of yours think community service takes too much time away from what they need or want to do.
    -   Your principal is deciding whether to require all students to perform community service.
    -   Write a letter to your principal in which you take a position on whether students should be required to perform community service. Support your position with examples.
10.  Your principal is considering changing school policy so that students may not participate in sports or other activities unless they have at least a grade B average. Many students have a grade C average.
     -   She would like to hear the students' views on this possible policy change. Write a letter to your principal arguing for or against requiring at least a grade B average to participate in sports or other activities. Be sure to support your arguments with specific reasons.
11.  In the article “Driverless Cars are Coming,” the author presents both positive and negative aspects of driverless cars. Using details from the article, create an argument for or against the development of these cars. Be sure to include: your position on driverless cars; appropriate details from the article that support your position; an introduction, a body, and a conclusion to your argumentative essay.
12.  Write a letter to your state senator in which you argue in favor of keeping the Electoral College or changing to election by popular vote for the president of the United States. Use the information from the texts in your essay. Manage your time carefully so that you can read the passages; plan your response; write your response; and revise and edit your response. Be sure to include a claim; address counterclaims; use evidence from multiple sources; and avoid overly relying on one source. Your response should be in the form of a multiparagraph essay. Write your response in the space provided.
13.  Your principal is reconsidering the school's cell phone policy. She is considering two possible policies:
     -   Policy 1: Allow students to bring phones to school and use them during lunch periods and other free times, as long as the phones are turned off during class time.
     -   Policy 2: Do not allow students to have phones at school at all.
     -   Write a letter to your principal convincing her which policy you believe is better. Support your position with specific reasons.
14.  Some schools offer distance learning as an option for students to attend classes from home by way of online or video conferencing. Do you think students would benefit from being able to attend classes from home? Take a position on this issue. Support your response with reasons and examples.
15.  When people ask for advice, they sometimes talk to more than one person. Explain why seeking multiple opinions can help someone make a better choice. Use specific details and examples in your response.

## 유용한 아티클 모음 (이걸 모아둔 아티클이 있어서, 전부 정리하긴 힘들었음)

### Kaggle Notebooks on Kaggle - LLM Science Exam (previous Competition) ; https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/452085

Getting Started With LLMs By Jeremy Howard
https://www.kaggle.com/code/jhoward/getting-started-with-llms/notebook

enwiki-cirrus-20230701-e5-large part01 - By Pascal Pfeiffer
https://www.kaggle.com/code/ilu000/enwiki-cirrus-20230701-e5-large-part01

[publish]0.91783-ddp-e150+e120+e113+e107 - By Yiemon773
https://www.kaggle.com/code/yoichi7yamakawa/publish-0-91783-ddp-e150-e120-e113-e107

LLM Sience Exam: 5th Place Solution - By Kaizaburochubachi
https://www.kaggle.com/code/zaburo/llm-sience-exam-5th-place-solution#Preparation

The Art of Prompt Engineering - By Steubk
https://www.kaggle.com/code/steubk/the-art-of-prompt-engineering

Give Llama 2 a science exam - By Paul T. Mooney
https://www.kaggle.com/code/paultimothymooney/give-llama-2-a-science-exam

1st place. Single model inference By Yauhen Babakhin, Pascal Pfeiffer and Psi.
https://www.kaggle.com/code/ybabakhin/1st-place-single-model-inference

How to train a Transformer - By Radek Osmulski
https://www.kaggle.com/code/radek1/how-to-train-a-transformer

2023KaggleLLM_DeBERTa-V3-Large_Model1 [Inference] - By HYC and Tereka
https://www.kaggle.com/code/hycloud/2023kagglellm-deberta-v3-large-model1-inference

TeamHydrogen/white-malamute-prompt-openorca-v2 By Psi
https://www.kaggle.com/code/philippsinger/teamhydrogen-white-malamute-prompt-openorca-v2

llm-021-platypus-no-lr By CPMP
https://www.kaggle.com/code/cpmpml/llm-021-platypus-no-lr

Xwin-LM-70B-V0-1 with Wikipedia RAG, By CPMP
https://www.kaggle.com/code/cpmpml/xwin-lm-70b-v0-1-with-wikipedia-rag

RAPIDS TF-IDF - [LB 0.904] - Single Model By Chris Deotte
https://www.kaggle.com/code/cdeotte/rapids-tf-idf-lb-0-904-single-model

How To Use 40k Dataset - By Chris Deotte
https://www.kaggle.com/code/cdeotte/how-to-use-40k-dataset

BAAI/bge-large-en , By Yauhen Babakhin
https://www.kaggle.com/code/ybabakhin/baai-bge-large-en

wiki31m-gte-large-title p4, By Yauhen Babakhin

thenlper/gte-large, By Yauhen Babakhin
https://www.kaggle.com/code/ybabakhin/thenlper-gte-large