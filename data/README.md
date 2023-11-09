# data info       
- source: 해당 데이터의 출처, {data_number}-{essay source}       
data_number는 각 데이터에 할당된 고유 번호, 01.blabla 에서 01이 고유 번호 임       
essay source는 어떤 llm에서 얻은 에세이인지, 사람이 작성한 에세이 인지 구분함.
          
- prompt: text를 작성할 때 사용한 지문, prompt가 없는 녀석은 NaN으로 표기됨       
- generated: 사람, llm이 작성했는지 여부 (label임)       
- text: 에세이       
- id: text에 할당된 고유 값       

# datasets list
## 01.[LLM Generated Essays for the Detect AI Comp!](https://www.kaggle.com/datasets/radek1/llm-generated-essays)
gpt3.5와 gpt4를 이용해 작성된 데이터, gpt3.5: 500, gpt4: 200개 합쳐서 전체 700개 데이터임
       
## 02.[LLM 7 prompt training dataset](https://www.kaggle.com/datasets/carlmcbrideellis/llm-7-prompt-training-dataset)
1. "Car-free Cities"
2. "Does the electoral college work?"
3. "Exploring Venus"
4. "The Face on Mars"
5. "Facial action coding system"
6. "Seeking multiple opinions"
7. "Phones and driving"       
프롬프트를 추가하니 리더보드 성능이 올라서 train_essay + 01, 07, 09, 03번의 데이터를 추가한 데이터,   
만약 데이터 이외의 데이터들을 합칠 때는 이 데이터가 어떤 데이터로 구성되었는지 잘 고려한뒤 하는 것을 추천함.       
그리고 합쳐진 데이터를 만들 떄  이 데이터를 만들 때 사용한 [전처리 코드](https://www.kaggle.com/code/carlmcbrideellis/llm-make-7-prompt-train-dataset-v2)를 이용해 데이터를 만드는 것을 추천.         

## 03.[Hello, Claude! 1000 essays from Anthropic...](https://www.kaggle.com/datasets/darraghdog/hello-claude-1000-essays-from-anthropic)
## 04.[DAIGT Proper Train Dataset](https://www.kaggle.com/datasets/thedrcat/daigt-proper-train-dataset)
'LLM Generated Essays for the Detect AI Comp!'
'persuade corpus 2.0'
'daigt data - llama 70b and falcon180b'
'Hello, Claude! 1000 essays from Anthropic...'
'대회 기본 train_essay'
데이터가 포함되어 있어서 필터링 하고 순수 데이터만 남김

## 05.[DAIGT External Train Dataset](https://www.kaggle.com/datasets/thedrcat/daigt-external-train-dataset)
 'llm이 작성한 에세이', '사람 혹은 llm이 작성 했을 수도 있는 에세이'로 분류 되어 있습니다.
이 중'사람 혹은 llm이 작성 했을 수도 있는 에세이'는 필터링함

## 06.[ArguGPT](https://www.kaggle.com/datasets/alejopaullier/argugpt)
## 07.[persuade corpus 2.0](https://www.kaggle.com/datasets/nbroad/persaude-corpus-2)
이번 대회 train, test 데이터를 만들 때 사용되었을 거라 추측되는 데이터,
## 08.[DeepFake_Text_Datasets](https://www.kaggle.com/datasets/myncoder0908/deepfake-text-datasets)
이 데이터는 일단 보류     
데이터의 양은 많으나 정작 어떤 프롬프트와 어떤 과정을 통해서 에세이를 작성한 건지 명확하지가 않아서 EDA를 통해 사용할지 말지를 결정하는게 좋을 듯     

## 09.[daigt data - llama 70b and falcon180b](https://www.kaggle.com/datasets/nbroad/daigt-data-llama-70b-and-falcon180b)
## 10.[LLMs-data](https://www.kaggle.com/datasets/steubk/llm-by-sakibsh)
이 데이터애는 BARD, gpt3.5, 사람이 작성한 에세이 + 시 + 스토리 + 코드 등이 포함되어 있음.       

에세이 이외의 다른 데이터는 전부 삭제함.      
## 11.[IELTS Writing Scored Essays Dataset](https://www.kaggle.com/datasets/mazlumi/ielts-writing-scored-essays-dataset)
[LLM - Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/453410)
내용에 test셋에 포함된 프롬프트로 학습시키면 리더보드 점수가 올랐지만 그렇지 않으면 내려갔다는 말이 있어서      
이 데이터는 추가적인 실험 후에 사용할 지 말지를 결정해야 할 듯      
