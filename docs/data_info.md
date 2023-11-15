# data format description
|             |           format          |              description         |
|:-----------:|:-------------------------:|:--------------------------------:|
|      id     |        unique_value       |      각 데이터에 할당된 고유 값    |
|     text    |            str            |       학생, llm이 작성한 에세이    |
|  generateed |           0 ~ 1           |         llm이 작성했는지 여부      |
| instruction |            str            |        text를 작성할 떄의 주제     |
| source_text |            str            |     text를 작성할 때 참고한 문장   |
| prompt_name |            str            |          instruction의 제목        |
|    source   | {data_idx}-{llm or human} |병합 되었을 때 각 데이터의 출처를 명시|

## 전처리 특징
기존 raw_data의 format을 맞추는 과정에서 기존의 컬럼에 generated, instruction과 같은 필수 컬럼만 추가함.     
이후 필수 컬럼 이외의 다른 컬럼들은 huggingface datasets로 변환하는 과정에서 삭제했음.       
그리고 generated의 카테고리는 [0, 1]로 고정임. 만약 이외의 카테고리를 가진 열은 필터링 함.     


# datasets list
## 01.[LLM Generated Essays for the Detect AI Comp!](https://www.kaggle.com/datasets/radek1/llm-generated-essays)
대회에서 제공하는 `Car-free Cities`, `Does the electoral college work?` instruction을 gpt3.5, gpt4를 이용해 생성한 데이터

- column      
    `id`, `prompt_id`, `text`, `generated`

<details>
<summary>raw_data 전처리 과정</summary>

1. gpt3.5, gpt4를 각각 파일이 있기 때문에 이를 병합시킴. 
2. 대회에서 기본 제공하는 prompt_table과 병합한 테이블 끼리 prompt_id 기준으로 병합함.

</details>

       
## 02.[LLM 7 prompt training dataset](https://www.kaggle.com/datasets/carlmcbrideellis/llm-7-prompt-training-dataset)
1. "Car-free Cities"
2. "Does the electoral college work?"
3. "Exploring Venus"
4. "The Face on Mars"
5. "Facial action coding system"
6. "Seeking multiple opinions"
7. "Phones and driving"       

01 + 07 + 09 + 03 데이터 중 persuade2.0의 15개 instrunction에서      
source_text가 할당된 7개 instruction 기반으로 생성된 데이터만 필터링 해서 합친 데이터     
이 데이터를 만들 때 사용한 [전처리 코드](https://www.kaggle.com/code/carlmcbrideellis/llm-make-7-prompt-train-dataset-v2)를 이용해 데이터를 만드는 것을 추천.            

- column      
    `text`, `label`

<details>
<summary>raw_data 전처리 과정</summary>

1. `label`:`generated`로 컬럼명을 변경함.
</details>

## 03.[Hello, Claude! 1000 essays from Anthropic...](https://www.kaggle.com/datasets/darraghdog/hello-claude-1000-essays-from-anthropic)
Claude에 persuade2.0의 15개 instrunction를 넣어서 1000개의 에세이로 구성되어 있는 데이터      

- column    
    `prompt_id`, `essay_title`, `essay_text`
<details>
<summary>raw_data 전처리 과정</summary>

1. persuade2.0 prompt_table를 이용해 사전 제작된 persuade_prompt_table을 이용해 promtp_name, instruction, source_text를 prompt_id를 이용해 병합함.      
사전 제작된 persuade_prompt_table의 prompt_id는 Claude에 있는 prompt_title과 prompt_id를 서로 매칭시켜서 맞췄음.
2. `promtp_name`:`essay_title`,`essay_text`:`text`로 컬럼명을 변경함.
3. generated 컬럼이 앖어서 추가. 값은 1로 채움.
</details>

## 04.[DAIGT Proper Train Dataset](https://www.kaggle.com/datasets/thedrcat/daigt-proper-train-dataset)
00 + 01 + 03 + 07 + 09에서 작성된 에세이가 포함되어 있는 데이터

- column    
    `essay_id`, `text`, `label`, `source`, `prompt`, `fold`
<details>
<summary>raw_data 전처리 과정</summary>

1. `label`:`generated`,`prompt`:`instruction`로 컬럼명을 변경함.
</details>

## 05.[DAIGT External Train Dataset](https://www.kaggle.com/datasets/thedrcat/daigt-external-train-dataset)

10 + 06 + `ChristophSchuhmann/essays-with-instructions` + `qwedsacf/ivypanda-essays` + `dim/essayforum_raw_writing_10k` + `whateverweird17/essay_grade_v1` + `whateverweird17/essay_grade_v2` + `nid989/EssayFroum-Dataset`의 에세이가 포함되어 있는 데이터     

generated기 `llm, 사람이 작성한 에세이`, `사람, llm이 작성 했을 수도 있는 에세이`로 되어 있기 때문에 llm, 사람이 작성한 것이 확실치 않은 에세이는 필터링 시킴.     
이 과정은 huggingface datasets로 변환하는 과정에서 이루어짐     

- column    
    `essay_id`, `text`, `label`, `source`, `prompt`
<details>
<summary>raw_data 전처리 과정</summary>

1. `label`:`generated`,`prompt`:`instruction`로 컬럼명을 변경함.
</details>


## 06.[ArguGPT](https://www.kaggle.com/datasets/alejopaullier/argugpt)
7개의 gpt모델에 `in-class or homework exercises`, `TOEFL`, `GRE`에서 참고한 프롬프트를 넣어서 생성된 4,038개의 논쟁으로 구성된 데이터

- column    
    `id`, `prompt_id`, `prompt`, `text`, `model`, `temperature`, `exam_type`, `score`, `score_level`
<details>
<summary>raw_data 전처리 과정</summary>

1. `machine-dev.csv`, `machine-text.csv`, `machine-train.csv`파일 삭제
2. `prompt`:`instruction`,`model`:`source`로 컬럼명을 변경함.
3. generated 컬럼이 없어서 컬럼을 추가. 값은 1로 채움.
</details>

## 07.[persuade corpus 2.0](https://www.kaggle.com/datasets/nbroad/persaude-corpus-2)
대회 train 데이터를 만들 때 사용한 데이터

폴더 안에 1.0과 2.0가 있지만 이중에서 2.0만 사용함.


- column    
    `essay_id_comp`, `full_text`, `holistic_essay_score`, `word_count`, `prompt_name`, `task`, `assignment`, `source_text`, `gender`, `grade_level`, `ell_status`, `race_ethnicity`, `economically_disadvantaged`,  `student_disability_status`


<details>
<summary>raw_data 전처리 과정</summary>

1. `full_text`:`text`,`assignment`:`instruction`로 컬럼명을 변경함
2. generated 컬럼이 없어서 컬럼을 추가. 값은 0로 채움.
</details>

## 08.[DeepFake_Text_Datasets](https://www.kaggle.com/datasets/myncoder0908/deepfake-text-datasets)
이 데이터는 일단 보류     
데이터의 양은 많으나 정작 어떤 프롬프트와 어떤 과정을 통해서 에세이를 작성한 건지 명확하지가 않아서 EDA를 통해 사용할지 말지를 결정하는게 좋을 듯     

## 09.[daigt data - llama 70b and falcon180b](https://www.kaggle.com/datasets/nbroad/daigt-data-llama-70b-and-falcon180b)
llama, falcon에서 생성한 에세이로 구성된 데이터셋

- column    
    `generated_text`, `writing_prompt`
<details>
<summary>raw_data 전처리 과정</summary>

1. `falcon_180b_v1.csv`, `llama_70b_v1.csv`를 `llama70b_and_falcon180b.csv`로 병합함.
2. `writing_prompt`:`instruction`,`generated_text`:`text`로 컬럼명 변경
3. generated 컬럼이 없어서 컬럼을 추가. 값은 0로 채움.
</details>
   

## 10.[LLMs-data](https://www.kaggle.com/datasets/steubk/llm-by-sakibsh)
이 데이터애는 BARD, gpt3.5, 사람이 작성한 에세이 + 시 + 스토리 + 코드 등이 포함되어 있음.       

- BARD-column    
    `prompts`, `BARD`
- GPT-column    
    `prompts`, `responses`
- human1-column    
    `ID`, `essays`
- human2-column    
    `question`, `sample`, `text`

<details>
<summary>raw_data 전처리 과정</summary>

1. 에세이 파일 이외의 다른 파일은 전부 제거, 예: poetry, stories, code
2. BARD, GPT, human이 생성한 에세이에 generated 컬럼 추가 및 컬럼명 변경
    BARD: `prompts`:`instruction`,`BARD`:`text`      
    GPT: `prompts`:`instruction`,`responses`:`text`      
    human1: `essays`:`text`      
    human2: `question`:`instruction`      
3. BARD에는 unnamed:0 과 같은 이상한 컬럼도 있어서 일단 전처리 단계에서 제거함.
2. BARD, GPT, human를 `sakibsh_dataset.csv`로 병합함.
3. generated 컬럼이 없어서 컬럼을 추가. human은 0로, BARD & GPT는 1로
</details>
   
## 11.[IELTS Writing Scored Essays Dataset](https://www.kaggle.com/datasets/mazlumi/ielts-writing-scored-essays-dataset)
IELTS 시험에서 응시자들이 작성한 에세이로 구성된 데이터

- column    
    `Task_Type`, `Question`, `Essay`, `Examiner_Commen`, `Task_Response`, `Coherence_Cohesion`, `Lexical_Resource`, `Range_Accuracy`, `Overall`


<details>
<summary>raw_data 전처리 과정</summary>

1. `Question`:`instruction`,`Essay`:`text로` 변경함.
2. generated 컬럼이 없어서 컬럼을 추가. 값은 1로 채움.
</details>

