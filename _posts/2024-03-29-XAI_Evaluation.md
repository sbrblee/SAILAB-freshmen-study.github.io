---
layout: post
title: "[논문리뷰] XAI Evaluation"
date: 2024-03-29 00:00:00 +0900
author: Yeonjea Kim
categories: ["XAI", "Evaluation Methods"]
use_math: true
---
**XAI Evaluation**

설명 가능 인공지능(Explainable AI, XAI) 에 대한 포괄적인 총정리 : XAI 평가는 어떻게 해야할까? 활용할 수 있는 개념과 방법은 어떤 것이 있을까?

이 포스트에서는 아래 논문을 다룹니다.

Meike Nauta, et al., (2023) XAI Evaluation - From Anecdotal Evidence to Quantitative Evaluation Methods: A Systematic Review on Evaluating Explainable AI

웹 뷰 - [https://dl.acm.org/doi/10.1145/3583558](https://dl.acm.org/doi/10.1145/3583558)

arxiv - [https://arxiv.org/abs/2201.08164](https://arxiv.org/abs/2201.08164)

제공중인 사이트를 둘러보는 것을 추천합니다 - [https://utwente-dmb.github.io/xai-papers/](https://utwente-dmb.github.io/xai-papers/)

### Summary

머신러닝 모델의 복잡성이 증가함에 따라 XAI에 대한 필요성이 커지고 있습니다. 일화적(anecdotal ; 사례나 개인적 경험을 바탕으로 하는) 증거는 평가에 있어 한계가 있어 정량적인 평가가 필요합니다. 

논문에서는 설명의 품질을 평가하기 위한 개념적 속성으로 Co-12 Explanation Quality Properties 를 소개하면서, 그 기준으로 300편 이상의 논문을 체계적으로 검토하여 평가한 결과를 설명합니다. 


### Method

#### Categorization of Explainable AI Methods

논문에서는 아래와 같이 XAI 방법을 6가지 디멘젼으로 구분합니다. 

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-03-29-XAI_Evaluation/01.png">

Type of Problem : XAI 가 풀 수 있는 4 종류의 문제

- (i) **Model Explanation** – globally explaining model 𝑓 through an interpretable, predictive model
- (ii) **Model Inspection** – globally explaining some specific property of model 𝑓 or its prediction
- (iii) **Outcome Explanation** – explaining an outcome/prediction of 𝑓 on a particular input instance
- (iv) **Transparent Box Design** – the explanation method is an interpretable model (i.e., 𝑒 = 𝑓 ) also making the predictions

Type of Method used to Explain : 모델을 설명하는 3 종류의 방법

- i) **Post-hoc explanation** methods (also called reverse engineering): explain an already trained predictive model
- ii) **Interpretability built into the predictive model**, such as white-box models, attention mechanisms or interpretability constraints (e.g. sparsity) included in the training process of the predictive model
- iii) **Supervised explanation training**, where a ground-truth explanation is provided in order to train the model to output an explanation.

#### Evaluation of XAI methods with Co-12 Properties

논문에서 정리한 설명 품질의 속성입니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-03-29-XAI_Evaluation/02.png">

- 01 Correctness : 진실성/충실성, 예측정확도가 아닌 설명정확도
- 02 Completeness : 모델 f 를 얼마나 설명하는지. 이상적인 것은 “the whole truth”
- 03 Consistency : 동일한 입력에 동일한 설명을 하는지
- 04 Continuity : 비슷한 입력에 비슷한 설명을 하는지, how continuous (i.e. smooth)
- 05 Contrastivity : 다른 대상이나 사건과 관련하여 비교를 용이하게 하는지
    - 사건을 설명할 뿐만 아니라 "발생하지 않은 *다른 사건과 비교하여*" 설명해야함
    - 서로 다른 모집단의 동일하지 않은 인스턴스는 서로 다른 설명을 가져야함
- 06 Covariate complexity : 설명에 사용된 covariates(특징) 는 인간이 해석가능한지
- 07 Compactness : 인간의 인지능력 한계때문에 요구되는 속성으로, 설명은 sparse, short and not redundant 해야함
- 08 Composition : 설명되는 *내용이* 아니라 설명되는 *방식에* 관한 것
    - 설명이 제시되는 방식이 '명확성'을 높일 수 있도록 해야함
- 09 Confidence :  certainty 나 probability 기준이 있는지
- 10 Context : 이해하기 쉬운 계획을 수립할 수 있는지
- 11 Coherence : 합리성, 타당성 및 "인간의 이성과의 일치"하는지
- 12 Controllability : 사용자가 설명을 어느 정도까지 제어, 수정 또는 상호작용할 수 있는지

### Experiments

저자는 300여개가 넘는 논문을 조사하여 사용한 설명법을 6가지 디멘젼, Co-12 속성에 맞추어 분류해서 다양한 방식으로 논문을 찾아볼 수 있는 사이트를 제공하고 있습니다. (현재도 논문은 계속 업데이트 되고 있으며, 누구나 논문을 등록할 수도 있습니다)

[https://utwente-dmb.github.io/xai-papers/](https://utwente-dmb.github.io/xai-papers/) 

논문에는 아래와 같은 방식으로 Co-12카테고리와 - XAI 방법 - 사용한 논문들을 정리해두어서 참고할 수 있습니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-03-29-XAI_Evaluation/03.png">

XAI 방법은 때로는 여러가지 Co-12 속성에 속하기도 하며 아래 테이블을 통해 확인할 수 있습니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-03-29-XAI_Evaluation/04.png">

### Discussion & Conclusion

XAI 의 속성을 보면 때로는 모순이 되는 속성도 존재합니다.

- Coherence vs Correctness : 일관 vs 정확
- Completeness vs Compactness : 완전 vs 간결

논문은, Co-12 를 모델 훈련 과정에 통합할 것을 제안하며, 표준화된 평가 지표의 중요성을 강조하고 있습니다.

### [Study] Questions & Discussion

스터디중에 나왔던 논의 주제 몇가지 소개합니다.

- Consistency 속성 경우 생성모델에는 적합하지 않은 것인가?
- 다양한 property들을 어떻게 aggregation할 수 있을까? Property간의 우선 순위가 있을까?
- Correctness, Completeness와 같은 점수가 높으면 시각적으로도 좋은 설명일까?
- Correctness, Completeness를 극대화하면 좋은 explainer를 얻을 수 있을까?