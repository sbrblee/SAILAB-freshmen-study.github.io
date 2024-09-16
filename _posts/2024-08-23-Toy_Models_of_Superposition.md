---
layout: post
title: "[논문리뷰] Toy Models of Superposition"
date: 2024-08-23 00:00:00 +0900
author: Jisu Yeo, Ted Jung
categories: ["Mechanistic Interpretability", "Antropic"]
use_math: true
---

** Explore when and how superposition occurs in a Transformer using a toy model. **

이 포스트에서는 아래 논문을 다룹니다.

Nelson Elhage, et al., (2022). [**Toy Models of Superposition**](https://transformer-circuits.pub/2022/toy_model/index.html). Transformer Circuits Thread.

# Summary

- Transformer의 미니 버전인 toy model을 통해 언제 그리고 어떻게 superposition이 일어나는지에 대한 구체적인 메커니즘을 시각적으로 이해하고 분석합니다. 
- Phase Change를 통해 특정 특징들이 superposition 상태에서 저장되는지 여부를 결정합니다. 
- Superposition이 digons, triangles, pentagons, tetrahedrons와 같은 기하학적 구조로 특징들을 조직한다는 것을 보여줍니다. 


# Introduction & Background

## (Section 1) Definitions and Motivation: Features, Directions, and Superposition
이 논문은 신경망의 표현에 대해 **linear representation hypothesis**를 가정.

- 신경망의 표현을 **선형적**으로 이해할 수 있다는 가설이 존재합니다. 즉, 입력의 **특징**이 **활성화 공간에서 방향으로 표현**된다는 것입니다.
- 이 가설의 중요한 두 가지 속성: **Decomposability, Linearity**
    - **Decomposability**: 신경망의 표현이 독립적으로 이해할 수 있는 특징들로 설명될 수 있음.
    - **Linearity**: 이러한 특징들이 활성화 공간에서 특정 방향으로 표현됨.

**Feature direction**을 식별하는 것이 쉬운 경우와 어려운 경우가 있는데, 이는 두 가지 주요 요인에 의해 결정된다. 

1. **Privileged Basis**: 일부 신경망 표현만이 basis와 일치하여 특징이 뉴런에 대응하도록 유도합니다.
2. **Superposition**: 신경망이 차원보다 더 많은 특징을 표현하기 위해 중첩을 사용하며, 이는 특징들이 뉴런과 대응하는 것을 방해할 수 있습니다.

**Superposition**이 신경망에서 발생할 경우, 이는 해석 가능성 연구에 큰 영향을 미칠 수 있으며, 따라서 이를 명확히 증명하는 것이 중요하다.


### Sparsity & Superposition

- Sparsity가 클수록 model은 dimension에 비해 더 많은 feature를 표현할 수 있다. (즉, superposition이 발생한다. )
- 나중에 언급이 되지만, sparsity가 클수록 feature가 **0이 될 확률이 높아서,  더 많은 특징들로 모델**을 표현할 수 있다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-1.png">


- 이 그림은 2차원에서 중요도가 서로 다른 5개의 feature embedding을 학습하는 토이 모델을 가정
- 첫번째 그림: 모델이 가장 중요한 두개의 feature만 표현하는 방법을 학습할 수 있다는 것을 보여줌. 나머지 덜 중요한 세개의 feature는 embedding 되지 않고, 중요한 2개의 feature만 orthogonal하게 표현.
- 두번째 그림: 가장 덜 중요한 feature 1개는 0에 매핑하고, 나머지 4개의 특징들은 최대한 서로 멀리 떨어질 수 있게 매핑.
- 세번째 그림: 이제 모든 특징들이 서로 간섭하며 하나의 공간에 오버랩되어 표현.
    
즉, sparsity를 높임으로써, 차원에 비해 더 많은 feature를 embedding할 수 있고, 대신 interference가 발생할 수 있음을 나타냅니다. 
    

### Empirical Phenomena

> 이 내용의 핵심은 **신경망이 입력 데이터를 특징(feature)로 표현**하고 **이 특징들이 방향(direction)으로 나타나는 것**으로, 이를 뒷받침하는 여러 **경험적 현상**이 소개됨.
> 
- **Word Embeddings**
    - 단어 임베딩에서 특정 벡터 연산이 의미적인 결과를 낳는 현상이 있습니다. 예를 들어, "왕 - 남자 + 여자 = 여왕"이라는 벡터 연산이 가능
- **Latent Spaces**
    - GANs 등에서도 유사한 벡터 연산과 해석 가능한 방향성이 발견됨.
- **Interpretable Neurons**
    - 신경망에서 특정 속성에 반응하는 뉴런들이 발견되었으며, 이들이 해석 가능하다는 연구가 많이 있다.
- **Universality**
    - 여러 신경망에서 동일한 속성에 반응하는 유사한 뉴런들이 발견됨.
- **Polysemantic Neurons**
    - 일부 뉴런은 다양한 입력에 대해 반응하는 경향이 있다.

### What are features?

> feature에 대한 명확한 정의가 어려워, 저자들은 세 가지 potential 정의를 제안하며, 마지막 정의를 염두에 둠.
> 
- Features as arbitrary functions
    - 특징을 입력에 대한 임의 함수로 정의.
- Features as interpretable properties
    - 특징을 인간이 이해할 수 있는 "concepts”으로 정의.
- Neurons in Sufficiently Large Models
    - 충분히 큰 신경망에서 특정 속성에 대해 뉴런이 할당될 때, 그 속성을 특징으로 정의.

### Features as Directions

> 신경망에서 특징(feature)이 방향(direction)으로 표현된다.
> 
- **linear representation**
    
    특징들이 활성화 공간에서 방향으로 표현될 때, 이를 선형적 표현이라고 함.
    
    장점은? 
    
    - **Linear representations are the natural outputs of obvious algorithms a layer might implement**
    - **Linear representations make features "linearly accessible."**
    - **Statistical Efficiency.**

- superposition은 **linear representation**이 많은 특징을 저장할 수 있다는 개념으로, 신경망이 더 많은 정보를 저장하고 처리할 수 있게 해준다. 

### Privileged vs Non-privileged Bases

> privileged basis에서는 뉴런의 해석 가능성(즉, 뉴런이 어떤 특징에 반응하는지 이해할 수 있는 가능성)이 더 높아지지만, non-privileged basis에서는 이러한 해석이 어려워진다.
> 

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-2.png">


### The Superposition Hypothesis

> 신경망이 제한된 수의 뉴런(즉, 제한된 차원)을 가지고도 더 많은 특징을 표현하기 위해 **고차원 공간**의 특성을 활용한다고 설명하며, superposition이 polysemantic 특성을 가진다를 설명.
> 
- **Polysemantic Neurons**
    - privileged basis가 있는 경우에도, 뉴런들이 여러 가지 관련 없는 특징들에 반응하는 경우가 종종 있다.
- mathematics:
    - **Almost Orthogonal Vectors**
        - 고차원 공간에서는 벡터들이 거의 직교하게 존재할 수 있습니다. 이러한 성질을 이용해 신경망은 다수의 특징을 효율적으로 표현할 수 있습니다.
    - **Compressed Sensing**
        - 고차원 공간에서 낮은 차원으로 투영된 벡터를 다시 원래의 고차원 정보로 복원할 수 있는 이론을 이용.
- **신경망의 시뮬레이션**
    - 작은 신경망이 더 큰 신경망을 noise이 섞인 상태로 시뮬레이션하는 것처럼, 중첩된 뉴런을 통해 다수의 특징을 표현할 수 있습니다.

# Method

## (Section 2) Demonstrating Superposition

### Experiment Setup

- Goal: Neural network가 높은 차원 벡터 → 낮은 차원 벡터 project → 다시 높은 차원으로 recover할 수 있는지를 탐구.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-3.png">


#### Feature Vector($X$)

- Create synthetic input data
- $x$는 고차원 입력 벡터로, 각 차원 $x_i$는 하나의 특징(feature)을 나타냄.
- 각 특징에는 Sparsity $S_i$ 와 Importance $I_i$가 할당됨.
- Sparsity $S_i$
    - 벡터 $x$의 각 차원 $x_i$는 하나의 특징을 나타내며, 이 특징이 특정한 확률 $S_i$ 로 **0이 될 확률**을 뜻함.
    - 즉, Sparsity가 높을수록 이 특징이 0일 확률이 높아짐.
- 각 특징 $x_i$는 $S_i$ 확률로 0이 되고, 그렇지 않을 경우에는 [0, 1] 범위에서 균등하게 분포된 값을 가짐.
- In practice, they focus on the case where all features have the same sparsity $S_i = S$.

#### Model ($X$→ $X^\prime$)

→ Superposition hypothesis를 검증하기 위해 두 가지 모델을 비교.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-4.png">


- Linear Model: superposition이 발생하지 않음.
- ReLU Output Model: superposition이 발생.

이 모델을 사용한 이유?

- $W$
    - **Superposition hypothesis**에 따르면, 고차원에서의 특징들이 저차원 공간에서 특정 방향과 대응되며, 이를 **선형 맵핑**을 통해 $h=Wx$로 표현할 수 있습니다.
    - 이때, 각 열 column $W_i$는 저차원 공간에서 특정 특징($x_i$)에 해당하는 방향을 나타냄.
- **$W^T$**
    - **원래 벡터를 복원**하기 위해 사용.
- **$b$**
    - **바이어스 추가**는 표현되지 않은 특징들을 처리하고, 작은 양의 noise를 제거하기 위함.
- **$ReLU$**
    - **활성화 함수**는 Superposition의 발생 여부를 결정짓는 중요한 요소이다. 실제 신경망에서는 활성화 함수가 계산에 사용되므로, 이를 포함하는 것이 합리적.

#### Loss function

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-5.png">


- 여러 input(x)이 있고, 각 input의 여러 feature들($x_i$)에 대한 예측, 실제 차이를 나타냄.
- 더 중요한 feature에 더 많은 가중치를 부여함으로써, 중요한 특징을 더 기억하는 역할인듯

### Basic Result

> 앞서 가정한 input에 대해 model(linear, ReLU)을 학습한 후, 학습 결과를 시각화함. 모델이 학습한 특징들에 중첩이 발생하는지 실험적으로 확인.
> 

**모델의 결과 시각화하는 간단한 방법**

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-6.png">

- $W^TW$
    - **가장 중요한 특징**들이 identity matrix로 나타나며, 덜 중요한 것들은 0으로 표시됨.
- $b$
    - 바이어스 벡터로, 일부 특징에 대해 0이 되고, 나머지에 대해서는 expected value을 가짐.

**Superposition 현상의 시각화**

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-7.png">

- 앞서, $W_i$는 특정 특징($x_i$)에 해당하는 방향을 나타낸다고 했음.
- 특징이 완전히 표현되었다면 norm $||W_i||$ 값이 1을 가지고, 표현되지 못했다면, 0을 가짐.
- 특징이 다른 특징들과 차원을 공유하는지를 이해하기 위해, 다른 모든 특징을 해당 특징의 방향 벡터에 투영한 값을 계산. $\sum_{j \neq i} (\hat{W}_i \cdot W_j)^2$ 이 값이 0에 가까우면 그 특징이 다른 특징들과 직교(독립적)하다는 것을 의미하며(중첩 없이 그 특징이 고유의 차원에서 표현되며), 1 이상이면 다른 특징과 차원을 공유한다고 보면된다.

**Linear Model vs ReLU Output Model**

여기서는 특징의 개수(n = 20), dimension (m=5), 중요도($I_i = 0.7^i$)를 설정

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-8.png">

- 그림 해석
    
    $||W_i||$ 에서 막대바는 특징이 얼마나 표현되었는지를 나타냄. (막대바가 조금만 있는건 특징이 완전히 다 표현되지는 않았다는 것. )
    
    막대바 색깔은 $\sum_{j \neq i} (\hat{W}_i \cdot W_j)^2$ 그 특징이 다른 특징들과 직교하는지를 나타내는 것. 모두 직교하면 값이 0으로 검은색으로 표시되고, 직교하지 않으면, 중첩이 있다는 뜻이고, 값이 커질수록 점점 노란색으로 표현됨. 
    
    bias는 reconstruction error를 줄이기 위함. ex) (1,0,0)에 $W$ 곱해서 더 작은 차원으로 매핑 → $W^T$ 곱해서 더 높은 차원으로 돌리는 과정에서 $W^TW$가 양수인 부분 때문에 (1, 0.3, 0.4) 처럼 제대로 복원이 안 돼서 bias가 (0, -0.3, -0.4)이렇게 학습되었을 것이다! 
    
    $W^TW$를 보면, sparsity가 1에 가까워질수록(커질수록),특징 xi가 0이 될 확률이 높다는 것이고, 그러면 더 많은 특징들로 표현을 할 수 있다. 더 많은 특징들로 표현하려다 보니, 그래서 $W^TW$ 에 0이 아닌 부분이 많이 생기는 기는 것이고,  orthogonal하지 않은 feature들이 생겨나는 것이다. 
    

**Linear Model**

- 항상 **가장 중요한 m개의 특징**을 학습. (PCA에서 가장 큰 주성분을 학습하는 것과 유사)
- 모델이 학습하는 특징은 직교(orthogonal) 상태로, 서로 간섭이 없음.

**ReLU Output Model**

- **Dense (1 - S = 1.0)**: Sparsity가 0인 경우, ReLU 모델도 선형 모델과 비슷하게 가장 중요한 특징을 학습.
- **Sparsity 증가**함에 따라, $W^TW$에 0이 아닌 부분이 많이 생겨서, 여러 feature들이 저차원에서 orthogonal하지 않게 배치하게 된다. ⇒ superposition

### Mathematical Understanding

> 앞에서는 superpostion 현상이 나타난다는 것을 실험적으로 확인했다. 지금부터는 수학적으로 왜 이런 현상이 발생하는지를 이해해본다.
> 

**Linear model**

- Definition of loss function

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-9.png">

- Feature benefit
    - 특정 feature를 정확하게 표현할 때 얻을 수 있는 성능 향상.
    - 만약 $||W_i||$ 가 1에 가까우면, 그 특징이 잘 표현되고 있다는 의미이고, 손실 L은 줄어든다.
- Interference
    - 두 feature가 완전히 직교하지 않으면 내적이 0이 아니게 되고, 이에 따라 간섭이 발생하여 손실이 증가함.

→ 이 두 가지 힘의 균형은 모델이 얼마나 많은 특징을 학습할 수 있는지를 결정

- 선형 모델에서 Superposition이 발생하지 않는 이유
    - 만약 모델이 더 많은 특징을 표현하려고 한다면, interference가 증가하여 성능이 저하됨.

**ReLU model**

- Definition of loss function
    
    <img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-10.png">
    
    여기서 x는 $x_i = 0$ 일 확률이 $S$인 분포에 따라 분포됨. 
    
    손실 함수는 이항 전개 binomial expansion $((1-S) + S)^n$를 통해 각 Sparsity 패턴에 대한 항으로 분해됨. 
    
    즉, 각 k-sparse 벡터(k개의 요소가 0이 아닌 벡터)에 대한 손실이 각각의 항으로 표현됨. 
    
    <img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-11.png">
    
    만약 S가 1로 가면, L1과 L0가 우세한 term이고, 이때 L0는 zero vector에 대한 손실함수라, L1 term을 유심히 살펴보면 된다. 
    
- $L1$
    
    : 1-sparse 벡터에 대한 손실. (0이 아닌 요소가 하나만 있는 벡터)
    
    <img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-12.png">
    
    **Feature benefit** term: bias 역할은 모델이 feature를 표현하지 못하는 경우, expected value를 가질 수 있도록한다. 
    
    **Interference** term: ReLU가 negative interference를 허용하여, **모델이 negative interference를 선호하게 만든다.** 
    

# Experiments

## (Section 3) Superposition as a **Phase Change**

> **Superposition** 현상이 물리학에서의 Phase Change와 유사하게 작용할 수 있다는 가설을 탐구. 특히, 모델이 학습하는 특징이 어떻게 다르게 표현될 수 있는지를 이해하기 위한 실험을 제안.
> 

모델을 학습시킬 때, 특징은 세 가지 방식으로 표현될 수 있습니다.

- Features is not learned
- Feature is learned, and given a dedicated dimension
- Feature is learned, and represented in superposition
    
    <img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-13.png">
    
    ⇒ 이 세 가지 결과 사이의 전환은 급격하게 일어나며, 이는 마치 물리학에서의 Phase Change와 유사한 현상으로 볼 수 있습니다.
    

[실험 설정]

- n = 2 (input feature가 2개), m = 1 (hidden layer dimension이 1) 을 가진 ReLU output model  $ReLU(W^TWx-b)$  을 가정.
- 첫 번째 feature의 중요도는 고정값 1.0이고, 두 번째 feature의 중요도는 0.1에서 10까지 변화시킴.  → x축이며, 
y 축은 모든 (첫번째, 두번째 ) feature의 Density를 0.01 to 1.0로 변화시킨 것.
    
    두 번째 feature가 위 세가지 결과 중 어떻게 학습되었는지를 관찰함. 
    
    <img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-14.png">
    
- 이 실험 결과를 이론적인 "toy model"과 비교하면 다음과 같다.
    
    2개의 feature를 1 dimension에 넣는 세가지 방법이 있다.
    
    - W = [1, 0] → 두번째 feature를 버리는 것.
    - W = [0, 1] → 첫번째 feature를 버리는 것.
    - W = [1, -1] → 두 feature 모두 고려하는 것. superposition에 넣는 것. ⇒ 이 경우 두 벡터는 서로 반대 방향으로 배치되며, 이 방법을 "양극 쌍(antipodal pair)"이라고 부름.
    
    <img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-15.png">
    

## (Section 4) The Geometry of Superposition

### Uniform superposition

이 실험에서는 다음과 같은 설정을 함. 

- 모든 feature들이 동일한 importance와 sparsity를 갖는다.
- n = 400(features), m = 30(hidden dimensions)
- 모델이 학습한 특징의 수는 **Frobenius norm** $\| W \|_F^2$ 을 사용하여 측정. 특징이 표현된 정도를 측정하며, 1에 가까울 때 특징이 잘 표현된거고, 0에 가까울 때 특징이 잘 표현안된 것.
- Average number of **dimensions per feature - 특징당 차원 수**
    
    <img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-16.png">
    
    - 그래프
    
    <img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-17.png">
    
    S를 바꿔가면서 $D^*$을 plot한 그래프.  
    
    sparsity가 증가하면 모델이 학습한 특징의 수가 증가해서, $D^*$ 이 감소함. 
    
    이때, 1과 1/2에서 "sticky"한 지점을 볼 수 있는데, 이는 모델이 이 지점에서 특징을 얼마나 효과적으로 압축할 수 있는지를 보여줌. 
    
    - 1에서는 각 특징이 독립적으로 표현되는 상태.
    - 1/2에서는 antipodal pairs과 관련. 두 특징이 서로 정확히 반대되는 값으로 설정되어, 한 차원에 두 개의 특징을 담을 수 있다고 해석됨.

#### Feature Dimensionality

- Dimensionality of the $i$th feature (특징의 차원성)
    
    조금 전 $D^*$는 모델이 표현하는 특징당 차원수를 평균 낸 것. 
    
    $D_i$는 개별 i번째 특징에 대한 dimensonality를 나타낸 것. 
    
    <img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-18.png">
    
    - $W_i$ : weight vector column associated with the $i$th feature
    - $W$  hat : unit version of that vector
    
    분자는 주어진 특징이 얼마나 잘 표현되었는지를 나타내며, 분모는 그 특징이 다른 특징들과 차원을 얼마나 공유하는지를 나타냄. 
    
    ex) antipodal pair의 경우, 학습된 feature는 $D_1$ = 1/(1+1), 학습되지 않은 feature는 $D_2$  = 0 
    

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-19.png">

- 위에서 사용했던 선형 플롯 사용. 이 선형 플롯 위에 개별 feature의 차원성에 대한 scatter plot을 오버레이한다. (각 피처가 다양한 희소성 수준에서 모델에 대해 가지는 차원성을 표시하는 것.)
- 특징의 차원성이 특정 비율에 클러스터링되는 경향이 있고, 이는 특정 기하 구조에 해당.
- 이 구조들을 시각적으로 보여주기 위해 Feature Geometry Graph를 생성합니다. 여기서 각 node는 특징을 나타내며, edge 가중치는 특징 임베딩 벡터의 내적(dot product)의 절대값으로 표현다. 특징들이 직교하지 않으면 연결됩니다.

#### Why These Geometric Structures?

> 모델이 학습한 특징들이 특정한 기하학적 구조를 따르며, 이러한 구조들이 Thomson 문제의 해결책과 관련이 있다는 내용. ⇒ Tegum Product를 통해 다양한 차원에서 특징을 배치하고, interference를 최소화하려는 경향이 있다.
> 
- Thomson problem와의 연관성
    
    이 기하학적 구조는 톰슨 문제(Thomson problem)의 해결책으로 나타날 수 있다. 
    
    Thomson problem는 구의 표면 위에 점들을 배치할 때, 그들이 서로 밀어내는 전하처럼 작용하여 에너지를 최소화하는 방식으로 배치하는 문제입니다.
    
- Tegum Product
    
    Thomson 해답은 Tegum Product로 이해될 수 있다.  이는 두 다면체를 직교하는 차원에서 임베딩하여, 작은 균일 다면체로 만드는 연산입니다.
    
    <img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-20.png">
    
    예를 들어, 삼각형 이중피라미드는 삼각형과 양극 쌍의 Tegum Product로, 오각형 이중피라미드는 오각형과 양극 쌍의 Tegum Product로 나타납니다. 이런 방식으로 두 개의 하위 그래프는 서로 간섭하지 않습니다.
    

### Non-uniform superposition

이전에 다룬 uniform superposition과 달리, 모든 특징이 동일한 importance나 sparsity를 가지지 않고, 그로 인해 superposition의 기하학적 구조가 왜곡된다. 

[Highlight some phenomena we observe]

- 특징들이 중요도(importance)나 희소성(sparsity)이 다를 경우, 다면체가 부드럽게 변형되다가 특정 임계점에서 급격히 다른 다면체로 변환됩니다.
- **Correlated Features**는 주로 함께 발생하는 특징들이며, 모델은 이들을 **직교하거나 or** **가까이 배치**하여 양의 간섭(Positive Interference)이 발생하도록 선호.
- **Anti-correlated Features**는 함께 발생할 가능성이 낮으며, 모델은 이들 사이에 음의 간섭(Negative Interference)이 발생하도록 구성하는 것을 선호.

#### Perturbing a Single Feature

> **하나의 특징만을 변형**시키는 실험을 설명. **다섯 개의 특징**을 가지고 있으며, 이 중 하나의 특징의 sparsity을 변화시키면서 나머지 특징들을 균일하게 유지하는 방식으로 진행.
> 

**실험 설정**:

- n = 5 개의 특징을 m = 2개의 차원에 표현.
- 초기 상태에서는, 모든 특징이 균일한 중요도(I = 1)와 활성화 density(1 - S = 0.05)를 가지며, 이때 정오각형(Pentagon)의 구조를 형성.
- 특정한 하나의 feature의 Sparsity를 변화시키다보면,  Pentagon ↔ Digon 구조로 전환이 됨.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-21.png">

### Correlated and Anti-correlated Features

#### Setup For Exploring Correlated and Anti-correlated Features

- Correlated Feature Sets
    - 상관된 특징 집합은 함께 발생하는 경향이 있는 특징들의 묶음으로 생각할 수 있습니다.
    - 이를 시뮬레이션하기 위해, 모든 특징들이 한 번에 활성화되거나 비활성화되도록 설정.
- Anticorrelated Feature Sets
    - 반상관된 특징 집합은 함께 발생할 가능성이 매우 낮은 특징들로 구성됩니다.
    - 이를 시뮬레이션하기 위해, 하나의 특징이 활성화될 때 다른 특징들은 비활성화되도록 설정합니다.

#### Organization of Correlated and Anti-correlated Features

- **상관된 특징(Correlated Features)의 직교적 배치**:
    - 모델은 상관된 특징들이 서로 간섭하지 않도록 가능한 한 직교하게 배치하려고 합니다.
- **반상관된 특징(Anticorrelated Features)의 반대 방향 배치**:
    - 모델은 반상관된 특징들이 서로 반대 방향에 위치하도록 하여, 이들 사이에 음의 간섭이 발생하도록 구성합니다.
- **직교할 수 없을 때 상관된 특징(Correlated Features)의 나란한 배치**:
    - 모델은 상관된 특징들이 가능한 한 가까이 배치되도록 구성합니다. 만약 공간에 충분한 차원이 없다면, 이들 특징들은 나란히 배치되며, 이는 양의 간섭(Positive Interference)이 발생할 수 있음을 시사.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-22.png">

## (Section 5) Superposition and Learning Dynamics

- Toy model이 학습 과정에서 evolve하는지 관찰
- Fully train된 model들은 simple structure로 수렴한다 (converges)
    - 이 구조를 이해함으로써 학습 과정에서의 evolution을 더 이해함
- Superposition이 무슨 학습 단계에서 나타나는지 이해하려고 한다
    - 이 article은 몇 가지 현상만 연구 한다

#### Phenomenon 1: Discrete “Energy Level” Jumps

- Learning dynamics는 “energy level jump”에 의해 지배되는 것으로 보인다
    - 특징들이 서로 다른 특징 차원 간에 jump한다
        - 차원성이 서로 자리를 바꾼다
        - Loss curve가 급격히 하락한다
    - Loss curve의 부드러운 감소가 여러 작은 특징 jump들로 구성되어 있다고 추측한다

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-23.png">

#### Phenomenon 2: Learning as Geometric Transformations

- 일부 경우에서 기하학적 구조로 이어지는 learning dynamics를 sequence of geometric transformations로 이해할 수 있다
    - Correlated features - 학습이 loss curve에서 관찰 가능한 여러 단계로 진행된다

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-24.png">

## (Section 6) Relationship to Adversarial Robustness

- Model without superposition: $(W^TW)_0=(1,0,0,0,…)$
- Model with superposition: $(W^TW)_0=(1,\epsilon, -\epsilon,\epsilon,…)$
    - Adversary는 가장 중요한 특징인 $\epsilon$을 공격할 수 있다 (interference)
- 각 특징에 대한 optimal L2 attack를 고려하며 adversarial attack를 분석적으로 도출한다
    - 성능에 가장 큰 영향을 미치는 공격을 선택
    - Superposition이 형성됨에 따라 adversarial example에 대한 취약성이 (vulnerability) 급격히 증가한다
        - 차원당 특징수 ($\frac{1}{특징 차원성}$)와 비슷하게 연관된다

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-25.png">

- Superposition과 관련된 여러 현항 (phenomena)로부터 adversarial example를 예측할 수 있다
    - 이러한 예제들이 존재하는 이유를 설명 할 수 있다
    - Adversarially robust model은 성능이 더 나빠질 수 있다 (superposition을 포기하기 때문)
        - 하지만 해석이 더 가능할 수도 있다
- 반대로 adversarial training을 통해 superposition을 줄이려는 시도 해 봤다
    - Superposition이 줄어들긴 했지만 attack를 아주 크게 만들어야 했다
    - 하지만 더 강한 adversarial attack이 더 효과적일 수 있다

## (Section 7) Superposition in a Privileged Basis

- Neural network representation에서 neuron이 privileged basis를 부과 (impose)하는 경우 (transformer MLP layers, conv net neurons, etc.)
- Hidden layer에 ReLU를 추가한다
    - $W$는 직접 해석이 가능 (특징을 basis-aligned neuron에 연결)
    - Input = features, basis elements in the middle layer = neurons
        - $W$는 특징을 neuron에 mapping한다
        - 특징이 구조적으로 neuron과 align된다
        - 많은 neuron이 하나의 특징을 표현한다 (monosemantic)

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-26.png">

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-27.png">

#### Visualizing Superposition in Terms of Neurons

- 이제 $W$를 직접 검사할 수 있다
    - Per-neuron stacked bar plot
        - 각 column은 $W$의 한 column을 표현
        - 각 사각형은 하나의 가중치 항목을 나타낸다
            - 높이는 절댓값에 해당한다
        - 사각형의 색깔은 작용하는 특징을 나타낸다

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-28.png">

- 어떤 sparsity 값 범위를 보면 그 data로 훈련된 모든 모델의 optimal solution은 같은 weight configuration을 가진다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-29.png">

- Sparsity level이 높아질수록 monosemantic에서 polysemantic neuron으로의 전환이 발생한다
    - Neuron-level phase change

#### Limitations of the ReLU Hidden Layer Toy Model Simulating Identity

- 문제 - 모델은 hidden layer를 안 쓸때도 있다
    - 강제로 ReLU를 사용할 경우에만 사용한다
    - “For example, given a hidden layer bias, the model will set all the biases to be positive, shifting the neurons into a positive regime where they behave linearly. If one removes the bias, but gives the model enough features, it will simulate a bias by averaging over many features.”
    - Useful as a simpler case study

## (Section 8) Computation in Superposition

- 가설 - neural network는 superposition 상태에서 계산을 할 수 있다
- 새로운 설정
    - Hypothetical disentangled model의 input & output layers
        - Hidden layer는 더 작은 layer이 된다 (observed model)
        - 모델은 hidden layer non-linearity를 무조건 사용해야 한다
    - 모델이 계산을 수행하도록 한다  ($y = |x|$)
        - ReLU neuron을 사용하여 계산하는 간단한 방법: $|x| = ReLU(x) + ReLU(-x)$

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-30.png">

### Experiment Setup

- Input vector은 아직 sparse

### Basic Results

- ReLU 때문에 $W_2^TW_1$만 연구할 수는 없다
- Neuron 측면에서 가중치를 visualize한다
    - Or a stack plot (전과 같이) or a graph

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-31.png">

- 모델은 양수 neuron ReLU와 음수 neuron ReLU를 구성한다
- Without superposition, the model needs two hidden layer neurons to implement absolute value on one feature.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-32.png">

### Superposition vs Sparsity

- 가중치의 절대값을 나타내는 stack plot을 보여준다
    - Polysemanticity에 따라 색이 흐릿하게 표시된다

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-33.png">

- Activation function은 privileged basis를 생성한다 (monosemantic)
    - 특징이 충분히 sparse해지면 이 모델도 superposition을 사용한다
        - Superposition으로 표현된 data를 neural network이 계산할 수 있는 기능
- 관찰된 행동
    - 같은 모델 내에는 monosemantic이랑 polysemantic neuron이 둘 다 존재 할 수 있다
        - 가장 중요한 특징을 나타내는 neuron은 monosemantic인 경향이 있다
    - Neuron은 primary 특징과 작은 가중치로 encoding된 secondary 특징이 연관되어 있는 것으로 보인다
        - 작은 activation은 more polysemantic
        - 많은 neuron은 가장 강한 activation으로 해석 가능하지만 다른 의미나 pattern에 대해서도 activate 되는 것으로 보일 수 있다

## (Section 9) The Strategic Picture of Superposition

- The article believes that superposition is connected to the challenge of using interpretability to make claims about the safety of AI systems

### Safety, Interpretability, & “Solving Superposition”

- 모델이 manipulate하거나 deliberately deceive 하지 않을 것이라는 confidence
    - “Unknown unknowns”을 해결하는 방법 - superposition
- 특징을 열거하는 것은 superposition와 얽혀 있다
    - 특징이 neuron에 해당하면 neuron을 열거함으로써 특징을 열거할 수 있다
        - 그러면 compressed sensing을 수행하여 모델의 activation을 더 큰 non-superposition 모델의 activation로 “펼칠” 수 있다
- “Solution to superposition”
- Other properties
    - Activation space 분해
        - 특징을 식별하는 것이 모델을 분해할 수 있게 해준다
        - 차원의 저주를 극복한다
    - 특징 측면에서 activation 설명
        - 특징이 기준 차원과 정렬될 때 activation을 개별 기본 특징으로 분해할 수 있다
    - 가중치 이해 (circuit analysis)
        - Neural network 가중치는 이해 가능한 특징에 연결 될 때만 이해될 수 있다
            - 이를 위해 superposition을 해결해야 한다

### Three Ways Out

- Superposition을 해결할 접근 방법
    - Superposition이 없는 모델 생성
    - 특징이 어떻게 표현되는지 설명하는 overcomplete basis를 찾기
    - Hybrid approaches - 다음 분석 단계가 더 쉬워지도록 만듬
- 모델의 유효성이 상관 없다면 가능하다
    - Superposition 제거는 할 수 있다
    - 문제는 제거하는 cost가 너무 높은지 여부이다

#### Approach 1: Creating Models Without Superposition

- Apply L1 regularization to the hidden layer activations
    - 특정 중요도 임계값 이하의 특징을 제거한다
    - Basis-aligned word embedding을 장려하기 위해 sparsity를 사용하는 것과 비슷하다
- Sparsity는 모델을 더 크게 만든다 - high performance cost

#### Approach 2: Finding an Overcomplete Basis

- 일반 모델 사용, 특징이 어떻게 embedding되어 있는지를 설명하는 overcomplete basis를 나중에 찾는다
- 장점: 모델 성능에 손상을 주는지 걱정 필요 없음
- 단점
    - 얼마나 많은 특징을 알아야 하는지 알기 어렵다
    - 원래 surface structure (neurons, attention heads, etc.) 으로 neural network를 이해 할 수 있는데 이제는 “virtual neuron”이 되면서 이해 하기가 많이 어려워 졌다
    - Interference는 더 이상 유리하게 작용하지 않는다
        - Superposition은 이미 훈련 과정에서 고정되어 있다
        - 나중에 superposition을 decoding하려고 하면 objective가 오히려 반대 할 수 있다

#### Approach 3: Hybrid Approaches

- Superposition을 training time 후에 순수하게 다루는 접근
    - Superposition이 적은 모델을 생성
    - Overcomplete basis를 찾기 쉽게 하기 위해 architecture을 변경한다

### Additional Considerations

- Phase change as cause for hope
    - Superposition은 유용하기 때문에 제거하기 어렵다
    - 하지만 phase change와 관련이 있으면 superposition이 없는 공간으로 모델을 “push”하면 superposition 없는 모델을 만들 수도 있을거다
- Superposition-free 모델은 연구에 쓸모있는 도구가 될 것이다
    - 성능이 뛰어난 superposition이 없는 모델을 만들 수 있을까?
    - Ground truth이 없기 때문에 현재는 도전적인 상황
- Local basis만으로는 부족하다
    - 모델은 종종 local orthogonal basis를 형성한다
    - 전제 분포에 대해 적용 가능한 mechanistic 설명이 필요하다
        - Local basis에서 나올 가능성은 낮다


# Discussion & Conclusion

## 10. Discussion

### To What Extent Does Superposition Exist in Real Models?

- Toy model은 superposition을 연구하는데 유용한 예제다
- Polysemanticity와 Emperical observation의 일관성:
    - Polysemantic neurons exist
    - Neuron은 때로 해석 가능하고 때로는 polysemantic이며 종종 같은 layer에서 발생한다
    - Inception V1의 later layer에는 polysemantic neuron이 더 자주 보인다
        - Polysemantic neuron의 비율이 깊이에 따라 증가한다
    - 초기 Transformer MLP neuron은 극도로 polysemantic하다
        - Very sparse features - 모델은 polysemanticity를 예측할 것이다
- 일부 현상은 generalize될 가능성이 높지만, 일부는 불확실하다

### Open Questions

- Superposition을 잡기 위한 statistical test이 있을까?
- Superposition와 polysemanticity의 발생을 제어하는 방법은 뭘까?
    - 특징이 phase diagram의 superposition section에 떨어지지 않도록 하는 것
        - L1 regularization of activations, adversarial training, changing the activation function
- Closed form solution을 가지고 있는 superposition 모델이 있을까?
- Toy model은 얼마나 현실적인가?
- 실제 모델의 특징 중요도 곡선이나 feature sparsity 곡선을 추정할 수 있을까?
    - Toy model을 실제 모델로 generalize하기
- 충분히 확장하면 superposition이 사라질까?
- 가장 원칙적인 것들을 측정하고 있을까?
    - Superposition/polysemanticity의 가장 원칙적인 정의는 무엇인가?
- Polysemantic neuron은 얼마나 중요한가?
    - x%의 neuron을 이해한다면 전체적으로 얼마나 이해한 것일까?
- 우리가 특징/neuron에서 관찰하는 phase change이 compressed sensing의 phase change와 연결 할 수 있을까?
- Superposition은 non-robust 특징과 어떻게 관련되는가?
    - Data의 주성분에서 유용성과 강건성 사이의 tradeoff
- Neural network이 superposition에서 유용한 계산을 얼마나 잘 수행할 수 있을까?
- 특징이 독립적이지 않으면 superposition은 어떻게 변할까?
    - 특징이 반상관적일 때 superposition이 더 효율적으로 특징을 packing할 후 있을까?
- 모델이 비선형 표현을 효과적으로 사용할 수 있을까?

## 11. Related Work

- Interpretable features
- Superposition
- Disentanglement
- Compressed sensing
- Sparse coding and dictionary learning
- Theories of neural coding and representation
- Additional connections

# [Study] Questions & Discussion

## 내용 이해를 위한 질문

- 수빈
    - Demonstrating superposition >> bias vector가 0 또는 expected value인 이유?
        
        <img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-34.png">
        
    - The Geometry of Superposition >> 1/2 ⇒ antipodal effect
        
        <img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-35.png">
        
    - Computation in superposition >> sparsity가 높은 이 모델에서 정확도는 괜찮게 나오는지 우려가 됨
        
        <img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-23-Toy_Models_of_Superposition/Untitled-36.png">

- 연지
    - Basic results 그림 해석이 궁금합니다.
        - 1. W 의 norm 은 포함할수 있는 피쳐수?
        2. 빨간색(1) 이라는건 순수하게 그 특징만 있고 다른게 안섞였다는 뜻?
        3. 파란색 -1 은 어떻게 해석을 해야할까?
        4. b 의 의미를 잘모르겠음..
        - 1-S = 0.3 그림에서,
        아래쪽 대각선으로 활용을 늘려가면 될텐데 왜 옆으로 침범하기 시작하지?
    - Mathematical Understanding 에서,
        - ReLU 에서 1-sparse case 에서 negative interference 가 발생하지 않는다는부분 이해가 안되요
        ⇒ ReLU 의 특성과 관련

## 다같이 생각해보면 좋은 질문

- 수빈
    - sparsity의 존재는 고차원 데이터가 더 낮은 차원의 geometry에 존재한다는 manifold hypothesis와도 관련이 있는 것인지?
