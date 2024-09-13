---
layout: post
title: "[논문리뷰] Transformers are RNNs"
date: 2024-07-19 10:30:00 +0900
author: Seongwook Chung
categories: ["Transformers", "Linear Attention"]
use_math: true
---
**Fast Autoregressive Transformers with Linear Attention**

이 포스트에서는 아래 논문을 다룹니다.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret. [**Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention**]
(https://arxiv.org/abs/2006.16236). Proceedings of the 37th International Conference on Machine Learning, Online, PMLR 119, 2020

### Summary

1. Transformer가 매우 길게 구성된 sequence에 대해 2차 방정식 연산의 복잡도(complexity)로 인해 느리다는 문제점을 제기합니다.
2. self-attention을 linear dot-product of kernel feature map으로 표현하고 행렬곱의 연관성을 활용해서 복잡도(complexity) 를 O(N제곱)에서 O(N)으로 줄였습니다.
3. 이는 autoregressive transformer를 가속화하고 RNN과 연관 있음을 보여줍니다.
(autoregressive model : 예측 시 이전 시점의 데이터 이용해서 현재 시점의 값을 예측하는 모델 → 시계열 데이터 또는 자연어 처리)
4. 구현한 linear transformer 가 기본 vanilla transformer보다 4000배 빠릅니다.

# Introduction & Background

1. self-attention : 크기 N 만큼의 input에 대해서 2차(제곱) 크기의 메모리를 사용, 이에 시간 복잡도가 O(N제곱)입니다. 
2. 이를 해결하는 연구가 있었고(Transformer-XL) 다만 연산이 많이 소요되었습니다.
    
    [arxiv.org](https://arxiv.org/pdf/1901.02860)
    
3. Sparse factorizations : O(N*squared root(N))
    
    [arxiv.org](https://arxiv.org/pdf/1904.10509)
    
4. Locality sensitivity hashing : O(N*log(N))
    
    [arxiv.org](https://arxiv.org/pdf/2001.04451)
    
5. Linear transformer : author’s introduction
    1. Result 
        1. Reduce the memory footprint
        2. Scale linearly with respect to the context length
        3. Relation between transformers and RNNs
        (perform autoregressive inference orders of magnitude faster)
    2. How
        1. kernel-based formulation (of self-attention)
        2. The associative property of matrix products (to calculate the self-attention weights) (Section 3.2)
        3. Causal masking with linear complexity and constant memory (Section 3.3)
6. Evaluation
    1. Image generation
    2. Automatic speech recognition

### Method

# Linear Transformers

제안하는 linear transformer를 공식화하는 섹션입니다.

- attention
    - 기존 : Traditional softmax attention
    - 제안 : Feature map based dot product attention results

시간과 메모리 복잡도 뿐 아니라 인과 모델 측면에서 선형 시간에서 시퀀스 생성을 수행하는 것은 RNN(recurrent neural network)과 유사합니다.

## 3.1 Transformers

기존의 Transformer의 수식은 어떠한지 확인합니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled.png">

fl 함수 : 각각의 feature vector를 독립적으로 변환, 보통 2개 layer의 feedforward network로 구현합니다.
Al 함수 : self-attention layer

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-2.png">

V’ : 기존의 value vector V를 attention 가중치와 곱해서 얻은 새로운 value vector

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-3.png">

수식(2)에서 수식(3)으로의 유도 과정입니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-4.png">

## 3.2 Linearized Attention

수식(2)의 attention을 보다 효율적으로 계산하고자 수식(3)을 선형화(linearization)할 수 있는 방법을 제안하고 → 커널 함수 ϕ(x)를 도입합니다. ( 각각의 벡터를 고차원의 특징 공간으로 매핑 → 내적으로 유사도를 계산)

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-5.png">

두 벡터 x와 y를 입력 받아 음수가 아닌 값을 출력하는 커널 함수입니다.(0 또는 양수 값을 반환)

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-6.png">

2개의 F차원 벡터 공간, x와 y가 각각 F차원의 벡터입니다. (x,y를 함께 나타내는 공간은 2 x F 차원)

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-7.png">

Non-negative 실수 공간 : 커널 함수 k(x,y)의 결과는 0 또는 그보다 큰 값을 나타냅니다.

기존 커널 방법 : k(x,y)는 두 입력 벡터 x와 y 사이의 유사도를 고차원 특징 공간에서 계산하는 함수입니다.
(reference : https://velog.io/@nawnoes/Performer-RETHINKING-ATTENTION-WITH-PERFORMERS )

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-8.png">

* 대표적으로 다항식 커널(polynomial kernel)과 RBF 커널(Radial Basis Function Kernel) 이 있습니다.
  (reference : https://velog.io/@nawnoes/Performer-RETHINKING-ATTENTION-WITH-PERFORMERS )
   - 다항식 커널의 커널 함수
<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-9.png">
   - RBF 커널 함수
<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-10.png">

* 본 논문은 기존 커널 이론을 Transformer의 attention 매커니즘에 적용합니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-11.png">

수식(4)는 커널 함수  ϕ(x)를 사용해서 쿼리와 key 벡터를 매핑합니다.
그러면 sim(Qi, Kj)를 ϕ 함수를 통해 다음과 같이 표현합니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-12.png">

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-13.png">

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-14.png">

전통적인 softmax attention인 경우에는 모든 쿼리 Qi에 대해 모든 Key Kj와의 유사도를 계산해야 되어서 O(N제곱x D)의 복잡도를 가짐, 참고로 D는 쿼리/키 벡터의 차원입니다.

수식(5)의 경우 최초에
→ 분자의 KjVj에 대한 sigma를 사전 계산하고 이 때에는 O(NxD제곱)의 복잡도를 가집니다.
→ 분모의 Kj에 대한 sigma를 미리 계산할 때에는 O(ND)의 복잡도를 가집니다.
그 다음에 어텐션을 적용하면서  ϕ(Qi)에 대해서는 사전 계산된 합을 사용하므로 O(ND)의 복잡도를 가집니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-15.png">

수식(5)를 벡터화하면 아래 수식 (6)이 됩니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-16.png">

### 3.2.1 FEATURE MAPS AND COMPUTATIONAL COST

- softmax attention의 complexity

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-17.png">

 → Query와 Key의 행렬 곱으로 N제곱
 → max(D, M) : Query와 Key의 차원(D)와  값의 차원(M) 중에서 큰 차원으로 맞추어집니다.

linear attention의 complexity
<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-18.png">


→ C : 커널 함수가 입력 벡터를 고차원 공간으로 변환할 때 사용되는 차원
→ NCM : key와 value에 대한 계산을 미리 수행하므로 복잡도가 N제곱이 아닌 NCM으로 나옵니다.
→ [개인적인 생각] 물론 차원 C가 너무 커서 O(NCM)이 O(N제곱x M)에 가까워질 수 있으므로 너무 고차원으로 선택하면 안 될 거라고 생각합니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-19.png">


실험에서는 특징 맵으로써 수식 (7)을 사용하는데  elu는 지수 선형 유닛(exponential activation unit)을 나타냅니다.
(reference : Activation fucntion(2)-tanh/ReLU/LeakyReLU/ELU/Maxout (kjhov195.github.io))

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-20.png">



아무래도 elu는 x 가 음수일 때도 그래디언트가 0이 아니므로 reul대신 elu를 선택합니다.

결론적으로 특징 맵은 O(NDM) 의 복잡도를 가진 곱셈과 덧셈을 요구하는 어텐션 함수를 생성합니다. 

## 3.3 Causal Masking

Causal masking은 원인과 결과의 시간적인 유지하고자 현재 시점(i)는 과거 시점(j)들에만 영향을 받고 미래 시점(j > i)로부터는 영향을 받지 않기 위해서 범위를 제한하는 것입니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-21.png">

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-22.png">


수식(8)은 수식(3)과 비교했을 때 수식(8)의 sigma의 범위가 j=1부터 j=N까지가 아닌 j=i까지로 제한됩니다. → j가 i보다 큰 경우는 영향을 받지 않습니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-23.png">

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-24.png">

수식(9) 는 수식(5)에서 causal masking을 적용해서 도출한 수식입니다.

수식 (10)의 Si 는 i번째 위치까지의 key와 value에 해당하는 부분을 Si로 정의한 것입니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-25.png">

수식 (11)의 Zi는 i번째 위치까지의 key에 해당하는 부분을 Zi로 정의한 것입니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-26.png">

Si와 Zi를 계산할 때 이전 단계인 Si-1와  Zi-1의 값을 이용해서 빠르게 계산할 수 있으므로 전체 시퀀스 길이에 대해 선형적인 계산 복잡도를 가집니다.

개인적인 생각 : 귀납법처럼 Si와 Zi는 이전 시퀀스에서 계산된 값들을 재사용한다. 따라서 다음 시퀀스에서의 값을 효율적으로 계산할 수 있습니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-27.png">

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-28.png">

이를 통해 계산 복잡도를 선형으로 줄일 수 있습니다.

수식 (9)를 수식 (10)과 수식 (11)를 바탕으로 수식 (12)로 표현 가능합니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-29.png">

### 3.3.1 GRADIENT COMPUTATION

Si 는 모델의 파라미터 업데이트에 필요한 정보인 중간 값(intermediate values)의 역할을 합니다.

예측 값과 실제 값의 차이를 줄이기 위해서 역전파(backpropagation) 과정의 그래디언트 계산이 필요합니다.

수식 12는 naive한 implementation으로써 중간 값 Si를 모두 저장하는데 메모리 사용량은 최대 max(D, M) 만큼 증가하고 시퀀스가 길어지거나 
레이어가 많아서 깊은 모델에서는 메모리 사용량이 크게 증가합니다.

수식 12에서 변수 Qi 에 대한 gradient를 계산하면 수식 (13)이 됩니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-30.png">

수식 12에서 변수 Ki에 대한 gradient를 계산하면 수식 (14)이 됩니다.
<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-31.png">

수식 12에서 변수 Vi에 대한 gradient를 계산하면 수식 (15)이 됩니다.
<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-32.png">

분자 (numerator) : query와 key의 유사도를 통해 가중치를 적용한 value의 합을 정규화한 결과입니다.

분모 (denominator) : key의 누적 합으로 특정 쿼리 Qi와 모든 이점 시점의 Key Kj 간의 유사도를 합산한 결과입니다. (정규화를 위해서 사용)

알고리즘 1 Linear transformers with causal masking

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-33.png">

### 3.3.2 TRAINING AND INFERENCE

- 전체 ground truth 시퀀스가 유효할 때 훈련이 가능합니다. 모델이 시퀀스의 모든 시간 단계에서 True인 값을 알고 있다는 의미 →  훈련 중에 모든 입력 시퀀스를 한 번에 사용할 수 있어서 병렬 처리 가능합니다.
- layerwise parallelism : 수식 (1)에 언급된 fl 함수와 attention 계산이 병렬로 수행할 수가 있습니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled.png">

- RNN 보다 효율적 : RNN이 time step 마다 순차적인 계산이 필요하지만 transformer인 경우에는 전체 시퀀스를 한 번에 처리가 가능합니다. → 병렬 처리가 가능하고 훈련 속도가 빨라집니다.
- 단점 : 추론 (inference) 단계에서는 시점 (timestep) i 의 출력이 i+1 시점의 입력이 되므로 이 때에는 순차적으로 진행되어야 합니다. → 병렬화(parallelize)가 어렵습니다.
- 기존 transfomer의 비용 이슈 : 시점 별 transformer의 비용은 sequence 길이의 제곱에 비례합니다.
- linear transformer 의 비용 : 훈련에서는 병렬화가 가능하고 추론 단계에서는 선형적인 연산으로 비용과 메모리가 일정하게 유지됩니다.

## 3.4 Transformers are RNNs

- 원래 transformer 모델은 RNN과 근본적으로 다른 접근 방식으로 여겨집니다.
- 그러나 3.3 섹션에서 설명된 causal masking 공식화와 이전 section들에서 언급한 논의를 통해, causal masking을 사용하는 transformer layer는 입력을 받아 내부 상태를 수정하고 출력을 예측하는 방식으로 RNN과 유사하게 동작할 수 있음을 보여주고 있습니다..
- 이는 transformer layer가 RNN과 동일한 방식으로 동작할 수 있음을 의미합니다.

- 수식 (1)의 transformer layer를 RNN으로 공식화했습니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled.png">

초기 상태 : 수식 (16)은 attention 메모리의 초기, 수식 (17)은 정규화 메모리입니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-34.png">

시점 (timestep) i 에서의  attention 메모리 si와 정규화 메모리 zi를 정의합니다. → 수식 (18)과 수식 (19)

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-35.png">
<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-36.png">

최종출력 yi는 수식 (20)으로 정의합니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-37.png">

### Experiments

4.1 : 계산 비용, 메모리 소비, 합성 데이터에서의 수렴 등을 평가합니다.
4.2 , 4.3 :  이미지 생성과 자동 음성 인식을 통해 실세계 응용에서의 효과를 평가합니다.

- linear transformer와 비교할 수 있는 baseline으로써 2개입니다.
    - full transformer with softmax attention
    - Reformer
- 각각의 실험 결과(그래프) 에서 서로를 구분할 수 있도록 bold 체로 이름을 명명합니다.
    - **softmax** : full transformer with softmax attention
    - **linear** :  our proposed linear transformer
    - **lsh-X** : Reformer (lsh-1, lsh-4, lsh-8 / X는 hashing rounds로써 hashing을 몇 번 수행하는지 의미)

## 4.1 Synthetic Tasks

### 4.1.1 Convergence Analysis

linear transformer의 수렴 특성이 어떠한지 조사합니다.

- Artificial Copy Task : 모델이 시퀀스 패턴을 학습하는데 실제 데이터를 사용하는 대신에 학습과 성능 평가를 위해서 고안한 합성 데이터입니다.
 → 아래 예시에서는 심볼 범위를 1부터 127까지로 설정하고 하나의 심볼을 선택해서 길이를 3으로 맞춤, 중간에 위치하는 0은 구분 기호(separator symbol)로 사용함, 두 번 복사된 시퀀스가 됩니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-38.png"> 


(Reference : [Reformer : The Efficient Transformer] https://arxiv.org/pdf/2001.04451 )

본 논문에서는 시퀀스 최대 길이를 128로 설정, 선택할 수 있는 심볼은 10개로 제한하고 구분 기호는 별도로 존재합니다.

- Setting
    - 4 layer transformer with 8 attention heads
    - batch size : 64 / RAdam optimzer
    - learning rate : 1/1000 → 1/10000 (3000 업데이트 이후)
- Result : Figure2
    - Due to the lack of noise introduced by hashing : 해싱(LSH : Locality Sensitive Hashing) 이 데이터를 분산시키지만 해시함수로 생성된 해시 값들이 원래 데이터와 다른 노이즈를 발생하는데 linear(ours)는 hashing을 사용하지 않아서 lsh-4보다 작은 loss를 갖는다는 의미입니다.
    - [개인적인 생각] linear (ours)의 높은 초기 loss : 커널 함수를 사용하여 고차원 공간으로 매핑된 값들을 사용함, 초기 학습 단계에서는 이러한 매핑이 아직 최적화되지 않았기 때문에 모델이 높은 loss를 보이는 것으로 추측합니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-39.png"> 


### 4.1.2 Memory and Computational Requirements

시퀀스 길이에 따른 각각의 transformer에서 소요되는 시간 및 GPU memory 요구사항을 비교합니다.

- 가변적인 시퀀스 길이 : 시퀀스 길이를 다양하게 조정해서 GPU memory의 정점(peak)을 발생하는 시퀀스가 무엇인지 확인합니다. (2의 9승부터 2의 16승까지)

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-40.png"> 

- 배치크기를 시퀀스 길이에 반비례(inversely)하게 설정합니다.
→ GPU 메모리가 한정되어 있어서 시퀀스 길이가 길어지면 시퀀스 처리에 많은 메모리 소요되며
   메모리 부족을 방지하고자 배치 크기를 줄입니다.
- GPU : NVidia GTX 1080 Ti with 11GB of memory
- Figure 1
    - softmax의 경우 GPU가 cover할 수 있는 sequence length의 최대 값이 4096 (2의 12승)임을 확인합니다, 아무래도 스케일이 주어진 sequence length에 대해 2차적으로(quadratically) 상승하므로 GPU 메모리가 cover할 수 있는 length가 Reformer와 linear(ours) 대비 낮습니다.
    - Reformer(lsh-4, lsh-8)의 경우에는 16,384 (2의 14승) 임을 확인, 그리고 linear(ours)와 lsh-1는 2의 16 임을 확인 → 이는 Reformer와 linear(ours)가 sequence length에 대해 선형적으로 스케일링 해서입니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-41.png">  

추가적으로 Reformer의 complexity는 O(NlogN)으로 알려져 있는데 막상 주어진 sequence 길이의 범위 내에서 실험해보니 logN의 값이 크지 않아서 연산 시간에 큰 영향을 주지 않는다고 판단합니다. (N이 1024라면 log N[밑이 2인 log]은 10 정도가 됨, N이 1024지만 logN은 10으로 작아서 영향을 미치지 않는 것으로 판단)
(원문 : Note that although the asymptotic complexity for Reformer is O(NlogN), logN is small enough and does not affect the computation time.)

## 4.2 Image Generation

- conditional autoregressive generation
    - 특정 조건 또는 맥락을 기반으로 데이터를 생성합니다. (문장의 앞부분을 조건으로 뒷부분 문장 생성 또는 이미지 일부를 조건으로 이미지 다른 영역 생성 등)
- unconditional autoregressive generation
    - 특정 조건 없이 초기 입력 값만을 기반으로 시퀀스를 계속 생성합니다. (예를 들어 랜덤하게 시작한 텍스트 시퀀스를 계속 이어서 생성)
- 결과 : softmax 대비 1000배 빠른 이미지 생성, 첫 번째 픽셀부터 마지막 픽셀까지 일정한 메모리를 사용합니다.

### 4.2.1 MNIST

- 8 attention layers , 하나의 레이어 별로 8 attention heads로 설정합니다.
- 10의 -4승의 학습률로 RAdam optimizer 설정하고 총 250번의 epoch으로 모델을 훈련합니다.
- Reformer의 경우, 관련 논문 저자의 제안에 따라 64개의 bucket 및 32개의 chunk를 사용합니다.
(783개의 입력 시퀀스에 대해서 27개의 chunk로, 1개의 chunk 당 29개의 elements로 설정)
- perplexity : 시퀀스 에측에서 모델의 성능을 평가하는데 사용하는 척도, 값이 낮을수록 모델의 예측이 더 정확합니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-42.png">  

- Table 1의 결과를 보면 linear (ours)가 softmax에 비해 300배 가량(317x) 빠릅니다.
- Bits/dim : bits per dimension → 모델이 데이터를 압축하는 측정 지표, 모델이 주어진 데이터의 각 차원을 인코딩하는데 필요한 평균 bit 수 → 값이 낮을 수록 모델이 더 효율적으로 압축할 수 있습니다.
    - 계산 방식 : 모델이 예측한 확률 분포의 log 손실 계산하고서 이를 데이터의 차원 수로 나눕니다.
    (Reference : https://seewoo5.tistory.com/3)


<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-43.png">  

Figure 3 : Unconditional image의 경우 입력 데이터에 대한 조건이 없어서 생성된 이미지의 일관성 평가가 어렵습니다.

반대로 conditional image의 경우, 
원본 이미지의 일부 지운  이미지인 (a)를 보고서 linear transformer가 생성한 것이 (b), 실제 원본 이미지는 (c)가 됨. 저자는 숫자 6 , 2, 9에 대해서 날카로운 경계와 노이즈 없이 유사하게 생성했다고 하고 있음, 일관성을 유지하면서 300배 빠른 결과를 장점으로 강조합니다.

### 4.2.2 CIFAR-10

- 16 transformers layers , 하나의 레이어 별로 설정은 MNIST와 동일합니다.
- Reformer : 64 buckets, 83chunks of 37 elements
- MNIST에 비해 sequence 길이가 4배 커서 Nvidia P40 GPU는 24GB의 메모리를 갖고 있음에도 불구하고 (1080 Ti는 11GB) 한 번에 처리할 수 있는 배치 크기가 제한됩니다.
- 이에 배치 사이즈를 4로 제한합니다.
- 결과 : (Table 2) perplexity 척도가 실험의 주요 포인트는 아니고 대신에 시퀀스 길이가 증가할수록 linear transformer와 같은 빠른 모델이 메모리와 시간 효율성이 높아집니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-44.png"> 

- Reformer와 Softmax attention에서는 이미지의 각 픽셀을 생성할 때, 메모리와 시간이 픽셀 수에 따라 2차적으로 증가합니다.
- Figure 4 : 모델이 생성하는 이미지가 공간적으로 일관성을 유지함 → 이미지 내의 픽셀들이 서로 일관된 패턴과 구조를 가집니다.
    - 따라서  설득력 있게 완성할 수 있다는 의미와 함께 이미지의 분류 인식에 방해가 되지 않습니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-45.png"> 

MNIST 경우와 마찬가지로 unconditional과 conditional의 차이점은 있으며 원본 이미지의 일부를 제거한 상태에서(a) linear transformer가 생성한 이미지들(b)은 강아지, 새, 트럭 으로써 원본이미지인 (c)와 유사하다고 합니다.(다만 (b)의 마지막 시퀀스가 우측 하단이나 좌측 하단이 완전하게 생성되지 않은 상태를 의미하는 것인지 아니면 마지막 바로 이전의 시퀀스인지 모르겠습니다.)

## 4.3 Automatic Speech Recognition

- Non-autoregressive task : 각 단계가 이전 단계에 의존하는 autoregressive와 다르게 Non-autoregressive task는 각 frame의 예측이 독립적으로 이루어져서 이전 frame에 대한 의존이 없습니다.
- Connectionist Temporal Classification Loss (CTC Loss) : 입력과 출력 사이의 정렬이 알려지지 않은 시퀀스 작업에서 사용되는 손실 함수 → 모델이 다양한 길이의 시퀀스를 예측하고 입력 시퀀스와 정렬할 수 있게 합니다.
- Phonemes : 특정 언어에서 단어를 구별하는 소리의 단위입니다. (bat, cat 은 각각의 phonemes가 /b/와 /k/가 됨)
- 40차원 멜 스케일 필터뱅크를 사용한 80시간의 WSJ 데이터셋(Paul & Baker, 1992)을 사용합니다.
    - WSJ 데이터셋: WSJ는 Wall Street Journal의 약자로, Wall Street Journal의 낭독 데이터를 모은 음성 데이터셋
    - Mel-scale Filterbanks : 사람의 귀의 반응을 더 가깝게 시뮬레이션하는 일련의 대역통과 필터로, 선형 주파수 대역보다 기계 학습에 더 적합한 형태로 신호의 주파수 도메인 표현을 변환합니다.
- transformer model : 9개의 layer, 6개의 head
- RAdam 옵티마이저 (학습률 : 10의 -4승 ) → 검증 오류가 감소하지 않을 때 학습률을 2로 나눕니다.
- PER(Phoneme Error Rate) : 음성 인식 시스템의 성능을 평가하는 지표로, 잘못 예측된 Phonemes 의 비율(값이 낮을 수록 성능 좋음)
- Table 3 : 비교 군은 3개의 320 크기 은닉층을 가진 양방향(순방향/역방향) LSTM, Bi-LSTM인 (Hochreiter & Schmidhuber, 1997)

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-19-Transformers_Are_RNNs/Untitled-46.png"> 

PER은 softmax가 가장 작은 결과를 도출했지만 Linear(ours)가 Bi-LSTM보다 낮은 PER을 도출하고 epoch당 소요 시간이 제일 낮은 결과를 Table 3에서 언급합니다.


### Discussion & Conclusion

- linear transformer
    - softmax 기반의 원래 transformer 대비 메모리와 연산 속도를 줄였다는 것에 의의를 둡니다.
    - 행렬 곱의 결합 법칙을 활용해서 self-attention 계산이 시퀀스 길이에 대해 선형적으로 확장되도록 했음 → 특히 커널 함수를 사용하여 고차원 공간으로 매핑해서 연산 복잡도를 줄입니다.
    - 모델이 causal masking을 사용하면서도 여전히 선형적인 점근적 복잡도를 유지합니다.
    - Transformer 모델을 RNN으로 표현할 수 있음을 보여주었으며, 이를 통해 autoregressive 작업에서 추론 속도를 획기적으로 높일 수 있습니다.
- 추가 사항
    - RNN과 transformer 모두에서 정보 저장 및 검색에 관한 연구의 새로운 방향을 열었습니다.
    - linear attention의 feature map 선택과 관련된 연구도 앞으로 탐구가 필요합니다.
        - 예를 들어 random Fourier feature를 사용해서 RBF kernel 를 근사하면 softmax attention로 사전 학습된 모델을 사용할 수 있을 것으로 예상합니다.