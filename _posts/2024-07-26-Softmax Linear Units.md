---
layout: post
title: "[논문리뷰] Softmax Linear Units"
date: 2024-07-26 00:00:00 +0900
author: Chanwoo Lee
categories: ["Mechanistic Interpretability", "Antropic"]
use_math: true
---
**Activation Function을 바꿔서 Monosemantic한 LLM 만들기**

이 포스트에서는 다음 Article을 다룹니다.

[1] Elhage, et al., (2022). [**Softmax Linear Units**](https://transformer-circuits.pub/2022/solu/index.html). Transformer Circuits Thread.

### Summary

해당 article은 신경망, 특히 트랜스포머의 MLP에 어떤 feature들이 학습되어 있는지 연구한다. feature에 대해 연구하기 위해 neuron activation에 있는 의미를 파악하는 방법으로 접근한다. 하지만 여기에는 뉴런에는 여러가지 feature가 동시에 발견되는 polysemantic한 특징이 있어서 뉴런별 의미를 알기 어렵다는 문제가 있다. 이를 위해 신경망의 개별 뉴런에 feature의 의미가 할당되는 원리를 파악하고, 이 원리가 극대화되도록 구조를 변형해서 모델을 좀 더 interpretable하게 바꾸려고 시도한다. 해결책으로 activation function을 **Softmax Linear Unit**으로 바꿔서 학습하면 신경망이 개별 뉴런에 monosemantic하게 feature를 할당하도록 학습된다는 결과를 보여준다. 


# Introduction

- Transformer에 대한 mechanistic interpretability를 높이는 것이 목표이다.
- 목적
    - (다양한 사용 예시가 있을 수 있겠지만) 현재의 safety 문제를 해결하기 위한 하나의 방안이 될 수 있기 때문에 중요한 연구이다.
- 문제
    - 신경망에 대한 mechanistic interpretability가 어려운 이유는 polysemantic한 뉴런 때문이다.
    - Polysemantic Neuron이란 관련이 없는 다양한 feature들에 동시에 반응하는 뉴런을 말한다.
- Polysemantic의 원인
    - Superposition hypothesis
        - 뉴런의 갯수는 정해져 있는데, 표현하고자 하는 feature는 그 숫자보다 많기 때문에 더 큰 신경망을 simulation하기 위해 sparse coding 전략의 일환으로 사용한다는 것이다.
        - 이는 neural network가 좋은 성능을 내기 위한 전략으로 사용하는 것일 가능성이 커서, 의도적으로 이를 해체하면 성능에 문제가 생길 여지가 있다.
- 해당 연구의 목적
    - Transformer의 구조를 조금 변화시켜서 ML의 성능은 유지하면서도 interpretability를 높일 수 있는 방법이 있는지 연구한다.
    - Activation Function을 GeLU에서 SoLU로 대체함으로써 interpretability가 높아질 수 있다고 주장한다.

# Key Result

- SoLU는 MLP Neuron 중 해석이 가능한 비율을 더 높이는 역할을 한다.
- SoLU는 다른 feature들을 가리는(억제하는) 방식으로 interpretabiliity를 높이는 것으로 이해될 수 있다.
- 해당 연구는 모델의 구조가 해석 가능성에 영향을 줄 수 있다는 점을 시사한다.
- 어떤 feature들이 있을 수 있는지에 대한 예시를 보여준다.
- Superposition hypothesis에 대한 증거를 보여준다.

# Background

- 3.1 The Importance of Understanding Activations
    - 모델을 이해할 때 파라미터에 초점을 맞추지 않고 activation에 초점을 맞추는 이유는?
    - 컴퓨터 프로그램 비유
        - Parameter보다 activation의 갯수가 더 작다는 장점도 존재한다.
- 3.2 Decomposing activations and the role of bases
    - Neural network의 representation에 담긴 의미를 해석하기 위해서는 각 representation을 여러 feature의 합으로 표현할 수 있어야 한다.
    - 이를 하기 위한 가장 기본적인 접근법은 representation을 나눌 수 있는 적당한 basis를 찾는 것이다.
    - 하지만 어떤 representation은 자유롭게 rotation이 가능한 basis(Non privileged basis)를 가지고 있기 때문에 특수한 의미를 가진 basis로 특정하기 어렵다.
        - Neural net 입장에서는 feature를 어떤 direction에 할당하더라도 이를 조합해서 representation을 만들어 낼 수 있기 때문에 어떤 방향에 어떤 feature가 존재하는지를 특정하기 어렵다.
    - 일부 representation은 privileged basis를 가지고 있기 때문에 각 basis가 특별한 의미를 가진다. ReLU를 가정했을 때 representation은 natural basis(element 1개만 1인 basis)의 조합으로 볼 수 있기 때문에 이러한 방향성이 의미를 가진다.
        - Neural net 입장에서는 ReLU의 output을 representation로 정해서 분류를 해야 한다. 이를 위해서 feature를 특정 방향에 할당할 것이다. 하지만 ReLU의 output은 아무 basis에 feature를 할당하고 조합해서 만들어내기가 어렵다. ReLU는 일부는 0이고, 일부는 positive value를 가지는 output을 만들기 때문이다. 이 경우 neural network가 feature의 조합으로 representation을 만들기 위해서는 표준 기저에 feature를 할당해야 이를 조합해서 의미를 담은 representation을 만들기가 쉽다. 그래서 여기서 표준 기저를 privileged basis라고 부르는 것으로 보이고, 표준 기저의 coefficient = neuron activation이기 때문에 “뉴런에 feature를 할당하는 것이 ReLU output으로 분류를 하는 neural network가 의미를 전달하기 유리하기 때문에 뉴런에 의미가 생긴다”로 이해해볼 수 있다.
    - 물론 natural basis가 feature와 대응된다고 보장하지는 못한다. 이것이 가능하다면 훨씬 해석이 쉬울 것이다.
- 3.3 **Neurons and Polysemanticity**
    - 뉴런의 의미를 실제로 확인해보면 확실한 패턴을 발견하지 못하는 경우가 있다. 하지만 특이하게도 아예 관계없는 의미의 다른 neuron들과 동시에 활성화되는 패턴을 보이기도 한다.
    - 왜 개별 neuron과 feature는 잘 align되지 않는가? 여기에서 제안하는 가설이 superposition hypothesis이다.
- 3.4 Superposition Hypothesis
    - Basis neuron은 orthogonal 좌표인데, neuron 갯수인 2개보다 더 많은 Feature를 표현하고자 하니, 당연히 neuron이 중첩될 수 밖에 없다는 가설이다.
    - 그리고 Almost orthogonal인 경우는 지수적으로 많고, sparse하다는 가정이 있으면 low dimensional projection 후 inverse reconstruction도 가능하다.
    - 다만 NN이 Loss를 줄이기 위해서는 낮은 차원에서 최대한 feature를 만들어서 집어넣어야 한다. 즉 더 큰 sparse model을 나타내기 위해서 낮은 차원에서 interference를 허용하는 결과로 갈 가능성이 높다.
    - Nonlinear activation은 feature가 basis와 더욱 align되게 유도하는 역할을 한다. 하지만 sparse coding의 장점이 너무 크면, superposition을 다시 만들어내기도 한다. 다만 non privileged의 경우와 비교하자면, 이 경우는 아주 강하게 superposition이 발생한다.
- 3.5 What can we do about superposition
    - 더 적은 superposition의 모델을 만든다.
    - superposition을 가정하고, representation을 해석한다.

# **SoLU: Designing for Interpretability**

- 4.1 Properties that May Reduce Polysemanticity
    - 넓게 보면 “representation의 sparsity를 높이는 방안”으로 요약이 가능하다.
    - Activation Sparsity
        - 신경망은 Polysemantic한 neuron을 만들어서 performance를 높이고자 한다. 하지만 이것이 잘 활용되려면 여러 layer에 걸쳐서 activation이 원활하게 잘 되어야 한다. 만약 activation을 바꿔서 여러 레이어에 걸쳐서 다양한 뉴런들을 동시에 활성화시키기가 더 어려워지면, polysemantic을 연결해서 의미를 만들기가 어려워지기 때문에, NN이 뉴런 1개에 하나의 의미를 할당하게 될 것이라고 가정한다. 그래서 activation이 작은 경우는 아예 0으로 밀어버리는 ReLU보다 hard한 activation을 써서 polysemantic neuron이 만들어지는 것을 방해하고자 하는 의도.
    - Lateral Inhibition
        - Lateral Inhibition라는 것은 한 레이어에 있는 여러개의 neuron이 각자의 activation에 대해 서로 경쟁한게 만든다는 의미이다. ReLU처럼 element wise activation을 해버리면 한 개의 feature의 activation이 커지는 것과 다른 feature의 activation이 커지는 것은 서로 관계가 없다. 하지만 softmax와 같은 방식을 사용하면 서로 activation 값에 대해 견제하는 효과(Inhibition)이 생기게 되므로 여러 개의 neuron이 동시에 많이 켜지는 것을 제한할 수 있다.
    - Weight sparsity
        - 만약 input과 output이 모두 privileged basis를 가지고 있다면 weight에 sparsity를 주는 것이 효과가 있을 수 있지만, transformer는 이러한 구조가 아니라서 적용하기는 어려움.
    - Superlinear Activation Functions
        - Suplerliner 성향의 activation을 사용하면 feature가 여러 뉴런에 나누는 경우 activation 값이 훨씬 줄어들게 되는 효과가 있다. 그래서 feature를 하나의 neuron에만 몰아서 학습하게 하도록 유도하는 효과를 만들 수 있다.
    - Change neurons per FLOP / param
        - model을 크게 만들지 않고 neuron을 더 많이 만드는 방법을 사용해볼 수 있다.
- 4.2 SoLU
    - GeLU의 sigmoid를 softmax로 변화
    
    $$
    \text{SoLU} = softmax(x) \cdot x
    $$
    
    - ReLU보다 좀 더 sparse하게 함.
    - basis aligned는 그대로 값을 유지하게 하고, 넓게 퍼진 value는 activation 값을 낮춰 버린다.

- 4.3 LayerNorm
    - SoLU로만 교체하면 Performance 손실이 너무 심해서, additional LayerNorm을 했더니 성능이 나아질 수 있다는 점을 제시.
    - LayerNorm은 사실 superposition이 smuggled through in smaller activation되게 한다. 하지만 이 효과를 고려하더라도 Interpretability는 크게 높아졌다.
    - LayerNorm이 가장 문제가 되는 케이스는 확실히 눈에 띄는 activation이 크지 않은 경우 SoLU의 효과를 완전히 무용화시켜버린다는 것이다.
    
    <img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-26-Softmax Linear Units/Untitled.png">
    
    - LayerNorm을 어차피 쓸거면 exponential activation을 쓰는 것과 동일한 효과를 준다.
    

# **Results on Performance**

5.1 실험

- SoLU + LayerNorm를 쓰면 performance 손실이 거의 없게 만들 수 있다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-26-Softmax Linear Units/Untitled 1.png">

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-26-Softmax Linear Units/Untitled 2.png">

# Results on Interpretability

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-26-Softmax Linear Units/Untitled 3.png">

- 6.1 Setup of Experiments
    - 사실 neuron이 어떤 의미를 가지고 있는지 명백히 밝히기 위해서는 매우 다양한 형태의 증거가 필요하기 때문에 아주 어려운 일이다.
    - 하지만 현실적으로 연구하기 위해서 몇 가지 snippet에서 특정 neuron이 강하게 발현되는 token에 표시를 한다.
    - Caveat
        - 원래 저자들은 activation이 아주 큰 샘플들이 monosemantic이면 monosemantic일거라고 생각했지만, activation이 작은 경우도 고려해야 함을 나중에 알았다고 함.
- 6.2 Quantitative Result
    - SoLU와 post LayerNorm이 포함된 경우 human evaluator가 패턴을 인식한 경우가 더 많았다.
- 6.3 Qualitative Exploration of SoLU Models
    - 1개 레이어
        - MLP의 특정 neuron이 얼마나 fire되는가에 따라 **output logit이 바뀌는 것을 linear하게 관측**할 수 있기 때문에 neuron과 특정 토큰과의 관계를 autoregressive generation 관점에서도 이해할 수 있다.
        - Evaluator의 결과를 cross check 가능하다.
        - 예시
            - Base64에 반응하는 neuron을 data example로도 볼 수 있고, output logit의 변화량에 미치는 영향을 통해서도 알 수 있다. 물론 대소문자의 결합 토큰에 반응하는 것으로 보아 base64라고 이름을 붙여도될지는 의문.
    - 6.3.2 초기 레이어(De Tokenization)
        - 토큰에서 기초적인 패턴을 학습해서, 여러 token들이 결합한 단어들에서 주로 반응한다. 단순한 단어 구조에 대해 이해하는 레이어로 볼 수 있다.
    
    <img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-26-Softmax Linear Units/Untitled 4.png">
    
    - 6.3.3 후기 레이어(Re Tokenization)
        - 합성어를 만들어내기 위해 필요한 token들에 집중하는 것
    
    <img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-26-Softmax Linear Units/Untitled 5.png">
    
    - 6.3.4 중간 레이어
        - 복잡하고 추상적인 내용을 처리하는 부분
        - 예시 : 특정 의미를 담은 clauses, 쌍따옴표의 강조, 컨텍스트 기반으로 해석해야 하는 토큰
    
    <img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-26-Softmax Linear Units/Untitled 6.png">
    
    - 6.3.5 추상적 패턴
        - Neuron Splitting : 모델이 커질수록 1개의 뉴런의 의미가 다른 뉴런으로 나뉜다
        - Neuron Families : 비슷한 역할을 하는 것으로 묶을 수 있는 뉴런이 존재한다.
        - Duality of early and later : 앞 레이어의 역할에 대응되는 후기 레이어가 있다. (decoder only에서 encoder의 역할을 초기 레이어가, decoder의 주요 역할을 후기 레이어가 하는 것으로 이해해볼 수도 있다.)
        - Similarities to CLIP Neurons : 여러 모델에서 비슷한 역할을 하는 경우가 보인다. (ex. 고유명사). 이는 universality와 연관되어 있다.
- 6.4 LayerNorm
    - Pre Layernorm과 Post Layernorm 각각에서 activation level을 달리 하는 여러 데이터셋 예시를 뽑는다. activation이 높아지는 방향으로 dataset example들을 볼 때, 기존에 예상했던 해당 뉴런의 역할과 일치하지 않는 dataset example의 비율을 그려본다. LayerNorm을 거치는 경우, activation이 약할 때 유독 원래 해당 뉴런의 역할과 다른 dataset example에도 크게 반응하는 문제가 있다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-07-26-Softmax Linear Units/Untitled 7.png">

### [Study] Questions & Discussion

# 내용 이해를 위한 질문

- 연지
    - 3.4 The Superposition Hypothesis 의 그림 부분 이해가 잘 안되요
        - That is, a small neural network may be able to approximately "simulate" a sparse larger model, at the cost of some "interference" (figure below). And if it’s the case that the underlying data the model is trying to represent genuinely has a lot of sparse features, then this may be the best thing for the model to do.
    - 4.1 Properties that May Reduce Polysemanticity
        - Superlinear Activation Functions 내용 이해를 잘 못함
    - 4.3 LayerNorm
        - LayerNorm is invariant to scaling the input, since LN(𝑥′) divides by 𝜎(𝑥′) and 𝜎(𝛼𝑥′)=𝛼𝜎(𝑥′) → 이부분 이해가 잘안됨
- 성준
    - privileged ↔  non-privileged 설명한번 더 부탁드립니다
        
        → privileged의 경우에는 ReLU와 같은 activation을 사용해서 특정 뉴런이 특정의미를 갖도록 강제해서 의미를 표현하는 basis
        
        → non-privileged의 경우에는 제한없이 전체 basis를 이용해서 하나의 의미를 표현하는 basis
        

# 다같이 생각해보면 좋은 질문

- 명진
    - SoLU를 사용해서 성능이 낮아진 이유로 저자들은 superposition hypothesis과 함께 polysemanticity 가 줄어듦을 언급.
        - 그러면 뉴런 갯수보다 많은 almost orthogonal한 vector들을 설정하고 그 vector들에 가깝게 학습시키면 (vq-vae처럼?) 성능 저하를 피할 수 있지 않을까? 미리 설정한 vector들이 monosemantic하다고 볼 수 있을 것 같고 뉴런 갯수보다 많은 feature를 학습할 수 있을 것 같은데 SoLU가 더 나은 이유는?
