---
layout: post
title: "[XAI] 논문리뷰 - Attention is not Explanation"
date: 2024-05-03 00:00:00 +0900
author: Subeen Lee
categories: ["Attention Mechanism"]
---
**Is attention explanation?**

이 포스트에서는 아래 두 논문을 다룹니다.

[1] Jain, S., & Wallace, B. C. (2019). [**Attention is not explanation**](https://aclanthology.org/N19-1357/). In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 3543–3556, Minneapolis, Minnesota. Association for Computational Linguistics.

[2] Wiegreffe, S., & Pinter, Y. (2019). [**Attention is not not explanation**](https://aclanthology.org/D19-1002/). In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 11–20, Hong Kong, China. Association for Computational Linguistics.

### Summary

Transformer의 핵심 구성 요소인 attention mechanism은 입력과 입력 또는 입력과 이전 출력과의 관계를 attention weight 또는 score 형태로 계산하고, 이를 입력에 곱한 후 합하여 레이어의 출력으로 사용합니다. 이 때 attention score는 출력에 대한 각 입력의 상대적인 중요도처럼 해석되곤 합니다.

그러나 해당 논문-**Attention is not explanation**에서는 그것이 항상 올바른 해석이 아님을 다양한 NLP task에 대해 실험함으로써 보이고자 합니다.

이후의 논문-**Attention is not not explanation**은 이에 반박하는 논문으로, attention score가 설명을 제공하는가에 대한 것은 “설명”이 무엇인지에 대한 정의에 의존한다고 주장하며, Attention is not explanation 논문의 실험의 부족한 점을 지적하고 새로운 실험 4가지를 제시합니다. 그리고 이 실험들을 통해 attention이 각 상황에서 설명을 제공할 수 있는지를 판단하고자 합니다.

### Attention is <span style="color : red">NOT</span> Explanation

**Claim 1) Attention이 설명이라면 다른 feature importance 형태의 설명과도 상관되어야 한다.**

- Specific research question - **Exp 1**) attention weights의 gradient-based feature importance, differences in model output induced by leave-one-out test setting와의 rank correlation

**Claim 2) Alternative or Counterfactual attention weight은 output (prediction)에 변화를 일으켜야 한다.**

- Specific research question - **Exp 2**) attention weights을 random permutation 또는 adversarial attack하였을 때 prediction이 어떻게 변하는가?

**NLP Tasks**

- binary text classification
    - input: text / output: binary label (e.g. negative / positive)
- QA (Question Answering)
    - input: (paragraph)+question / output: answer
- NLI (natural language inference)
    - task of determining whether a "hypothesis" is true (entailment), false (contradiction), or undetermined (neutral) given a "premise"
    - input: premise / output: neutral / contradiction / entailment
    

**Exp 1)** attention weights의 gradient-based feature importance와의 rank correlation in the leave-one-out test setting

- To measure the rank correlation, [Kendall’s Tau](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient) was used. Kendall’s Tau는 [-1, 1]의 범위를 가지며, 1에 가까울수록 양의 상관관계, -1에 가까울수록 음의 상관관계를 가집니다. 절대값이 작을수록 낮은 rank correlation을 가집니다.
- feature importance computations
    
    <img src="../assets/imgs/2024-05-03-Attention_Explanations/image.png">
        
    - $\hat\alpha$: attention weight
    - $g:=\{g_t\}_{t=1}^T$: gradient-based feature importance
    - $\Delta\hat y:=\{\Delta \hat y_t\}_{t=1}^T$: differences in model output by leave-one-out (LOO)
- 따라서 우리는 각 입력마다 시간축(t)을 따라 rank correlation을 계산합니다. 데이터셋 크기만큼의 correlation coefficient를 얻을 수 있으며, 이를 히스토그램으로 나타내면 아래와 같습니다.
    - 그림에서는 attention weight과 gradient-based feature importance 사이의 다양한 dataset, task에서의 상관관계를 나타내며, LOO와의 상관관계를 나타낸 figure는 비슷한 개형을 보인다고 합니다.
    - binary text classification task에서 주황색은 positive로 예측된 샘플들, 보라색은 negative로 예측된 샘플들입니다. NLI task에서 주황색은 entailment, 보라색은 contradiction, 초록색은 neutral로 예측된 샘플들입니다.
    
    ![image.png](../imgs/2024-05-03-Attention_Explanations%20fabee195ccac43adb6e165e0937dc162/image%201.png)
    
- 측정된 correlation은 약한 양의 상관관계를 가집니다. 그에 비해 attention weight과 LOO 결과는 일관되게 더 강한 상관관계를 가진다고 합니다.

**Exp 2)** attention weights을 random permutation 또는 adversarial attack하였을 때 prediction이 어떻게 변하는가?

Attention weight이 출력에 대한 상대적인 입력 중요도를 나타낸다면, 해당 중요도를 바꾸었을 때 출력이 크게 변하거나 잘못될 것입니다. 이 research question에서는 두 가지의 실험 세팅을 통해 이를 테스트하고자 하며, 결론적으로 attention이 이 세팅에서는 설명이 될 수 없음을 보입니다.

- randomly permuting attention weights
    
    ![image.png](/imgs/2024-05-03-Attention_Explanations%20fabee195ccac43adb6e165e0937dc162/image%202.png)
    
    - 이 세팅에서는 다른 모든 것들은 고정한 채로 attention weights만을 랜덤하게 섞어 모델의 출력이 변하는지를 관찰합니다.
    
    ![image.png](/imgs/2024-05-03-Attention_Explanations%20fabee195ccac43adb6e165e0937dc162/image%203.png)
    
    - 결과는 큰 차이를 보이지 않았습니다.
- adversarial attention weights
    - 일반적으로 adversarial attack은 원하는 objective function을 최대로 하면서 설정한 constraint을 만족하는 입력을 찾는 과정을 말합니다. 이 세팅에서는 objective function은 attention weight 사이의 divergence 값, constraint은 원래 출력과 변한 출력의 total variance distance (TVD) 사이의 norm 값으로 설정하여 출력을 최대한 바꾸지 않는 attention weight 값을 찾는 것을 목적으로 합니다. 다시 말하면, attention weight에 큰 변화를 주었지만 출력은 변하지 않는 상황을 찾는 것입니다.
    - 참고로, any two categorical distributions의 JS Divergence 값은 0.69의 상한값을 갖습니다.
    
    ![image.png](/imgs/2024-05-03-Attention_Explanations%20fabee195ccac43adb6e165e0937dc162/image%204.png)
    
    ![image.png](/imgs/2024-05-03-Attention_Explanations%20fabee195ccac43adb6e165e0937dc162/image%205.png)
    
    ![image.png](/imgs/2024-05-03-Attention_Explanations%20fabee195ccac43adb6e165e0937dc162/image%206.png)
    
    - 결과는 위와 같습니다. 많은 max JSD 값이 상한선인 0.69에 쏠려있는 것을 확인할 수 있습니다. 이는 모델의 출력을 유지하면서 아예 다른 attention weight 값을 찾는 것이 쉽게 가능함을 의미합니다.
    

### Attention <span style="color : blue">CAN BE</span> Explanation

앞선 논문에서는 attention weight distribution이 다른 feature importance와 비슷한 ranking 또는 일부가 배제되더라도 일관된 출력을 보이는 것을 기준으로 하여 attention weight의 설명성이 없음에 대한 주장을 합니다. 그러나 이 논문에서는 이에 대한 반박으로 다음과 같은 주장을 제시합니다.

**Claim 1) Attention distribution is not a primitive.**

- 모델의 다른 부분들을 고정한 채로 attention weight을 조작하는 행위는 모델의 function을 망가뜨립니다.
- attention weight은 arbitrary하게 할당되는 부분이 아니라 다른 model components에 의해 계산되는 값입니다.

**Claim 2) Existence does not entail exclusivity**

- 같은 출력을 내는 것에는 다양한 attention weights의 조합이 존재할 수 있으며, 마지막 레이어에서 그것을 종합, 선택하여 출력을 내기 때문에 더욱 그렇습니다.

본 논문에서 제시하는 네가지의 테스트는 attention이 설명으로 쓰일 수 있는지 판단하기 위한 방법으로써 제시됩니다.

1. Uniform as the adversary
    - 모델의 다른 부분은 고정한 채로 attention weight distribution을 uniform distribution으로 치환하여 샘플링하여 사용합니다.
2. Variance within a model
    - different random seeds를 사용하여 training한 모델들로부터 attention weight distribution을 비교합니다.
3. Diagnosing attention distributions by guiding simpler models
    - 학습된 모델의 attention weights을 pre-trained weight으로 사용하여 simple MLP model을 학습시킵니다.
4. Training an adversary model
    - 출력은 유지하면서 (차이를 최소화하면서) attention weights의 divergence는 크게 하는 adversarial neural network를 새로 학습합니다.

이러한 테스트를 통해 각각의 NLP task와 dataset의 결과로부터 attention weight의 설명성을 확인할 수 있었으며 설명이 가능한 경우도 있고 그렇지 않은 경우도 있음을 확인하였습니다.

### [Study] Questions & Discussion

-