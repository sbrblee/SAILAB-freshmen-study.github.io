---
layout: post
title: "[논문리뷰] This Looks Like That"
date: 2024-04-05 00:00:00 +0900
author: Seongwook Chung
categories: ["XAI", "Prototype"]
use_math: true
---
**Deep Learning for Interpretable Image Recognition**

이 포스트에서는 아래 논문을 다룹니다.

Chaofan Chen, Oscar Li, Chaofan Tao, Alina Jade Barnett, Jonathan Su, Cynthia Rudin. [**This Looks Like That: Deep Learning for Interpretable Image Recognition**]
(https://arxiv.org/abs/1806.10574). Advances in Neural Information Processing Systems 32 (NeurIPS 2019)

### Summary

- 학습 모델이 특정 결정을 내리는데 중요한 역할을 하는 데이터의 예시 또는 대표 사례를 prototype로 정의하고서 소스 이미지에서 input 이미지의 부분과 유사하다고 예측하는 prototype를 activation map으로 표시할 수 있으며 이를 통해 각각 클래스 별 prototype과 비교 및 예측이 가능하다고 하였습니다.

### Introduction & Background

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-04-05-Prototype_Explanations/untitled-1.png">

- 위 그림(Figure 1)을 보고 점토색의 참새(clay-colored sparrow)라고 어떻게 식별할 수 있는가요?
    - bounding box(yellow box) : 얼굴 (위부터 첫 번째) / 체형 (두 번째) / 깃털 (세 번째) / 다리 (네 번째)
    - looks like : bounding box는 어느 prototype가 유사한지 분석합니다.
        - prototype : 학습 모델이 특정 결정을 내리는데 중요한 역할을 하는 데이터의 예시 또는 대표 사례가 됩니다.
    - activation map : prototype를 갖는 소스 이미지에서 input 이미지의 부분과 유사하다고 예측하는 prototype를 표시합니다(indication)
- 논문 저자들은 그림의 부분(part)이 각각 클래스 별 prototype과 비교를 해서 예측할 수 있다고 생각합니다.
- 그리고 머신 러닝이 이와 같은 방식을 모방해서 학습하고, 판단한 근거를 사람이 이해할 수 있는 방식으로도 설명할 수 있는지 여부를 검증하는 것이 연구의 목표입니다.

### Method
## Case study 1: bird species identification

- 바로 case study를 시작하면서 prototype 기반 딥러닝 모델과 학습 알고리즘을 어떻게 설계했는지 등등 기술합니다.
- 기존 모델과 비교하고자 Convolution Neural Network(CNN) 구조의 모델(VGG, ResNet 등)들을 baseline 모델로 선택합니다.
# ProtoPNet architecture
<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-04-05-Prototype_Explanations/untitled-2.png">
- 기존 CNN와의 차이점은 Convolutional layer(f)와 Fully connected layer(h) 사이에 Prototype layer(g_p)를 두었다는 점입니다.

# Convolutional layer

- 기존 CNN 모델인 VGG-16, VGG-39, ResNet-34, ResNet-152, DenseNet-121, DenseNet-161 중에서 선택합니다.
- Convolutional layer의 output 크기 : H (Height) x W (Weight) x D (Dimension)
    - 입력 이미지의 크기가 224(H) x 224(W) x 3(channel dimension like RGB) 이라면
    - convolutional layer의 output  크기는 
    7(H) x 7(W) x 128(channel), 7 x 7 x 256 또는 7 x 7 x 512 중의 하나입니다.
- Convolutional layer 끝에 1x1 크기의 layer 2개를 추가했습니다.
    - 언급한 이유는 명시하지 않았으나 Prototype layer에 있는 prototype과 비교하기 전에 파라미터 수를 조정하거나 output의 채널 수를 조정하는 등의 사전 작업으로 추정합니다.

# Prototype layer

- 각각의 prototype은 bird image 내에 잠재적으로 존재할 수 있다고 합니다.
    - 프로토타입적인 부분(prototypical part)을 표현할 수 있는, 잠재적인 표현(latent representation)이 될 수 있다는 것을 뜻합니다.
    - Prototype layer에는 input image의 bounding box와 비교할 Prototype이 있습니다.
        - prototype은 총 m개가 있음 (위 Figure 2를 보면 P1부터 Pm까지 있습니다.)
        - prototype의 shape는 H1 x W1 x D이고 H1과 D1의 크기는 convolutional layer의 output의 크기인 H와 W 보다는 작거나 같음, D는 그대로 임(128, 256, 512 중 하나입니다.)
            - 실험에서는 H1 = W1 = 1로 설정했습니다.
    - 각각의 Prototype을 convolutional layer의 output과 비교했을 때 dimension은 같고 H와 W는 작습니다.  **convolutional output중에 activation pattern이 있을 것으로 추측합니다**, 그리고 이는 원본 이미지의 pixel space **(original pixel space)에 prototype의 이미지 패치가 있다는 것**과도 같은 맥락으로 볼 수 있습니다.

- 특정 이미지에 있는 패치가 얼마나 Prototype과 유사한지 평가하는 방법
<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-04-05-Prototype_Explanations/untitled-3.png">
    - 각 Prototype 단위(unit)는 입력 이미지의 특정 부분과 가장 유사한 patch를 찾아내어서 유사도를 계산합니다
    - 유사도 계산은 Prototype과 입력 이미지 패치 간 거리 기반으로써 거리가 작을수록 유사도는 높습니다
    - Prototype 단위와 출력이 커지면 Prototype과 매우 유사한 개념을 가진 패치가 입력 이미지에 존재합니다

# Training algorithm

## Stochastic gradient descent(SGD: 확률적 경사하강법) of layers before last layer

- Convolution layer(f)와 Prototype layer(g_p)는 SGD를 사용해서 손실을 줄이는 것이 목표입니다.
    - 최적화 할 손실 함수는 Cross Entropy(CrsEnt), Clustering (Clst), Seperate(Sep) 3가지로 구성합니다.
<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-04-05-Prototype_Explanations/untitled-4.png">
    - Cross Entropy : 모델이 정확한 분류를 얼마나 잘 하는지 측정합니다.
    - Clustering : 모델이 동일한 클래스 내 데이터를 유사한 Prototype 주위에 잘 군집화 되는지 측정합니다.
    - Seperate : 서로 다른 클래스의 Prototype이 잘 분리되어 있는지 측정합니다.

- Fully connected layer는 prototype과 class(최종 분류) 사이의 연결입니다.
<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-04-05-Prototype_Explanations/untitled-5.png">
    - prototype의 index는 j, class의 index는 k로 설정합니다.
    - wh(k,j) : j번째 prototype과 k번째 class 사이의 연결에 사용되는 weight입니다.
        - Fully connected layer인 h는 weight를 고정합니다.(fixed)
        - positive connection : 특정 prototype가 특정 class K의 patch에 해당됩니다.
            - weight 값을 1로 설정합니다.(고정)
        - negative connection : 특정 prototype가 특정 class K의 patch에 해당되지 않습니다.
            - weight 값을 -0.5로 설정합니다.(고정)

# Projection of prototypes

- Projection : Prototype을 동일한 class의 가장 가까운 훈련 image patch에 근접하도록 하는 작업입니다.
    - 각각의 Prototype을 훈련 image patch와 개념적으로 동일시할 수 있게 합니다.
    - Prototype이 Neural Network에 의해 어떻게 학습하는지 설명입니다.
<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-04-05-Prototype_Explanations/untitled-6.png">
    - 수식의 의미 : 각각의 class k에 대한 Prototype Pj를 업데이트 하기 위해서
        - 해당 class에 속한 모든 패치들 중에서 Pj와 가까운 패치를 찾는 최소화 문제를 풀어나갑니다.
        - Prototype가 훈련 데이터를 잘 대표하도록 해서 Neural Network가 더 의미 있는 특징을 학습 가능하게 합니다.

# Convex optimization of last layer

- ProtoPNet의 마지막 layer인 Fully connected layer h에 대해 미세 조정하는 최적화 과정입니다.
    - 최종 가중치인 wh(k,j)가 희소성(sparsity)를 가진다는 것입니다.
        - 희소성 : 대부분의 가중치 값이 0이거나 0에 가까운 값을 가집니다.
        - 모델이 부정적인 추론 과정에 덜 의존합니다.
            - 부정적인 추론 과정 : 특정 클래스에 속하지 않는 이유로 이미지를 특정 클래스로 분류하는 논리 (”이 새가 클래스 k가 아니라 클래스 k’ 인 이유는 클래스 k의 전형적인 특징을 갖지 않는 patch를 포함하기 때문입니다.”)
            - 반면 희소성은 불필요한 특성들의 가중치가 0에 가까우며 이를 통해 긍정적인 정보(즉, 클래스 k에 대한 증거)에 더 많이 의존하고 있음을 의미합니다.
    - Convex optimization
        - 볼록 최적화 : 함수가 볼록하다는 것은 그 함수의 그래프 상의 임의의 두 점을 연결하는 선분이 그래프 위에 위치하지 않고 항상 아래에 있거나 그래프 위에 걸쳐 있습니다.
            - 볼록 함수에서 최적화를 했어도 특정 지점에서 함수의 값이 감소하지 않는다면(즉, 지역적으로[locally하게] 최소 값에 도달했다면) 해당 지점의 값을 전역 최소 값과 일치 하다고 결론을 내릴 수 있습니다.
            - 비볼록 함수(non-convex function)과 비교해보면, 비볼록 함수의 경우에는 여러 개의 로컬 최소 점이 존재해서 어느 것이 전역 최소 값인지 알기 어렵습니다.
            - 반면 볼록함수에서 이러한 문제가 발생하지 않음, 최적화 알고리즘을 볼록 함수에서 수행 시 결국 로컬 최소점에 도달할 것이고 그것이 최선의 해인 전역 최소값임을 확신할 수 있습니다.
<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-04-05-Prototype_Explanations/untitled-7.png">
        - 목적함수
            - 볼록 최적화 문제를 풀기 위한 목적 함수입니다.
                - Cross Entropy를 최소화 하는 항과 가중치의 절대 값의 합을 최소화 하는 정규화 항입니다.
            - convolution layer와 Prototype layer의 모든 parameter들이 고정되어 있다는 점에서 볼록합니다.
            - 모델은 최적화를 통해 정확성을 유지하면서도 중요 특성만을 사용해서 희소 모델이 됩니다.

# Prototype visualization
<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-04-05-Prototype_Explanations/untitled-8.png">
Prototype projection 과정에서 어떠한 patch가 prototype과 연관되는지 확인하는 것을 visualization하기 위해서 activation map(활성화 맵)을 생성하였습니다.

- 특정 prototype(pj)에 의해 활성화된 x의 이미지 patch를 pj의 시각화로 사용
- 위의 Figure 8은 부록(Supplementary)의 S7에 명시된 Figure로써
    - (1) 단계 : 훈련된 ProtoPNet 모델을 통해 x를 전달하고 Prototype unit인 g_pj에 의해 생성된 활성화 맵을 x 이미지의 크기로 업샘플링(upsampling)하여 수행합니다.
    - (2) 단계 : 활성화 맵을 얻은 후에 upsampling된 활성화 맵에서 활성화 영역을 찾아 pj가 가장 강하게 활성화되는 x의 patch를 찾을 수 있습니다.
    - (3) 단계 : 활성화 맵의 모든 활성화 값 중 최소 95% 백분위 이상의 활성화 값이 있음, 이 값에 해당하는 픽셀들을 포함하는 가장 작은 직사각형 영역을 활성화 영역으로 정의합니다.. 
    
    이를 통해 활성화 영역에 해당하는 x의 이미지 patch를 사용해서 Prototype pj를 시각화할 수 있습니다.

# Reasoning process of our network

ProtoPNet의 추론 과정을 보여주는 paragraph (하단의 Figure 3 사용)
<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-04-05-Prototype_Explanations/untitled-9.png">
Figure 3에서 좌측과 우측으로 나누었을 때 좌측의 내용이 아래입니다.
<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-04-05-Prototype_Explanations/untitled-10.png">

- 빨간배딱따구리(red-bellied woodpecker) 클래스의
    - 첫 번째 Prototype은 테스트 이미지에서 새의 머리 부분에서 가장 강하게 활성화됩니다.
        - Original image의 머리와 빨간배딱따구리의 머리 사이의 높은 유사성(점수 6.499)
    - 두 번째 Prototype은 테스트 이미지에서 날개 부분에서 가장 강하게 활성화됩니다.
        - Original image의 날개와 빨간배딱따구리의 날개 사이의 높은 유사성(점수 4.392)
    - 유사성 점수는 가중치를 곱해져 합산되어 테스트 이미지의 새가 빨간배딱따구리라는 클래스에 속한다는 최종 점수인 32.736을 제공합니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-04-05-Prototype_Explanations/untitled-11.png">

위 Figure가 Figure 3의 우측의 내용입니다.

- 입력 이미지를 갖고 붉은딱따구리(red-cockaded woodpecker) 클래스에 있는 Prototype과의 유사성을 계산해서 최종 점수를 매겨보면 16.886이 나오고 유사하다고 보이는 patch가 Prototype에 있기는 있습니다.
- 하지만 앞서 빨간배딱따구리 클래스의 최종 점수인 32.736과 비교하면 상대적으로 낮은 점수로써 입력 이미지는 붉은딱따구리 클래스로 분류될 가능성이 낮다고 볼 수 있습니다.

# Comparison with baseline models and attention-based interpretable deep models

- Baseline model : Prototype layer 없이 훈련된 모델입니다.
    - VGG16 / VGG19 / Res34 / Res152 / Dense 121 / Dense 161
- attention-based interpretable deep models : Network가 결정 내릴 때 입력 이미지의 어떤 부분에 주목하는지 밝히는 모델입니다.
    - Object attention : class activation map
    - Part-level attention : Part R-CNN부터 RA-CNN까지 있으며 모든 목록은 Table 1에 나열됩니다.
- Table 1에서 Baseline 모델과의 비교 결과는 아래이며 왼쪽 값은 평균(mean accuracy), 오른쪽 값은 표준편차(standard deviation)입니다.
<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-04-05-Prototype_Explanations/untitled-12.png">
- Table 1에서 attention-based interpretable deep model과의 비교 결과는 아래이며
    - bb는 bounding box의 cropped image (이미지에서 해당 객체만을 포함하는 사각형 사용)
    - full은 full image (이미지 전체 영역 사용)
    - anno. 은 annotations
        - 주로 이미지에 있는 객체들의 위치, 크기, 유형 또는 다른 관련 정보들을 나타내는 주석입니다.
        - 모델이 훈련될 때 특정 객체를 인식하고 위치를 추정하기 위해 참조하는 추가적인 정보를 활용합니다.
<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-04-05-Prototype_Explanations/untitled-13.png">
    - ProtoPNet이 attetion based 모델에 비해 장점은 Figure 4를 토대로 설명하고 있습니다.
<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-04-05-Prototype_Explanations/untitled-14.png">

위와 같이 Figure 4-(a)의 Object attention 경우는 이미지 내에서 새의 영역 전체를 새라고 분류한 결정의 “이유”라고 activation map으로 표현하고 있습니다(coarsest level).  Figure 4-(b)의 Part-level attention 경우는 Figure 4-(a) 대비 새의 특징 별(깃털이나 몸의 윤곽선, 머리 모양 또는 색상 등)로 판단한 근거를 activation map으로 표현하고 있습니다.

반면에 ProtoPNet은 Figure 4-(b)와 같은 Part-level attention 뿐 아니라 유사한 prototypical case를 제공한다는 점에서 차별점을 강조합니다. (아래 Figure 4-(c) )

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-04-05-Prototype_Explanations/untitled-15.png">

## Analysis of latent space and prototype pruning

- ProtoPNet이 잠재공간(latent space) 내에서의 구조를 분석합니다.
<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-04-05-Prototype_Explanations/untitled-16.png">

- 위의 Figure는 Figure 5(a)로써 테스트 이미지와 가장 가까운 Prototype을 보여줍니다.
Florida jay의 경우 날개와 가장 유사한 3개의 Prototype이 보여짐(right-top)
Cardinal의 경우 머리 / 얼굴과 가장 유사한 3개의 Prototype이 보여짐(right-top)
- ProtoPNet이 실제로 특정 종의 새를 인식하는데 어떤 Prototype(새의 특정 부위나 특징)을 사용하는지 이해할 수 있음, 즉 네트워크가 학습한 잠재 공간의 구조를 잘 이해하고 Network가 어떻게 각 이미지를 각각의 종에 속하는지 판단에 대한 통찰력을 제공합니다.
- Prototype Pruning : Prototype과 가장 가까운 훈련 patch들이 혼합되어서 정체성을 가지고 배경 patch(background patch)에 해당한다고 설명함. 즉 특정 클래스의 중요 특징이 아닌 배경 정보 또는 불필요한 요소들을 반영할 수 있습니다.
    - 이에 불필요한 Prototype을 제거해야 하며 부록(Supplmentary)의 S8에서 어떻게 정리(Pruning)하는지 기술합니다.


# Case study 2: car model identification

CUB-200-2011 dataset을 갖고 196가지의 자동차 모델을 ProtoPNet가 식별하는 것을 훈련시켰습니다.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-04-05-Prototype_Explanations/untitled-17.png">

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-04-05-Prototype_Explanations/untitled-18.png">

### [Study] Questions & Discussion
