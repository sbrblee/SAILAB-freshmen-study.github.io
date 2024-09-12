---
layout: post
title: "[논문리뷰] A Mathematical Framework for Transformer Circuits"
date: 2024-08-09 00:00:00 +0900
author: Myeongjin Lee
categories: ["Mechanistic Interpretability", "Antropic"]
use_math: true
---
**transformer model의 동작 원리를 이해할 수 있게 해주는 mathematical framework**

이 포스트에서는 아래 논문을 다룹니다.

Elhage, et al., (2021). [**A mathematical Framework for Transformer Circuits**](https://transformer-circuits.pub/2021/framework/index.html). Transformer Circuits Thread.

# Summary

- They conceptualized the operation of **transformers with two layers or less which have only attention blocks** in a new but **mathematically equivalent way**, which allows us to gain **significant understanding of how they operate internally**.
- **Zero layer** transformers model **bigram statistics**.
- **One layer** attention-only transformers are an **ensemble of bigram and skip-trigram** (sequences of the form “A…B C” models.
    - Implementation of a very simple in-context learning
- **Two layer** attention-only transformers can implement much more complex algorithms using **compositions of attention heads**.
    - Attention head composition is used for creating **induction heads**, a very general in-context learning algorithm.

# Introduction & Background

## Transformer Overview

### High-Level Architecture

This paper focuses on autoregressive, **decoder-only transformer** language models.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-09-Framework_Transformer_Circuits\Untitled.png"> 

### Virtual Weights

Virtual weights are the product of the output weights of one layer with the input weights of another. This describes the extent to which **a later layer read in the information written by a previous layer**. 

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-09-Framework_Transformer_Circuits\Untitled 1.png">

### Subspaces and residual stream bandwidth

The residual stream is a high-dimensional vector space. Layers can **send different information to different layers** by storing it in different subspaces. 

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-09-Framework_Transformer_Circuits\Untitled 2.png">

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-09-Framework_Transformer_Circuits\Untitled 3.png">

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-09-Framework_Transformer_Circuits\Untitled 4.png">

### Observations about attention heads

- Attention heads **move information** from the residual stream of one token to another.
- An attention head is really applying two linear operations, $A$ and $W_oW_v$, which operate on different dimensions and act independently.
    - $A$ governs **which token’s information is moved** from and to.
    - $W_{o}W_{v}$ governs **which information is read** from the source token and **how it is written** to destination token.

# Method: Proposed mathematical framework

## Zero-Layer Transformers (Not considering the context)

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-09-Framework_Transformer_Circuits\Untitled 5.png">

- The model takes a token, **embeds** it, **unembeds** it to **produce logits predicting the next token**.
- It **cannot move information from other tokens** and simply **predicting the next token from the present token** (use **bigram statistics**).
- This helps represent bigram statistics which **aren’t described by more general grammatical rules**, such as the fact that **“Barack” is often followed by “Obama”**.

## One-Layer Attention-Only Transformers (Copying)

One-layer attention-only transformers can be understood as an ensemble of a **bigram model** and several **skip-trigram models**. (bigram + skip-trigram)

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-09-Framework_Transformer_Circuits\Untitled 6.png">

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-09-Framework_Transformer_Circuits\Untitled 7.png">

### **Splitting Attention Head terms into Query-Key and Output-Value Circuits**

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-09-Framework_Transformer_Circuits\Untitled 8.png">

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-09-Framework_Transformer_Circuits\Untitled 9.png">

### Interpretation as Skip-Trigrams

### Copying / Primitive In-context Learning

Most attention heads in one layer models dedicate an enormous fraction of their capacity to **copying**. 

- The **OV circuit** gives the **highest probability to the token itself**, and to a lesser extent, similar tokens.
    - If a token is attended to, the OV circuit increases the probability of that token.
- The **QK circuit** then only **attends back to tokens which could plausibly be the next token**.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-09-Framework_Transformer_Circuits\Untitled 10.png">

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-09-Framework_Transformer_Circuits\Untitled 11.png">

Some attention heads appear to partially specialize in handling copying for words that split into two tokens without a space. When these attention heads observe a fragmented token (e.g., R) they attend back to **tokens which might be the complete word with a space**  (e.g.,  Ralph) and then predict the continuation (e.g., alph). 

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-09-Framework_Transformer_Circuits\Untitled 12.png">

## Two-Layer Attention-Only Transformers

**Key difference** between one-layer and two-layer models?

- **Composition** of attention heads
- Two-layer models discover ways to **exploit attention head composition** to **express a much more powerful mechanism** for accomplishing in-context learning.

### Three kinds of composition

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-09-Framework_Transformer_Circuits\Untitled 13.png">

Q- and K- Composition both **affect the attention pattern**, allowing attention heads to **express much more complex patterns**.

V-Composition affects what information an attention head moves when it attend to a given position.

The result is that V-Composed heads really act more like a single unit and can be thought of as creating an additional virtual attention heads. 

### Path Expansion of Logits

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-09-Framework_Transformer_Circuits\Untitled 14.png">

The **direct path term** and **individual attention head** terms are **identical to the one-layer model**. The final **virtual attention head** term corresponds to V-Composition.

### Path Expansion of Attention Score QK Circuit

Q-Composition and K-Composition cause them to have much more expressive second layer attention patterns. 

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-09-Framework_Transformer_Circuits\Untitled 15.png">

# Experiments

### Analyzing a Two-Layer Model

https://transformer-circuits.pub/2021/framework/index.html

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-09-Framework_Transformer_Circuits\Untitled 16.png">

One quick observation: Most attention heads are **not involved in any substantive composition**, which indicates **there is a larger collection of skip trigrams**. 

## Induction Heads

### - Function of induction heads

Induction heads **search over the context** for previous examples of the present token.

- If they don’t find it, they attend to the first token (e.g., a special token placed at the start), and do nothing.
- If they do find it, they look at the **next token** and **copy** it.
    - This allows them to **repeat previous sequences of tokens**, both exactly and approximately.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-09-Framework_Transformer_Circuits\Untitled 17.png">

- It knows **how the token was previously used** and **looks out for similar cases**.
- It’s also **less vulnerable to distributional shift**, since it **doesn’t depend on learned statistics** about whether one token can plausibly follow another.

**Examples of Induction Head** 

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-09-Framework_Transformer_Circuits\Untitled 18.png">

### -How induction heads work

- The key is computed from tokens shifted one token back.
- The query searches for similar keys are shifted, finds the next token.

<img src="https://sbrblee.github.io/SAILAB-freshmen-study.github.io/imgs/2024-08-09-Framework_Transformer_Circuits\Untitled 19.png">

# Discussion & Conclusion

## What about the virtual attention heads?

Virtual attention heads seemed **not to have any significant roles**.

However, there are two things worth noting regarding virtual attention head:

1. Useful predictive power from the token two back can be gained via virtual heads
2. There are a lot of virtual attention heads.
    1. The model may have a lot more space to gain useful predictive power via the virtual attention heads. 

## Can this framework generalize to general transformers?

- Although this paper focuses on understanding a restricted class of transformers, i.e., one-layer and two-layer attention-only transformers, they argue that **these methods can be utilized to understand portions of general transformers**.
    - One of the reasons is that normal transformers contain some circuits which appear to be primarily attentional.
    - In practice, they found instances of interpretable circuits involving only attention heads and the embeddings. (The focus on their next paper)
- However, there are large parts of general transformers without analyzing the MLP layers which make up 2/3rds of the entire parameters.
    - Since a lot of attention heads interact with the MLP layers, more complete understanding will require analysis on the MLP layers.

# [Study] Questions & Discussion

## 내용 이해를 위한 질문

- 지수
    - eigenvector와 eigenvalue를 copying behavior matrix로 사용한 이유.
- 성준
    
    그럼 정리하면 One Layer와 Two Layer 사이의 차이는 Two layer는 현재 토큰에 기반해서 경험한 정보를 in-context에서 copy 해오는 것이고, One layer는 attention이 가장 높은 것을 copy해오는 것으로 정리할 수 있을까요? 
    

## 다같이 생각해보면 좋은 질문

- 찬우
    - Language model이 아니어도 비슷한 head가 생길까?
    - Memory in Attention head vs Memory in MLP