---
layout: post
title:  "[XAI] Attention is not Explanation"
date:   2024-08-30 16:00:00 +0900
---

### Abstract

- attention score → often presented as communicating the relative importance of inputs
- However, it is not always.
- In this work, perform extensive experiments across a variety of NLP tasks that aim to assess the degree to which attention weights provide meaningful “explanations" for predictions.

### Introduction and motivation

- attention mechanism: for a given output one can inspect the inputs to which the model assigned large attention weights
- It is common in the literature to argue the interpretability of attention
    - Implicit in this is the assumption that the inputs (e.g., words) accorded high attention weights are responsible for model outputs.
    - But this assumption is not formally evaluated
- If attention provides a faithful explanation…
    - Attention weights should correlate with feature importance measures such as gradient-based input attribution
    - Alternative (or counterfactual) attention weight conf igurations ought to yield corresponding changes in prediction
        
        ![Untitled](Attention%20is%20not%20Explanation%20b789b2e62301413693ffec46d679e616/Untitled.png)
        
- research questions
    1. To what extent do induced attention weights correlate with measures of feature importance– specifically, those resulting from gradients and leave-one-out (LOO) methods?
        
        *A: only weakly and inconsistently*
        
    2. Would alternative attention weights (and hence distinct heatmaps/“explanations”) necessarily yield different predictions?
        
        *A: No (~vulnerable to adversarial attack)*
        
- experimental setting in this paper
    - BiLSTM encoder
    - task: text classification, QA, NLI

### Task explanations

- binary text classification
    - input: text / output: binary label
    - e.g. negative / positive
- QA
    - input: (paragraph)+question / output: answer
- NLI (natural language inference)
    - task of determining whether a "hypothesis" is true (entailment), false (contradiction), or undetermined (neutral) given a "premise"
    - input: premise / output: neutral / contradiction / entailment

### Experiments

- first question
    
    ![Untitled](Attention%20is%20not%20Explanation%20b789b2e62301413693ffec46d679e616/Untitled%201.png)
    
    - emphleave-one-out (LOO) test
    - for feature attribution method, keeping the attention distribution fixed, just turn-off the neuron to disconnect the computation graph so that the gradient does not flow through this layer
    - kendall’ tau: [-1, 1] ranged rank correlation; 1 If the agreement between the two rankings is perfect, while -1 If the disagreement between the two rankings is perfect(perfect reverse order)
    
    ![Untitled](Attention%20is%20not%20Explanation%20b789b2e62301413693ffec46d679e616/Untitled%202.png)
    
    - consideration of error in kendall tau
        - irrelevant features may add noise to the correlation measure
            
            ⇒ One observation that may mitigate this concern is that we might expect such noise to depress the LOO and gradient correlations to the same extent as they do the correlation between attention and feature importance scores; but as per Figure 4, they do not.
            
- second question
    - randomly permuting observed attention weights and recording associated changes in model outputs
        
        ![Untitled](Attention%20is%20not%20Explanation%20b789b2e62301413693ffec46d679e616/Untitled%203.png)
        
        ![Untitled](Attention%20is%20not%20Explanation%20b789b2e62301413693ffec46d679e616/Untitled%204.png)
        
        - scrambling the attention makes little difference to the prediction.
    - adversarial attention weights: maximally differ from the observed attention weights and yet yield an effectively equivalent prediction
        
        ![Untitled](Attention%20is%20not%20Explanation%20b789b2e62301413693ffec46d679e616/Untitled%205.png)
        
        ![Untitled](Attention%20is%20not%20Explanation%20b789b2e62301413693ffec46d679e616/Untitled%206.png)
        
        ![Untitled](Attention%20is%20not%20Explanation%20b789b2e62301413693ffec46d679e616/Untitled%207.png)
        
- to measure the output change, total variation distance is used to measure the difference between output distributions
- to measure the attention change, JSD is used to measure the difference between attention distributions

### Limitation

- We have reported the (generally weak) correlation between learned attention weights and various alternative measures of feature importance, e.g., gradients. We do not intend to imply that such alternative measures are necessarily ideal or that they should be considered ‘ground truth’.
- irrelevant features may be contributing noise to the Kendall τ measure, thus depressing this metric artificially.
- only consider handful model
- counterfactual attention experiments demonstrate the existence of alternative heatmaps that yield equivalent predictions; thus one cannot conclude that the model made a particular prediction because it attended over inputs in a specific way.
- limited our evaluation to tasks with unstructured output spaces, i.e., we have not considered seq2seq tasks