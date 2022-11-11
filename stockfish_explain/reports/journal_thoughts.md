---
marp: true
---


# Chess XAI Journal

---

## Topics of submitted paper

- Piece values
- Piece values per bucket
- Shapley values of concepts
- Weights of linear surrogate model
- Weights of linear surrogate model with custom weights
- Concept probing 
    - Global during training
    - Per bucket

---


## Challenges of concept probing (Deepmind paper)

- What is the right probing architecture?
- How should we interpret complex or subjective concepts?
    - "when regression accuracy is low, is this because the components of the concept **aren’t present** (i.e. the relevant threats aren’t being computed by the model) or because our assessment of **the value of those threats** differs from that of the network?"
- When can we say a concept is represented?
    - "if a concept is accurately predicted half of the time, and completely wrongly the other half, then we may be able to **refine our concepts to better reflect network activations** by understanding what the successes or failures have in common."
- When is a network really representing or using a concept?
    - "When we train a probe we cannot tell if we are getting a confounder or the concept itself."


--- 

## Goals of journal

1) Understand how human-understandable concepts are used by a DNN
2) Find the best ways to probe for human understandable concepts
3) Find ways to make use of concept-probing
4) Use the DNN to better understand the concepts

---

# 1) Understand how human-understandable concepts are used by a DNN

- Global and local surrogate models
- Identify importance of concepts
- Shapley values
- Can we identify how a concept materializes?
    - If a concept is high

---

# 2) Find the best ways to probe for human understandable concepts

- What are the best probing arcitectures?
- How does compression affect concept probing?


---


# 3) Find ways to make use of concept-probing

- Can we use the DNNs to improve the pre-defined concepts?
    - Can we optimize the concept for probing accuracy?
- Can we use concept-probing results to improve the model?
- Can we use surrogate model accuracy as a proxy for classical model performance?
    - Does an increase in r2 indicate a better classical model?


---

# 4) Use the DNN to better understand the concepts

- When a concept is not well represented, can we understand why?
- Can we identify similarities between concepts?
- How does the concept histogram look for the nearest neighbours?
- How can we think about distance between concepts in the model representation?
- How can we perturb the concepts?
- Can we create a saliency map for a concept?
- Concept probe for simpler parts of complex concepts


--- 


## Possible tasks

- Extract all subconcepts from Stockfish
- Compare concept-probing on input with DNN probing. 
- Create a custom bottleneck architecture 
    - Probe at different compression with different probing methods
- Probe deeper layers (using bucket subsets)


---

# Structure of paper

- Implement new chess concepts
- What is the best probing architecture?
    - autoencoder architecture
    - How does compression affect concept probing?
- How can we use concept probing?
    - Can we  