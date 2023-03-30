# Language-Image Quality Evaluator (LIQE)

The official repo of [**Blind Image Quality Assessment via Vision-Language Correspondence: A Multitask Learning Perspective**](https://arxiv.org/pdf/2303.14968.pdf) (CVPR2023)

# Abstract

We aim at advancing blind image quality assessment (BIQA), which predicts the human perception of image quality without any reference information. We develop a general and automated multitask learning scheme for BIQA to exploit  auxiliary knowledge from other tasks, in a way that the model parameter sharing and the loss weighting are determined automatically. Specifically, we first describe all candidate label combinations (from multiple tasks) using a textual template, and compute the joint probability from the cosine similarities of the visual-textual embeddings. Predictions of each task can be inferred from the joint distribution, and optimized by carefully designed loss functions. Through comprehensive experiments on learning three tasks - BIQA, scene classification, and distortion type identification, we verify that the proposed BIQA method 1) benefits from the  scene classification and distortion type identification tasks and outperforms the state-of-the-art on multiple IQA datasets, 2) is  more robust in the group maximum differentiation competition, and 3) realigns the quality annotations from different IQA datasets more effectively.

![image](https://github.com/zwx8981/LIQE/blob/main/clip_biqa.png)

Source code will be released soon.
