# Awesome-Federated-LLM-Learning
[![Contribution Welcome](https://img.shields.io/badge/Contributions-welcome-blue)]()

⚠️ NOTE: If there is any missing or new relevant literature, please feel free to submit an issue. we will update the Github and Arxiv papers regularly. 😊

<!-- omit in toc -->
## 📒 Table of Contents

- [Awesome-Federated-LLM-Learning](#awesome-federated-llm-learning)
  - [Part 1: LoRA-based Tuning](#part-1-lora-based-tuning)
    - [1.1 Homogeneous LoRA](#11-homogeneous-lora)
    - [1.2 Heterogeneous LoRA](#12-heterogeneous-lora)
    - [1.3 Personalized LoRA](#13-personalized-lora)
  - [Part 2: Prompt-based Tuning](#part-2-prompt-based-tuning)
    - [2.1 General Prompt Tuning](#21-general-prompt-tuning)
    - [2.2 Personalized Prompt Tuning](#22-personalized-prompt-tuning)
    - [2.3 Multi-domain Prompt Tuning](#23-multi-domain-prompt-tuning)
  - [Part 3: Adapter-based Tuning](#part-3-adapter-based-tuning)
    - [3.1 General Adapter Tuning](#31-general-adapter-tuning)
    - [3.2 Personalized Adapter Tuning](#32-personalized-adapter-tuning)
    - [3.3 Multi-domain Adapter Tuning](#33-multi-domain-adapter-tuning)
  - [Part 4: Selective-based Tuning](#part-4-selective-based-tuning)
    - [4.1 Bias Tuning](#41-bias-tuning)
    - [4.2 Partial Tuning](#42-partial-tuning)
  - [Part 5: Other Tuning Methods](#part-5-other-tuning-methods)
    - [5.1 Zero-Order Optimization](#51-zero-order-optimization)
    - [5.2 Split Learning](#52-split-learning)
    - [5.3 Model Compression](#53-model-compression)
    - [5.4 Data Selection](#54-data-selection)


## Part 1: LoRA-based Tuning
### 1.1 Homogeneous LoRA
* Fedra: A random allocation strategy for federated tuning to unleash the power of heterogeneous clients. [[Paper]](https://arxiv.org/abs/2311.11227) 
* Towards building the federatedGPT: Federated instruction tuning.[[Paper]](https://arxiv.org/abs/2305.05644)
* Communication-Efficient and Tensorized Federated Fine-Tuning of Large Language Models. [[Paper]](https://arxiv.org/abs/2410.13097)
* Selective Aggregation for Low-Rank Adaptation in Federated Learning. [[Paper]](https://arxiv.org/abs/2410.01463) 
* Federa: Efficient fine-tuning of language models in federated learning leveraging weight decomposition. [[Paper]](https://arxiv.org/abs/2404.18848)
* LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement. [[Paper]](https://arxiv.org/abs/2411.14961)
* Federated LoRA with Sparse Communication. [[Paper]](https://arxiv.org/abs/2406.05233)
* SA-FedLora: Adaptive Parameter Allocation for Efficient Federated Learning with LoRA Tuning. [[Paper]](https://arxiv.org/abs/2405.09394)
* SLoRA: Federated parameter efficient fine-tuning of language models. [[Paper]](https://arxiv.org/abs/2308.06522)
* FederatedScope-LLM: A Comprehensive Package for Fine-tuning Large Language Models in Federated Learning [[Paper]](https://arxiv.org/abs/2309.00363)
* Robust Federated Finetuning of Foundation Models via Alternating Minimization of LoRA. [[Paper]](https://www.arxiv.org/abs/2409.02346)
* Automated federated pipeline for parameter-efficient fine-tuning of large language models. [[Paper]](https://arxiv.org/abs/2404.06448)
* Low-Parameter Federated Learning with Large Language Models. [[Paper]](https://arxiv.org/abs/2307.13896)
* Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients. [[Paper]]()
* FedRA: A Random Allocation Strategy for Federated Tuning to Unleash the Power of Heterogeneous Clients. [[Paper]](https://arxiv.org/abs/2410.22815)
* Fed-piLot: Optimizing LoRA Assignment for Efficient Federated Foundation Model Fine-Tuning. [[Paper]](https://arxiv.org/abs/2410.10200)

### 1.2 Heterogeneous LoRA
* Heterogeneous lora for federated fine-tuning of on-device foundation models. [[Paper]](https://arxiv.org/abs/2401.06432)
* Flora: Federated fine-tuning large language models with heterogeneous low-rank adaptations. [[Paper]](https://arxiv.org/abs/2409.05976)
* Federated fine-tuning of large language models under heterogeneous tasks and client resources. [[Paper]](https://arxiv.org/abs/2402.11505)
* Federated LLMs Fine-tuned with Adaptive Importance-Aware LoRA. [[Paper]](https://arxiv.org/abs/2411.06581)
* Towards Federated Low-Rank Adaptation of Language Models with Rank Heterogeneity. [[Paper]](https://arxiv.org/abs/2406.17477)
* Fedhm: Efficient federated learning for heterogeneous models via low-rank factorization. [[Paper]](https://arxiv.org/abs/2111.14655)
* RBLA: Rank-Based-LoRA-Aggregation for Fine-Tuning Heterogeneous Models. [[Paper]](https://arxiv.org/abs/2408.08699)
  
### 1.3 Personalized LoRA
* FDLoRA: Personalized Federated Learning of Large Language Model via Dual LoRA Tuning. [[Paper]](https://arxiv.org/abs/2406.07925)
* Fedlora: Model-heterogeneous personalized federated learning with lora tuning. [[Paper]](https://arxiv.org/abs/2310.13283)
* FedLoRA: When Personalized Federated Learning Meets Low-Rank Adaptation. [[Paper]](https://openreview.net/forum?id=bZh06ptG9r)
* Dual-Personalizing Adapter for Federated Foundation Models. [[Paper]](https://arxiv.org/abs/2403.19211)
* Personalized Federated Instruction Tuning via Neural Architecture Search. [[Paper]](https://arxiv.org/abs/2402.16919)
* Communication-Efficient Personalized Federated Learning for Speech-to-Text Tasks. [[Paper]](https://arxiv.org/abs/2401.10070)
* Personalized Federated Fine-Tuning for LLMs via Data-Driven Heterogeneous Model Architectures. [[Paper]](https://arxiv.org/abs/2411.19128)

## Part 2: Prompt-based Tuning
### 2.1 General Prompt Tuning
* Prompt federated learning for weather forecasting: Toward foundation models on meteorological data. [[Paper]](https://arxiv.org/abs/2301.09152)
* Promptfl: Let federated participants cooperatively learn prompts instead of models-federated learning in age of foundation model. [[Paper]](https://arxiv.org/abs/2208.11625)
* Fedbpt: Efficient federated black-box prompt tuning for large language models. [[Paper]](https://arxiv.org/abs/2310.01467)
* Federated learning of large language models with parameter-efficient prompt tuning and adaptive optimization. [[Paper]](https://arxiv.org/abs/2310.15080)
* Efficient federated prompt tuning for black-box large pre-trained models. [[Paper]](https://arxiv.org/abs/2310.03123)
* Text-driven prompt generation for vision-language models in federated learning. [[Paper]](https://arxiv.org/abs/2310.06123)
* Learning federated visual prompt in null space for mri reconstruction. [[Paper]](https://arxiv.org/abs/2303.16181)
* Fed-cprompt: Contrastive prompt for rehearsal-free federated continual learning. [[Paper]](https://arxiv.org/abs/2307.04869)
* Fedprompt: Communication-efficient and privacy-preserving prompt tuning in federated learning. [[Paper]](https://arxiv.org/abs/2208.12268)
* Tunable soft prompts are messengers in federated learning. [[Paper]](https://arxiv.org/abs/2311.06805)
* Hepco: Data-free heterogeneous prompt consolidation for continual federated learning. [[Paper]](https://openreview.net/forum?id=dsWg7n6zoo)
* Prompt-enhanced Federated Learning for Aspect-Based Sentiment Analysis. [[Paper]](https://www.computer.org/csdl/proceedings-article/icicce/2023/956100a081/1WAIBqJhwpG)
* Towards practical few-shot federated nlp. [[Paper]](https://arxiv.org/abs/2212.00192)
* Federated prompting and chain-of-thought reasoning for improving llms answering. [[Paper]](https://arxiv.org/abs/2304.13911)
* FedHPL: Efficient Heterogeneous Federated Learning with Prompt Tuning and Logit Distillation. [[Paper]](https://arxiv.org/abs/2405.17267)
* Probabilistic Federated Prompt-Tuning with Non-IID and Imbalanced Data. [[Paper]](https://arxiv.org/abs/2502.19752)
* Federated Class-Incremental Learning with Prompting. [[Paper]](https://arxiv.org/abs/2310.08948)
* Explore and Cure: Unveiling Sample Effectiveness with Context-Aware Federated Prompt Tuning. [[Paper]](https://www.computer.org/csdl/journal/tm/2024/12/10629177/1ZdiYS9RJxC)
* Federated Prompt Learning for Weather Foundation Models on Devices. [[Paper]](https://arxiv.org/abs/2305.14244)

### 2.2 Personalized Prompt Tuning
* Efficient model personalization in federated learning via client-specific prompt generation. [[Paper]](https://arxiv.org/abs/2308.15367)
* Unlocking the potential of prompt-tuning in bridging generalized and personalized federated learning. [[Paper]](https://arxiv.org/abs/2310.18285)
* Pfedprompt: Learning personalized prompt for vision-language models in federated learning. [[Paper]](https://dl.acm.org/doi/10.1145/3543507.3583518)
* Global and local prompts cooperation via optimal transport for federated learning. [[Paper]](https://arxiv.org/abs/2403.00041)
* Visual prompt based personalized federated learning. [[Paper]](https://arxiv.org/abs/2303.08678)
* Personalized federated continual learning via multi-granularity prompt. [[Paper]](https://arxiv.org/abs/2407.00113)
* FedLPPA: Learning Personalized Prompt and Aggregation for Federated Weakly-supervised Medical Image Segmentation. [[Paper]](https://arxiv.org/abs/2402.17502)
* Harmonizing Generalization and Personalization in Federated Prompt Learning. [[Paper]](https://arxiv.org/abs/2405.09771)
* Tackling Feature-Classifier Mismatch in Federated Learning via Prompt-Driven Feature Transformation. [[Paper]](https://arxiv.org/abs/2407.16139)
* Personalized Federated Learning for Text Classification with Gradient-Free Prompt Tuning. [[Paper]](https://aclanthology.org/2024.findings-naacl.286.pdf)
* Mixture of Experts Made Personalized: Federated Prompt Learning for Vision-Language Models. [[Paper]](https://openreview.net/forum?id=xiDJaTim3P)
* CP 2 GFed: Cross-granular and Personalized Prompt-based Green Federated Tuning for Giant Models. [[Paper]](https://ieeexplore.ieee.org/document/10682866/)
  
### 2.3 Multi-domain Prompt Tuning
* DiPrompT: Disentangled Prompt Tuning for Multiple Latent Domain Generalization in Federated Learning. [[Paper]](https://arxiv.org/abs/2403.08506)
* Prompt-enhanced Federated Content Representation Learning for Cross-domain Recommendation. [[Paper]](https://arxiv.org/abs/2401.14678)
* Dual prompt tuning for domain-aware federated learning. [[Paper]](https://openreview.net/forum?id=pVaMBfI2eR)
* Federated adaptive prompt tuning for multi-domain collaborative learning. [[Paper]](https://arxiv.org/abs/2211.07864)
* Breaking physical and linguistic borders: Multilingual federated prompt tuning for low-resource languages. [[Paper]](https://openreview.net/forum?id=zzqn5G9fjn)
* Federated Domain Generalization via Prompt Learning and Aggregation. [[Paper]](https://arxiv.org/abs/2411.10063)
* CP-Prompt: Composition-Based Cross-modal Prompting for Domain-Incremental Continual Learning. [[Paper]](https://arxiv.org/abs/2407.21043)

## Part 3: Adapter-based Tuning
### 3.1 General Adapter Tuning
* Efficient federated learning for modern nlp. [[Paper]](https://arxiv.org/abs/2205.10162)
* Efficient federated learning with pre-trained large language model using several adapter mechanisms. [[Paper]](https://www.mdpi.com/2227-7390/11/21/4479)
  
### 3.2 Personalized Adapter Tuning
* Client-customized adaptation for parameter-efficient federated learning. [[Paper]](https://aclanthology.org/2023.findings-acl.75/)
* Fedclip: Fast generalization and personalization for clip in federated learning. [[Paper]](https://arxiv.org/abs/2302.13485)

### 3.3 Multi-domain Adapter Tuning
* Communication efficient federated learning for multilingual neural machine translation with adapter. [[Paper]](https://arxiv.org/abs/2305.12449)
* Adapter-based Selective Knowledge Distillation for Federated Multi-domain Meeting Summarization. [[Paper]](https://arxiv.org/abs/2308.03275)
* Feddat: An approach for foundation model finetuning in multi-modal heterogeneous federated learning. [[Paper]](https://arxiv.org/abs/2308.12305)

## Part 4: Selective-based Tuning
  
### 4.1 Bias Tuning
* Differentially private bias-term only fine-tuning of foundation models. [[Paper]](https://arxiv.org/abs/2210.00036)
* Conquering the communication constraints to enable large pre-trained models in federated learning. [[Paper]](https://arxiv.org/html/2210.01708v3)

### 4.2 Partial Tuning
* Bridging the gap between foundation models and heterogeneous federated learning. [[Paper]](https://arxiv.org/abs/2310.00247)
* Exploring Selective Layer Fine-Tuning in Federated Learning. [[Paper]](https://openreview.net/forum?id=eu1PIDPYwC)
  
## Part 5: Other Tuning Methods

### 5.1 Zero-Order Optimization
* Federated full-parameter tuning of billion-sized language models with communication cost under 18 kilobytes. [[Paper]](https://arxiv.org/abs/2312.06353)
* $\{$FwdLLM$\}$: Efficient Federated Finetuning of Large Language Models with Perturbed Inferences. [[Paper]](https://www.usenix.org/conference/atc24/presentation/xu-mengwei)
* ZooPFL: Exploring black-box foundation models for personalized federated learning. [[Paper]](https://arxiv.org/abs/2310.05143)
* On the convergence of zeroth-order federated tuning for large language models. [[Paper]](https://arxiv.org/abs/2402.05926)
* Thinking Forward: Memory-Efficient Federated Finetuning of Language Models. [[Paper]](https://arxiv.org/abs/2405.15551)
* Communication-Efficient Byzantine-Resilient Federated Zero-Order Optimization. [[Paper]](https://arxiv.org/abs/2406.14362)


### 5.2 Split Learning
* FedBERT: When federated learning meets pre-training. [[Paper]](https://dl.acm.org/doi/10.1145/3510033)
* Federated split bert for heterogeneous text classification. [[Paper]](https://arxiv.org/abs/2205.13299)
* FedSplitX: Federated Split Learning for Computationally-Constrained Heterogeneous Clients. [[Paper]](https://arxiv.org/abs/2310.14579)
  
### 5.3 Model Compression
* Fedbiot: Llm local fine-tuning in federated learning without full model. [[Paper]](https://arxiv.org/abs/2406.17706)
  
### 5.4 Data Selection
* Federated Data-Efficient Instruction Tuning for Large Language Models. [[Paper]](https://arxiv.org/abs/2410.10926)

<!-- omit in toc -->
## ⭐ Citation
If you find this work useful, welcome to cite us.

```bib
@inproceedings{
ye202512surveyfedllm,
      title={A Survey on Large Language Models Federated Learning}, 
      author={Yebo Wu and Chunlin Tian and Kahou Tam and Li Li and Chengzhong Xu},
      year={2025}
}
```
