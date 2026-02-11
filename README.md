# Natural Language Processing & Large Language Models  


This course introduces Natural Language Processing (NLP) and transformer-based Large Language Models (LLMs). Students will explore foundational NLP concepts, including tokenization, word embeddings, and language modelling. They will learn the core mechanics of LLMs, such as architecture, training, fine-tuning, reasoning, evaluation, and deployment strategies. The curriculum includes practical applications such as text classification, machine translation, summarization, and zero-/few-shot prompting. 

Through hands-on work with real-world datasets, students will design NLP pipelines and evaluate model performance in multilingual settings, with particular emphasis on low-resource and under-represented languages. By the end of the course, students will also build a simple language model from scratch.

**Note:** We are pairing the ArewaDS programme with the [Google DeepMind: AI Research Foundations learning path](https://www.skills.google/paths/3135). To receive an ArewaDS certificate, participants must complete all courses in the learning path and submit their certificates of completion as evidence.

## Course Updates & Announcements
For the latest course schedule updates and important announcements, please join our Discord community: 
👉 [Join the Discord channel](https://discord.gg/ACt7uGFcw)

## **Part  A: Natural Language Processing**


| Lecture | Title                                  | Resources | YouTube Videos | Suggested Readings |
|--------:|----------------------------------------|-----------|----------------|--------------------|
| **1** | Introduction to NLP and LLMs - (07-Feb-2026) | [Slide](https://github.com/shmuhammadd/aims-nlp-course/blob/main/slides/01_NLP_Lecture.pdf) | [YouTube](https://youtu.be/ffPT5ddW-T4) | 1. [Natural Language Processing: State of the Art, Current Trends and Challenges](https://arxiv.org/pdf/1708.05148)<br>2. [The Rise of AfricaNLP: Contributions, Contributors, and Community Impact (2005–2025)](https://arxiv.org/pdf/2509.25477)<br>3. [HausaNLP: Current Status, Challenges and Future Directions for Hausa NLP](https://arxiv.org/pdf/2505.14311) |
| **2** | How Language Modelling Started (N-grams) - (-08-Feb-2026) | [Slide](https://github.com/shmuhammadd/aims-nlp-course/blob/main/slides/02_NLP_Lecture.pdf)<br><br>[Practical ![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arewadataScience/nlp-llm-course/blob/main/practicals/ngram_language_models_class.ipynb)<br>[Exercise ![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arewadataScience/nlp-llm-course/blob/main/practicals/ngram_practice_huggingface.ipynb) | [YouTube](https://youtu.be/kUKwWIOthUE) | 1. Jurafsky & Martin — *Speech and Language Processing*, Chapter 3<br>2. Rosenfeld (2000) — [Two Decades of Statistical Language Modeling](https://www.cs.cmu.edu/~roni/papers/survey-slm-IEEE-PROC-0004.pdf) |
| **3** | Text Classification | [Slide](https://github.com/shmuhammadd/aims-nlp-course/blob/main/slides/03_NLP_Lecture.pdf)<br><br>[Intro to PyTorch ![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shmuhammadd/aims-nlp-course/blob/main/practicals/pytorch_intro_notebook.ipynb) | — | 1. Jurafsky & Martin — *Speech and Language Processing*, Chapter 4<br>2. Muhammad et al. (2022) — [AfriSenti](https://arxiv.org/pdf/2302.08956)<br>3. [Learn PyTorch: Zero to Mastery](https://www.learnpytorch.io) |
| **4** | Word Vectors | [Slide](https://github.com/shmuhammadd/aims-nlp-course/blob/main/slides/04_NLP_Lecture.pdf)<br><br>[Training Embeddings ![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shmuhammadd/aims-nlp-course/blob/main/practicals/word_embeddings_lab.ipynb) | — | 1. Mikolov et al. (2013) — [Efficient Estimation of Word Representations](https://arxiv.org/pdf/1301.3781)<br>2. Mikolov et al. (2013) — [Linguistic Regularities](https://arxiv.org/pdf/1310.4546) |
| **5** | Sequence Modelling | [Slide](https://github.com/shmuhammadd/aims-nlp-course/blob/main/slides/05_NLP_Lecture.pdf)<br><br>[Sentiment Analysis ![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shmuhammadd/aims-nlp-course/blob/main/practicals/text_classification_pytorch.ipynb) | — | 1. Goodfellow et al. — *Deep Learning*, Chapter 6<br>2. Goldberg (2016) — [Neural Network Models for NLP](https://arxiv.org/pdf/1510.00726) |
| **6** | Attention | [Slide](https://github.com/shmuhammadd/aims-nlp-course/blob/main/slides/06_NLP_Lecture.pdf)<br><br>[Attention ![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shmuhammadd/aims-nlp-course/blob/main/practicals/attention_practical.ipynb)<br>[Exercise ![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shmuhammadd/aims-nlp-course/blob/main/practicals/attention_exercises.ipynb) | — | 1. Bahdanau et al. (2014) — [Neural Machine Translation](https://arxiv.org/abs/1409.0473)<br>2. Luong et al. (2015) — [Attention-based NMT](https://aclanthology.org/D15-1166/) |

## **Part B: Large Language Models**

| Lecture | Title                                   | Resources | Suggested Readings |
|---------|-----------------------------------------|-----------|---------------------|
| **7**   | Introduction to Transformers            | [Slide 1](https://github.com/shmuhammadd/aims-nlp-course/blob/main/slides/07_NLP_Lecture_1.pdf), [Slide 2](https://github.com/shmuhammadd/aims-nlp-course/blob/main/slides/07_NLP_Lecture_2.pdf)   | 1. [Vaswani et al. (2017) — Attention is All You Need](https://arxiv.org/pdf/1706.03762) <br> 2. [Alammar — Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) |
| **8**   | Pretraining   | [Slide 1](https://github.com/shmuhammadd/aims-nlp-course/blob/main/slides/08_NLP_Lecture_1.pdf), [Slide 2](https://github.com/shmuhammadd/aims-nlp-course/blob/main/slides/08_NLP_Lecture_2.pdf)    <br><br>  [Pre-training ![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shmuhammadd/aims-nlp-course/blob/main/practicals/pretraining_encoder_decoder.ipynb) <br> [Fine-tuning ![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shmuhammadd/aims-nlp-course/blob/main/practicals/finetuning_afroXLMR_afrisenti.ipynb) <br> [Exercise ![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shmuhammadd/aims-nlp-course/blob/main/practicals/practice_exercises.ipynb)  | 1. [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/pdf/1810.04805) <br> 2. [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165)  |
| **9**   |  Post-training       | [Slide 1](https://github.com/shmuhammadd/aims-nlp-course/blob/main/slides/08_NLP_Lecture_1.pdf), [Slide 2](https://github.com/shmuhammadd/aims-nlp-course/blob/main/slides/08_NLP_Lecture_2.pdf)        | 1. [FLAN: Finetuned Language Models](https://arxiv.org/pdf/2109.01652) <br> 2. [T0: Multitask Prompted Training](https://arxiv.org/pdf/2110.08207) |
| **10**   | Model Compression            | [Slide](https://github.com/shmuhammadd/aims-nlp-course/blob/main/slides/08_NLP_Lecture_1.pdf)       | 1. [Wei et al. (2022) — Chain-of-Thought Prompting](https://arxiv.org/pdf/2201.11903) <br> 2. [Kojima et al. (2022) — Zero-Shot CoT](https://arxiv.org/pdf/2205.11916) |
| **11**  | Benchmarking and Evaluation | [Slide](https://github.com/shmuhammadd/aims-nlp-course/blob/main/slides/08_NLP_Lecture_1.pdf) | 1. [Holistic Evaluation of Language Models (HELM)](https://arxiv.org/pdf/2211.09110)   |


## Project

 Each student will conduct a project. More detalais coming soon.

 ## Mentors

 

#  Resources  

1.  **Speech and Language Processing** – Jurafsky & Martin ([Online Draft](https://web.stanford.edu/~jurafsky/slp3/))  
2.  [Hands-On Large Language Models: Language Understanding and Generation](https://www.amazon.in/Hands-Large-Language-Models-Understanding/dp/935542552X/ref=pd_sbs_d_sccl_1_1/521-7549942-9569643?pd_rd_w=Ueibj&content-id=amzn1.sym.6d240404-f8ea-42f5-98fe-bf3c8ec77086&pf_rd_p=6d240404-f8ea-42f5-98fe-bf3c8ec77086&pf_rd_r=Z9BASYAF4RW1MVP0D173&pd_rd_wg=ZUKds&pd_rd_r=95ab3bb8-4c74-458a-8089-fa654d4b720c&pd_rd_i=935542552X&psc=1) 
3.  [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
4.  [LLM-course](https://github.com/mlabonne/llm-course)  
5. **Natural Language Processing with Python** – Steven Bird, Ewan Klein, Edward Loper ([Free Online](https://www.nltk.org/book/))  
6. **Transformers for Natural Language Processing** – Denis Rothman  
7. **Deep Learning for NLP** – Palash Goyal, Sumit Pandey, Karan Jain  
8. **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow** – Aurélien Géron  
