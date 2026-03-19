# scratch-llm
This repository contains my personal implementation and notes as I build a Large Language Model (LLM) based on the Generative Pre-trained Transformer (GPT) architecture from the ground up. The goal is to understand the inner workings of LLMs. This project follows the methodology described in the book "Build a Large Language Model (from Scratch)" by Sebastian Raschka.

## Fundamentals of Large Language Models and Artificail Intelligence

Large Language Models (LLMs) are deep neural network models designed to understand, generate, and interpret human language. The term 'understand' specifically means the model's capability to output contextually accurate and coherent responses. Today, LLMs are used for a wide range of NLP tasks, including text classification, translation, sentiment analysis, and many more.

Earlier NLP models were typically narrow and designed for specific tasks (like basic human-to-machine communication or simple filtering). Now, modern LLMs demonstrate broad proficiency, meaning they can handle many different types of tasks simultaneously without needing a separate model for each one.

The term 'Large' in Large Language Model (LLM) refers to two main factors: the model's total parameter count and the massive dataset on which it is trained. Regarding parameters, modern models typically consist of billions of them. During training, these internal weights are adjusted to master next-token prediction. By leveraging the sequential nature of language, the model learns to recognize complex relationships and structural nuances within the text.

I also want to mention that LLMs utilize the Transformer architecture. This architecture is a specific design that allows the model to 'pay attention' to the most important parts of the text.

Moving forward, I would like to discuss the well-known heirarchy of Artificail Intelligence. This framework is familiar to anyone studying the fiel, as it illustrates how different concepts are nested within one another. Artificial Intelligence serves as the broad field dedicated to creating machines capable of performing tasks that typically require human-like intelligence. Within this field, Machine Learning represents a subfield focused on developing algorithms that learn from data to make predictions. Deep Learning is a branch of Machine Learning that utilizes neural networks with three or more layers to model complex patterns and abstractions in data.

A fundamental distinction between machine learning and deep learning lies in how data features are identified. Machine Learning requires manual feature extraction, a process where humans must identify and select the most relevant characteristics for the model to analyze. Deep Learning eliminates the need for manual selection by automatically identifying and extracting relevant features through its neural network layers. Despite this difference, both approaches generally rely on labeled datasets and a training process designed to minimize prediction errors.

Large Language Models represent a specific application of deep learning techniques, specifically designed to process and generate human-like text. Since LLMs can generate new content, they are categorized as a form of Generative Artificial Intelligence, or GenAI.

Why did I start this project? First, LLMs possess advanced capabilities to parse and understand unstructured text data across nearly every occupation. According to the recent Anthropic report, 'Labor Market Impacts of AI: A New Measure and Early Evidence' (published March 5, 2026, by Maxim Massenkoff and Peter McCrory), the most significant impacts are seen in business, finance, computer science, legal, and administrative sectors.

Coding an LLM from scratch will be an excellent exercise for me to understand its mechanisms and limitations. Since I will be using PyTorch, this project will give me valuable experience with a tool used to build LLMs.

To create an LLM, we need a vast amount of raw data. Here, raw data refers to ordinary text that has no manual labeling. After collecting the raw data, we train the LLM on this text. Upon completion of the training stage, we obtain a pre-trained LLM, also known as a foundation model. Theoretically, such a model is capable of text completion (completing an incomplete sentence) and possesses limited few-shot learning capabilities.

Also, LLMs use self-supervised learning during the pre-trained stage, which allows the model to generate its own labels directly from unlabeled data. This enables the model to learn without human annotation. Specifically, the model learns by predicting parts of the input data that have been intentionally masked. 

As I understand it, once we have obtained a pre-trained LLM, it can be further enhanced by training it on labeled data. This process is called fine-tuning. A fine-tuned LLM is used for tasks like classification, summarization, translation, building personal assistants, and more.

### Transformer architecture and Attention mechanism

The Transformer was introduced by a Google team in the article 'Attention Is All You Need' (Vaswani et al., 2017). Since modern LLMs rely on the Transformer architecture, I will examine it to better understand them.

Vaswani et al. proposed the Transformer, a model architecture that avoid recurrence and relies entirely on an attention mechanism to draw global dependencies between input and output. Before Transformers, models like Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM), and Gated Recurrent Units (GRU) were the standard. These models processed text sequentially, word by word. Unlike these models, the Transformer can process entire sequences of data simultaneously due to parallelization.

### Encoder and Decoder structure

According to the article, the Transformer use an encoder-decoder architecture. The encoder maps an input sequence of symbol representations (x1, x2, ..., xn) to a sequence of continuous representations (z1, z2, ..., zn). The decoder then uses z to generate an output sequence, one element at a time. A key feature of this process is that the model is auto-regressive, meaning it uses previously generated symbols/token as input when generating the next one.

The encoder is composed of a stack of Nx identical layers. Each layer is a small block with two sub-layers. The first sub-layer is a multi-head self-attention mechanism, and the second is a position-wise feed-forward network. Around each of these sub-layers, the authors apply a residual connection followed by layer normalization.

- a residual connection means that instead of passing only the result of a sub-layer forward, they add the original input back to it. I guess it is for gradients during backpropagation;
- layer normalization stabilizes the output by rescaling the values.

The decoder is also composed of a stack of Nx identical layers. It follows the same structure but introduces a third sub-layer between the self-attention and feed-forward sub-layers. This additional sub-layer performs multi-head attention over the output of the encoder. Also, the decoder uses masked multi-head attention, which masks future positions to ensure that predictions for a given token can only depend on the known outputs at previous positions.

### Attention Function

According to the article, the attention function takes three vectors as input a **Query (Q)**, a **Key (K)**, and a **Value (V)**, and maps them to an output. Particualr attention, which is authors called **Scaled Dot-Product Attention**, where the computation is performed on a set of queries packed into a matrix Q, with key and values packed into matrices K and V. Softmax is used to convert the scores into a probability distribution.

Instead of performing a single attention function, the authors propose **Multi-Head Attention**, which runs the attention function h times in parallel, each time with different learned linear projections of Q, K, and V. This allows the model to jointly attend to information from different representation subspaces at different positions.

### Positional encoding

The Transformer has no built-in sense of token order since it has no recurrence. To address this, positional encodings are added to the input embeddings. These are vectors computed using sine and cosine functions of different frequencies.

### About GPT-1

The first GPT was introduced by the OpenAI team in the paper "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018). The authors proposed an approach based on a two-stage training procedure. **Unsupervised pre-training**, as the first stage, trains the model on a large corpus of unlabeled text using a multi-layer Transformer decoder as the language model. The second stage is **supervised fine-tuning**, where the pre-trained model's parameters are adapted to a specific supervised task, such as text classification, question answering, and semantic similarity assessment.

## How I will create a large language model

As I mentioned earlier, I will create a large language model following the methodology described in the book "Build a Large Language Model (from Scratch)" by Sebastian Raschka. At the end of the first chapter there is a clear diagram of the process of creating an LLM. The process consists of three main stages.

### Stage 1 — Building an LLM

The first stage involves data preparation and sampling, implementing the attention mechanism, and building the LLM architecture. After the model is built, it is ready for pretraining.

### Stage 2 — Obtaining a Foundation Model

To obtain a foundation model, the LLM is trained on unlabeled data, evaluated, and the pretrained weights are loaded. At this stage, the model is already capable of generating new text.

### Stage 3 — Fine-tuning

The third stage is project-specific as I understand, and requires defining the goal of the LLM. So the basis of the third stage is a labeled dataset that is loaded and used for fine-tuning the pretrained model. After fine-tuning, the model can be used for specific purposes, for example, as a text classifier or a personal assistant.

## How I understand data preprocessing

### Why we need word embeddings

Raw text cannot be directly processed by an LLM because LLM is a deep neural network model, and neural networks cannot work with categorical data. The LLM requires vector formats to perform mathematical operations during training. Therefore, raw text must be represented as continuous vectors.

This process relates to a fundamental concept in machine learning and AI called **embedding**. Embedding is the concept of transforming data into a numerical vector representation that computers can process and understand. It is worth mentioning that different data formats require distinct embedding models — text, video, and audio each have their own embedding approaches.

A vector is simply a list of numbers. For example, the word 'cat' can be represented as [0.2, -0.5, 1.3, 0.8]. The higher the dimensionality of the vector, the more information it can store. The key idea is that similar words have similar vectors. 

It is also worth mentioning that LLMs do not rely on a separate pre-built word embedding model. Instead, LLMs create their own embedding layers as part of the LLM pipeline. 

Embeddings need to be optimized for the specific task. In the context of LLMs, embeddings have a much higher dimensionality compared to traditional word embeddings, because the model needs to capture complex relationships and patterns across large amounts of text.

The required steps for preparing embeddings used by an LLM are:
- splitting text into words and converting words into tokens;
- turning tokens into embedding vectors.

### How input text be splited into tokens

In this section I want to show how I understand the tokenization process works using a simple example with Python's regular expression library. It is also worth mentioning that LLMs use prebuilt tokenizers in practice — for example, tiktoken from OpenAI, which is used in GPT. Such tokenizers are already optimized, work faster, and use BPE.

Splitting text into individual tokens is a required preprocessing step for creating embeddings for an LLM. I practiced this with an example from Raschka's repository on GitHub, using the file "the-verdict.txt". First, the file contains a total of 20,479 characters, which need to be tokenized into individual words and special characters. These tokens are then converted into token IDs using a vocabulary, and later into embedding vectors that the LLM can process.

To split the text I used Python's regular expression library `re`. The splitting is performed using the `re.split` command: `re.split(r'([,.:;?_!"()\']|--|\s)', text)`. Here, each character inside the brackets — such as commas, periods, punctuation marks, question marks, and quotation marks — is treated as a separate token. Additionally, double dashes -- and whitespaces \s are also split into separate tokens via the | operator. After splitting, whitespaces are removed by `strip()` command so that the tokenizer works more efficiently. The first 10 tokens of the resulting output are: `['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', 'rather', 'a', 'cheap', 'genius']`.

### How tokens are converted to token IDs

Token IDs are an intermediate representation used before converting tokens into embedding vectors. Each token ID is an integer in the range `[0, vocab_size-1]`.

In a basic tokenizer, the mapping tokens to token IDs is done via a dictionary built from a training dataset. This is the simple approach. The vocabulary collects all unique tokens (words and special characters), sorts them alphabetically, and assigns each a unique integer. 

A major limitation of this approach is that it struggles with unknown words. Words that were not present in the training vocabulary. Depending on the implementation, it either raises an error or replaces the unknown word with a special '<unk>' token.

As I understand, GPT models use **Byte Pair Encoding (BPE)**, a more modern tokenization approach. I found a clear implementation of BPE in 'tiktoken', an open-source library from OpenAI. BPE is a tokenization algorithm that converts text into tokens. It can handle any words, for example, common words are converted into tokens directly, while unknown words are broken down into smaller subword and then represented as a sequence of subword tokens.
