# scratch-llm
This repository contains my personal implementation and notes as I build a Large Language Model (LLM) based on the Generative Pre-trained Transformer (GPT) architecture from the ground up. The goal is to understand the inner workings of LLMs. This project follows the methodology described in the book "Build a Large Language Model (from Scratch)" by Sebastian Raschka.

## Fundamentals of Large Language Models and Artificail Intelligence

Large Language Models (LLMs) are deep neural network models designed to understand, generate, and interpret human language. The term 'understand' specifically means the model's capability to output contextually accurate and coherent responses. Today, LLMs are used for a wide range of NLP tasks, including text classification, translation, sentiment analysis, and many more.

Earlier NLP models were typically narrow and designed for specific tasks (like basic human-to-machine communication or simple filtering). Now, modern LLMs demonstrate broad proficiency, meaning they can handle many different types of tasks simultaneously without needing a separate model for each one.

The term 'large' in Large Language Model (LLM) refers to two main factors: the model's total parameter count and the massive dataset on which it is trained. Regarding parameters, modern models typically consist of billions of them. During training, these internal weights are adjusted to master next-token prediction. By leveraging the sequential nature of language, the model learns to recognize complex relationships and structural nuances within the text.

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

According to the article, the Transformer use an encoder-decoder architecture. The encoder maps an input sequence of symbol representations (x1, x2, ..., xn) to a sequence of continuous representations (z1, z2, ..., zn). The decoder then uses z to generate an output sequence, one element at a time. A key feature of this process is that the model is auto-regressive, meaning it uses previously generated symbols/token as input when generating the next one.

A key component of the Transformer is the self-attention mechanism. It weighs the importance of different words or tokens in a sequence in order to capture contextual relationships and long-range dependencies.






