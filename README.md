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

Coding an LLM from scratch is an excellent exercise for understanding its mechanisms and limitations. Furthermore, since most modern LLMs are built using the PyTorch deep learning library, this project provides me with valuable experience with the industry's leading tool. 
