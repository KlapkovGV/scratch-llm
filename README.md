# scratch-llm
This repository contains my personal implementation and notes as I build a Large Language Model (LLM) based on the Generative Pre-trained Transformer (GPT) architecture from the ground up. The goal is to understand the inner workings of LLMs. This project follows the methodology described in the book "Build a Large Language Model (from Scratch)" by Sebastian Raschka.

## What is an LLM?

Large Language Models (LLMs) are deep neural network models designed to understand, generate, and interpret human language. It is important to clarify that when we say a model "understands," we mean it can process and produce text that is coherent and contextually relevant, rather than possessing actual consciousness. Today, LLMs are used for a wide range of NLP tasks, including text classification, translation, sentiment analysis, and many more.

Earlier NLP models were typically narrow and designed for specific tasks (like basic human-to-machine communication or simple filtering). In contrast, modern LLMs demonstrate broad proficiency, meaning they can handle many different types of tasks simultaneously without needing a separate model for each one.

The breakthrough of LLMs is built on two main pillars:
1. The Transformer Architecture: A specific design that allows the model to "pay attention" to the most important parts of the text.
2. Massive Training Data: LLMs are trained on vast amounts of text using unsupervised learning. By predicting words in large datasets, the model learns the statistical patterns and nuances of natural language.
