## Attention Mechanisms

Now that I have finished the embedding pipeline, I am moving to the Attention Mechanism, a crucial building block of an LLM. My goal in this section is to implement and understand how the model focuses on the most relevant parts of the input data when generating outputs.

I want to build Multi-Head Attention,but first, I will implement its components: Simplified Self-Attention, Self-Attention (with trainable weights), and Causal Attention.

It is also important to mention the foundational paper 'Attention Is All You Need' (Vaswani et al., 2017). As I understand the Transformer architecture, Multi-Head Attention works by performing "linear projections" of the input vectors. Instead of computing attention once, the model runs the attention function in parallel across these projected versions.
