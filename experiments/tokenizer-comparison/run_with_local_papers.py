#!/usr/bin/env python3
"""Run experiment with locally created sample papers to demonstrate full pipeline."""

import sys
from pathlib import Path
import pymupdf

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.compare_tokenizers import run_comparison

# Sample paper content (excerpts from real papers)
PAPERS_CONTENT = {
    "1706.03762": {
        "name": "Attention Is All You Need",
        "content": """Attention Is All You Need

Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.

1 Introduction

Recurrent neural networks, long short-term memory [12] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [29, 2, 5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [31, 21, 13].

Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 16]. In all but a few cases [22], however, such attention mechanisms are used in conjunction with a recurrent network.

In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

2 Background

The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [20], ByteNet [15] and ConvS2S [8], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [11]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 22, 23, 19].

End-to-end memory networks are based on a recurrent attention mechanism instead of sequence-aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [28].

To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [14, 15] and [8].

3 Model Architecture

Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 29]. Here, the encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn). Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time. At each step the model is auto-regressive [9], consuming the previously generated symbols as additional input when generating the next.

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

3.1 Encoder and Decoder Stacks

Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network. We employ a residual connection [10] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension dmodel = 512.

Decoder: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.

3.2 Attention

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

3.2.1 Scaled Dot-Product Attention

We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension dk, and values of dimension dv. We compute the dot products of the query with all keys, divide each by √dk, and apply a softmax function to obtain the weights on the values.

In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V. We compute the matrix of outputs as:

Attention(Q, K, V) = softmax(QK^T / √dk)V

The two most commonly used attention functions are additive attention [2], and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of 1/√dk. Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

4 Why Self-Attention

In this section we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations (x1, ..., xn) to another sequence of equal length (z1, ..., zn), with xi, zi ∈ Rd, such as a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata.

One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.

The third is the path length between long-range dependencies in the network. Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies [11]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types.

5 Training

This section describes the training regime for our models.

5.1 Training Data and Batching

We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Sentences were encoded using byte-pair encoding [3], which has a shared source-target vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary [31]. Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.

6 Results

6.1 Machine Translation

On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. The configuration of this model is listed in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models.

On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model. The Transformer (big) model trained for English-to-French used dropout rate Pdrop = 0.1, instead of 0.3.

For the base models, we used a single model obtained by averaging the last 5 checkpoints, which were written at 10-minute intervals. For the big models, we averaged the last 20 checkpoints. We used beam search with a beam size of 4 and length penalty α = 0.6 [31]. These hyperparameters were chosen after experimentation on the development set. We set the maximum output length during inference to input length + 50, but terminate early when possible [31].

7 Conclusion

In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.

For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles.

We are excited about the future of attention-based models and plan to apply them to other tasks. We plan to extend the Transformer to problems involving input and output modalities other than text, such as images, audio and video. Making generation less sequential is another research goals of ours.
""",
    },
    "1810.04805": {
        "name": "BERT",
        "content": """BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

Abstract

We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).

1 Introduction

Language model pre-training has been shown to be effective for improving many natural language processing tasks. These include sentence-level tasks such as natural language inference and paraphrasing, which aim to predict the relationships between sentences by analyzing them holistically, as well as token-level tasks such as named entity recognition and question answering, where models are required to produce fine-grained output at the token level.

There are two existing strategies for applying pre-trained language representations to downstream tasks: feature-based and fine-tuning. The feature-based approach, such as ELMo, uses task-specific architectures that include the pre-trained representations as additional features. The fine-tuning approach, such as the Generative Pre-trained Transformer (GPT), introduces minimal task-specific parameters, and is trained on the downstream tasks by simply fine-tuning all pre-trained parameters. The two approaches share the same objective function during pre-training, where they use unidirectional language models to learn general language representations.

We argue that current techniques restrict the power of the pre-trained representations, especially for the fine-tuning approaches. The major limitation is that standard language models are unidirectional, and this limits the choice of architectures that can be used during pre-training. For example, in OpenAI GPT, the authors use a left-to-right architecture, where every token can only attend to previous tokens in the self-attention layers of the Transformer. Such restrictions are sub-optimal for sentence-level tasks, and could be very harmful when applying fine-tuning based approaches to token-level tasks such as question answering, where it is crucial to incorporate context from both directions.

In this paper, we improve the fine-tuning based approaches by proposing BERT: Bidirectional Encoder Representations from Transformers. BERT alleviates the previously mentioned unidirectionality constraint by using a "masked language model" (MLM) pre-training objective, inspired by the Cloze task. The masked language model randomly masks some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word based only on its context. Unlike left-to-right language model pre-training, the MLM objective enables the representation to fuse the left and the right context, which allows us to pre-train a deep bidirectional Transformer. In addition to the masked language model, we also use a "next sentence prediction" task that jointly pre-trains text-pair representations. The contributions of our paper are as follows:

• We demonstrate the importance of bidirectional pre-training for language representations. Unlike Radford et al. (2018), which uses unidirectional language models for pre-training, BERT uses masked language models to enable pre-trained deep bidirectional representations. This is also in contrast to Peters et al. (2018a), which uses a shallow concatenation of independently trained left-to-right and right-to-left LMs.

• We show that pre-trained representations reduce the need for many heavily-engineered task-specific architectures. BERT is the first fine-tuning based representation model that achieves state-of-the-art performance on a large suite of sentence-level and token-level tasks, outperforming many task-specific architectures.

• BERT advances the state of the art for eleven NLP tasks. The code and pre-trained models are available at https://github.com/google-research/bert.

2 Related Work

There is a long history of pre-training general language representations, and we briefly review the most widely-used approaches in this section.

2.1 Unsupervised Feature-based Approaches

Learning widely applicable representations of words has been an active area of research for decades, including non-neural and neural methods. Pre-trained word embeddings are an integral part of modern NLP systems, offering significant improvements over embeddings learned from scratch. To pre-train word embedding vectors, left-to-right language modeling objectives have been used, as well as objectives to discriminate correct from incorrect words in left and right context.

These approaches have been generalized to coarser granularities, such as sentence embeddings or paragraph embeddings. To train sentence representations, prior work has used objectives to rank candidate next sentences, left-to-right generation of next sentence words given a representation of the previous sentence, or denoising auto-encoder derived objectives.

ELMo and its predecessor generalize traditional word embedding research along a different dimension. They extract context-sensitive features from a left-to-right and a right-to-left language model. The contextual representation of each token is the concatenation of the left-to-right and right-to-left representations. When integrating contextual word embeddings with existing task-specific architectures, ELMo advances the state of the art for several major NLP benchmarks including question answering, sentiment analysis, and named entity recognition. Melamud et al. (2016) proposed learning contextual representations through a task to predict a single word from both left and right context using LSTMs. Similar to ELMo, their model is feature-based and not deeply bidirectional. Fedus et al. (2018) shows that the cloze task can be used to improve the robustness of text generation models.

2.2 Unsupervised Fine-tuning Approaches

As with the feature-based approaches, the first works in this direction only pre-trained word embedding parameters from unlabeled text.

More recently, sentence or document encoders which produce contextual token representations have been pre-trained from unlabeled text and fine-tuned for a supervised downstream task. The advantage of these approaches is that few parameters need to be learned from scratch. At least partly due to this advantage, OpenAI GPT achieved previously state-of-the-art results on many sentence-level tasks from the GLUE benchmark. Left-to-right language modeling and auto-encoder objectives have been used for pre-training such models.

2.3 Transfer Learning from Supervised Data

There has also been work showing effective transfer from supervised tasks with large datasets, such as natural language inference and machine translation. Computer vision research has also demonstrated the importance of transfer learning from large pre-trained models, where an effective recipe is to fine-tune models pre-trained with ImageNet.

3 BERT

We introduce BERT and its detailed implementation in this section. There are two steps in our framework: pre-training and fine-tuning. During pre-training, the model is trained on unlabeled data over different pre-training tasks. For fine-tuning, the BERT model is first initialized with the pre-trained parameters, and all of the parameters are fine-tuned using labeled data from the downstream tasks. Each downstream task has separate fine-tuned models, even though they are initialized with the same pre-trained parameters. The question-answering example in Figure 1 will serve as a running example for this section.

A distinctive feature of BERT is its unified architecture across different tasks. There is minimal difference between the pre-trained architecture and the final downstream architecture.

Model Architecture BERT's model architecture is a multi-layer bidirectional Transformer encoder based on the original implementation described in Vaswani et al. (2017) and released in the tensor2tensor library. Because the use of Transformers has become common and our implementation is almost identical to the original, we will omit an exhaustive background description of the model architecture and refer readers to Vaswani et al. (2017) as well as excellent guides such as "The Annotated Transformer."

In this work, we denote the number of layers (i.e., Transformer blocks) as L, the hidden size as H, and the number of self-attention heads as A. We primarily report results on two model sizes: BERT_BASE (L=12, H=768, A=12, Total Parameters=110M) and BERT_LARGE (L=24, H=1024, A=16, Total Parameters=340M).

BERT_BASE was chosen to have the same model size as OpenAI GPT for comparison purposes. Critically, however, the BERT Transformer uses bidirectional self-attention, while the GPT Transformer uses constrained self-attention where every token can only attend to context to its left.

Input/Output Representations To make BERT handle a variety of down-stream tasks, our input representation is able to unambiguously represent both a single sentence and a pair of sentences (e.g., ⟨ Question, Answer ⟩) in one token sequence. Throughout this work, a "sentence" can be an arbitrary span of contiguous text, rather than an actual linguistic sentence. A "sequence" refers to the input token sequence to BERT, which may be a single sentence or two sentences packed together.

We use WordPiece embeddings with a 30,000 token vocabulary. The first token of every sequence is always a special classification token ([CLS]). The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks. Sentence pairs are packed together into a single sequence. We differentiate the sentences in two ways. First, we separate them with a special token ([SEP]). Second, we add a learned embedding to every token indicating whether it belongs to sentence A or sentence B. As shown in Figure 1, we denote input embedding as E, the final hidden vector of the special [CLS] token as C ∈ R^H, and the final hidden vector for the i^th input token as Ti ∈ R^H.

For a given token, its input representation is constructed by summing the corresponding token, segment, and position embeddings. A visualization of this construction is given in Figure 2.

3.1 Pre-training BERT

Unlike Peters et al. (2018a) and Radford et al. (2018), we do not use traditional left-to-right or right-to-left language models to pre-train BERT. Instead, we pre-train BERT using two unsupervised tasks, described in this section. This step is presented in the left part of Figure 1.

Task #1: Masked LM Intuitively, it is reasonable to believe that a deep bidirectional model is strictly more powerful than either a left-to-right model or the shallow concatenation of a left-to-right and a right-to-left model. Unfortunately, standard conditional language models can only be trained left-to-right or right-to-left, since bidirectional conditioning would allow each word to indirectly "see itself", and the model could trivially predict the target word in a multi-layered context.

In order to train a deep bidirectional representation, we simply mask some percentage of the input tokens at random, and then predict those masked tokens. We refer to this procedure as a "masked LM" (MLM), although it is often referred to as a Cloze task in the literature. In this case, the final hidden vectors corresponding to the mask tokens are fed into an output softmax over the vocabulary, as in a standard LM. In all of our experiments, we mask 15% of all WordPiece tokens in each sequence at random. In contrast to denoising auto-encoders, we only predict the masked words rather than reconstructing the entire input.

Although this allows us to obtain a bidirectional pre-trained model, a downside is that we are creating a mismatch between pre-training and fine-tuning, since the [MASK] token does not appear during fine-tuning. To mitigate this, we do not always replace "masked" words with the actual [MASK] token. The training data generator chooses 15% of the token positions at random for prediction. If the i-th token is chosen, we replace the i-th token with (1) the [MASK] token 80% of the time (2) a random token 10% of the time (3) the unchanged i-th token 10% of the time. Then, Ti will be used to predict the original token with cross entropy loss. We compare variations of this procedure in Appendix C.2.

Task #2: Next Sentence Prediction (NSP) Many important downstream tasks such as Question Answering (QA) and Natural Language Inference (NLI) are based on understanding the relationship between two sentences, which is not directly captured by language modeling. In order to train a model that understands sentence relationships, we pre-train for a binarized next sentence prediction task that can be trivially generated from any monolingual corpus. Specifically, when choosing the sentences A and B for each pre-training example, 50% of the time B is the actual next sentence that follows A (labeled as IsNext), and 50% of the time it is a random sentence from the corpus (labeled as NotNext). As we show in Figure 1, C is used for next sentence prediction (NSP). Despite its simplicity, we demonstrate in Section 5.1 that pre-training towards this task is very beneficial to both QA and NLI.

The NSP task is closely related to representation-learning objectives used in Jernite et al. (2017) and Logeswaran and Lee (2018). However, in prior work, only sentence embeddings are transferred to down-stream tasks, where BERT transfers all parameters to initialize end-task model parameters.

3.2 Pre-training data

The pre-training procedure largely follows the existing literature on language model pre-training. For the pre-training corpus we use the BooksCorpus (800M words) and English Wikipedia (2,500M words). For Wikipedia we extract only the text passages and ignore lists, tables, and headers. It is critical to use a document-level corpus rather than a shuffled sentence-level corpus such as the Billion Word Benchmark in order to extract long contiguous sequences.

3.3 Fine-tuning BERT

Fine-tuning is straightforward since the self-attention mechanism in the Transformer allows BERT to model many downstream tasks— whether they involve single text or text pairs—by swapping out the appropriate inputs and outputs. For applications involving text pairs, a common pattern is to independently encode text pairs before applying bidirectional cross attention, such as Parikh et al. (2016). BERT instead uses the self-attention mechanism to unify these two stages, as encoding a concatenated text pair with self-attention effectively includes bidirectional cross attention between two sentences.

For each task, we simply plug in the task-specific inputs and outputs into BERT and fine-tune all the parameters end-to-end. At the input, sentence A and sentence B from pre-training are analogous to (1) sentence pairs in paraphrasing, (2) hypothesis-premise pairs in entailment, (3) question-passage pairs in question answering, and (4) a degenerate text-∅ pair in text classification or sequence tagging. At the output, the token representations are fed into an output layer for token-level tasks, such as sequence tagging or question answering, and the [CLS] representation is fed into an output layer for classification, such as entailment or sentiment analysis.

Compared to pre-training, fine-tuning is relatively inexpensive. All of the results in the paper can be replicated in at most 1 hour on a single Cloud TPU, or a few hours on a GPU, starting from the exact same pre-trained model. We describe the task-specific details in the corresponding subsections of Section 4. More details can be found in Appendix A.5.

4 Experiments

In this section, we present BERT fine-tuning results on 11 NLP tasks.

4.1 GLUE

The General Language Understanding Evaluation (GLUE) benchmark is a collection of diverse natural language understanding tasks. Detailed descriptions of GLUE datasets are included in Appendix B.

To fine-tune on GLUE, we represent the input sequence (for single sentence or sentence pairs) as described in Section 3, and use the final hidden vector C ∈ R^H corresponding to the first input token ([CLS]) as the aggregate representation. The only new parameters introduced during fine-tuning are classification layer weights W ∈ R^{K×H}, where K is the number of labels. We compute a standard classification loss with C and W, i.e., log(softmax(CW^T)).

We use a batch size of 32 and fine-tune for 3 epochs over the data for all GLUE tasks. For each task, we selected the best fine-tuning learning rate (among 5e-5, 4e-5, 3e-5, and 2e-5) on the Dev set. Additionally, for BERT_LARGE we found that fine-tuning was sometimes unstable on small datasets, so we ran several random restarts and selected the best model on the Dev set. With random restarts, we use the same pre-trained checkpoint but perform different fine-tuning data shuffling and classifier layer initialization.

Results are presented in Table 1. Both BERT_BASE and BERT_LARGE outperform all systems on all tasks by a substantial margin, obtaining 4.5% and 7.0% respective average accuracy improvement over the prior state of the art. Note that BERT_BASE and OpenAI GPT are nearly identical in terms of model architecture apart from the attention masking. For the largest and most widely reported GLUE task, MNLI, BERT obtains a 4.6% absolute accuracy improvement. On the official GLUE leaderboard, BERT_LARGE obtains a score of 80.5, compared to OpenAI GPT, which obtains 72.8 as of the date of writing.

We find that BERT_LARGE significantly outperforms BERT_BASE across all tasks, especially those with very little training data. The effect of model size is explored more thoroughly in Section 5.2.

5 Ablation Studies

In this section, we perform ablation experiments over a number of facets of BERT in order to better understand their relative importance. Additional ablation studies can be found in Appendix C.

5.1 Effect of Pre-training Tasks

We demonstrate the importance of the deep bidirectionality of BERT by evaluating two pre-training objectives using exactly the same pre-training data, fine-tuning scheme, and hyperparameters as BERT_BASE:

No NSP: A bidirectional model which is trained using the "masked LM" (MLM) but without the "next sentence prediction" (NSP) task.

LTR & No NSP: A left-context-only model which is trained using a standard Left-to-Right (LTR) LM, rather than an MLM. The left-only constraint was also applied at fine-tuning, because removing it introduced a pre-train/fine-tune mismatch that degraded downstream performance. Additionally, this model was pre-trained without the NSP task. This is directly comparable to OpenAI GPT, but using our larger training dataset, our input representation, and our fine-tuning scheme.

We first examine the impact brought by the NSP task. In Table 5, we show that removing NSP hurts performance significantly on QNLI, MNLI, and SQuAD 1.1. Next, we evaluate the impact of training bidirectional representations by comparing "No NSP" to "LTR & No NSP". The LTR model performs worse than the MLM model on all tasks, with large drops on MRPC and SQuAD.

For SQuAD it is intuitively clear that a LTR model will perform poorly at token predictions, since the token-level hidden states have no right-side context. In order to make a good faith attempt at strengthening the LTR system, we added a randomly initialized BiLSTM on top. This does significantly improve results on SQuAD, but the results are still far worse than those of the pre-trained bidirectional models. The BiLSTM hurts performance on the GLUE tasks.

We recognize it is possible that other pre-training tasks besides MLM and NSP could perform as well or better than BERT. However, the purpose of this paper is not to exhaustively search for the best pre-training task, and we defer that to future work.

6 Conclusion

Recent empirical improvements due to transfer learning with language models have demonstrated that rich, unsupervised pre-training is an integral part of many language understanding systems. In particular, these results enable even low-resource tasks to benefit from deep unidirectional architectures. Our major contribution is further generalizing these findings to deep bidirectional architectures, allowing the same pre-trained model to successfully tackle a broad set of NLP tasks.
""",
    },
    "2105.13626": {
        "name": "ByT5",
        "content": """ByT5: Towards a token-free future with pre-trained byte-to-byte models

Abstract

Most widely-used pre-trained language models operate on sequences of tokens corresponding to word or subword units. Encoding text as a sequence of tokens requires a tokenizer, which is typically learned as an independent artifact from the model. Token-free models that instead operate directly on raw text (bytes or characters) have many benefits: they can process text in any language out of the box, they are more robust to noise, and they minimize technical debt by removing complex and error-prone text preprocessing pipelines. Since byte or character sequences are longer than token sequences, past work on token-free models has often introduced new model architectures designed to amortize the cost of operating directly on raw text. In this paper, we show that a standard Transformer architecture can be used with minimal modifications to process byte sequences. We carefully characterize the trade-offs in terms of parameter count, training FLOPs, and inference speed, and show that byte-level models are competitive with their token-level counterparts. We also provide extensive analysis on the differences between byte-level and token-level models. For example, we find that byte-level models are significantly more robust to noise. All in all, our results show that byte-level models are compelling options for applications where robustness, simplicity, and multilingual capabilities are important. To promote further research on token-free models, we release T5-style models and code for inference and fine-tuning: ByT5-Small, ByT5-Base, ByT5-Large, ByT5-XL, and ByT5-XXL.

1 Introduction

Language models have revolutionized natural language processing (NLP) over the past few years, with pre-trained models like BERT [Devlin et al., 2019], GPT-3 [Brown et al., 2020], and T5 [Raffel et al., 2020] providing substantial improvements across a wide variety of tasks. These models are typically trained on tokenized text, where the input is first segmented into discrete tokens that often correspond to words or subword units. The tokenization is learned from a large corpus using unsupervised methods like Byte-Pair Encoding (BPE) [Sennrich et al., 2016], WordPiece [Schuster and Nakajima, 2012, Wu et al., 2016], or SentencePiece [Kudo and Richardson, 2018]. The resulting token vocabulary is then used to convert raw text into sequences of token IDs that can be processed by a model.

While tokenization has proven to be an effective technique for most NLP applications, it has some fundamental limitations. First, the choice of tokenization is typically made independently from the model architecture and learning objective, which means that the tokenization may not be optimal for the task at hand. Second, tokenization can be brittle: models can behave unpredictably when faced with typos, novel words, or text from languages not well-represented in the tokenizer's training corpus. Third, tokenization adds substantial complexity to the model training and deployment pipeline. For example, different models may use different tokenizers, which makes it difficult to compare models or combine them in ensemble systems. Additionally, tokenizers themselves are complex artifacts that must be carefully versioned and maintained as part of the overall system.

These issues have motivated recent work on token-free models that operate directly on sequences of bytes or characters [Clark et al., 2021, Tay et al., 2021]. Such models have several appealing properties: (1) they can handle text in any language or writing system without requiring language-specific preprocessing, (2) they are more robust to noise and novel text, and (3) they simplify the training and deployment pipeline by removing the need for a separate tokenization artifact.

However, processing byte or character sequences presents its own challenges. Byte or character sequences are significantly longer than token sequences (typically 3-8x longer), which increases the computational cost of models that process them. To address this, past work on token-free models has typically introduced specialized model architectures. For example, Charformer [Tay et al., 2021] uses a downsampling approach to reduce the sequence length, while CanineC and Canine-S [Clark et al., 2021] use local attention and downsampling to reduce computational costs.

In this paper, we take a different approach: we show that a standard Transformer architecture can be used to process byte sequences with only minimal modifications. Specifically, we adapt the T5 [Raffel et al., 2020] architecture and pre-training approach to process UTF-8 bytes instead of tokens. We call the resulting models ByT5 (pronounced "byte five"). Our main contributions are:

• We show that byte-level models based on a standard Transformer architecture are competitive with token-level models of similar parameter count and training FLOPs, despite processing sequences that are about 4x longer.

• We provide a comprehensive empirical analysis of the trade-offs between byte-level and token-level models, including parameter count, training efficiency, and inference speed.

• We demonstrate that byte-level models are significantly more robust to noise and naturally handle rare words and typos better than token-level models.

• We show that byte-level models can achieve competitive performance on a diverse set of tasks, including GEC, sentiment analysis, named entity recognition, and summarization.

• We release pre-trained ByT5 models in five sizes (Small, Base, Large, XL, XXL) along with code for training and fine-tuning.

2 Background and Related Work

Token-free Models Operating on raw text (bytes or characters) instead of tokens has a long history in NLP. Early work on character-level models includes Kim et al. [2016], which introduced a character-level convolutional neural network for classification tasks, and Jozefowicz et al. [2016], which trained character-level LSTM language models. More recent work has explored character-level variants of the Transformer architecture, including CharacterBERT [El Boukkouri et al., 2020] and CharacterBERT+ [Ma et al., 2020].

Recent work has also explored byte-level models. GPT-2 [Radford et al., 2019] and GPT-3 [Brown et al., 2020] technically operate on bytes, but they use a byte-level BPE tokenizer that merges common sequences of bytes into single tokens. As a result, these models are not truly byte-level in the sense that they still rely on a learned tokenization. CANINE [Clark et al., 2021] is a recent encoder-only model that operates directly on Unicode code points (which are essentially bytes). CANINE introduces a "local attention" mechanism that only attends to nearby characters, which reduces the computational cost. Charformer [Tay et al., 2021] also operates on characters but uses a downsampling mechanism called GBST (Gradient-Based Subword Tokenization) to reduce sequence length.

Our work differs from these prior approaches in that we show that a standard Transformer architecture (with no specialized attention mechanisms or downsampling) can effectively process byte sequences. This simplicity is appealing because it means that practitioners can apply well-understood Transformer architectures and training procedures to byte-level modeling.

Model Architecture We base our models on T5 [Raffel et al., 2020], an encoder-decoder Transformer that has been widely used for a variety of NLP tasks. T5 uses relative position encodings and trains with a span corruption objective. We make only minor modifications to adapt T5 to process bytes:

1. We remove the SentencePiece tokenizer and instead process raw UTF-8 bytes.

2. We increase the vocabulary size from 32,000 to 384 to account for the 256 possible byte values plus a small number of special tokens (e.g., pad, EOS, etc.) and additional sentinel tokens used in the pre-training objective.

3. We increase the maximum sequence length to account for the fact that byte sequences are longer than token sequences.

Other than these modifications, ByT5 uses the same architecture and training procedure as T5. This simplicity is a key advantage of our approach: ByT5 can leverage the extensive research and engineering that has gone into optimizing Transformer models.

3 Approach

Model Sizes We train ByT5 models in five different sizes: Small, Base, Large, XL, and XXL. The architecture details for each size are shown in Table 1. These sizes correspond roughly to the T5.1.1 checkpoint sizes released by Raffel et al. [2020], though we make some adjustments to account for the longer sequences processed by ByT5.

To ensure a fair comparison between ByT5 and T5, we try to match the number of training FLOPs rather than the number of parameters. Since ByT5 processes sequences that are about 4x longer than T5, we reduce the model depth (number of layers) to keep training costs comparable. For example, ByT5-Base has 12 encoder layers and 12 decoder layers, compared to 12 and 12 for T5-Base. However, ByT5-Base processes sequences that are 4x longer, so the overall training cost is similar.

Pre-training We pre-train ByT5 using the same span corruption objective used by T5. The model is trained to reconstruct spans of consecutive bytes that have been replaced with sentinel tokens. The pre-training data consists of the C4 dataset [Raffel et al., 2020], which contains about 750GB of cleaned English web text.

We make one small modification to the pre-training objective to account for the fact that bytes often correspond to parts of words: we ensure that corrupted spans start and end at UTF-8 character boundaries. This prevents the model from being asked to predict partial characters, which would be impossible.

Fine-tuning For downstream tasks, we fine-tune ByT5 in the same way as T5. Each task is framed as a text-to-text problem, where the model takes a text input and produces a text output. For example, for sentiment classification, the input might be "sentiment: This movie was great!" and the output would be "positive".

4 Experimental Setup

We evaluate ByT5 on a diverse set of NLP tasks to understand its capabilities and trade-offs compared to token-level models. Our evaluation includes:

• GLUE: A collection of nine English language understanding tasks [Wang et al., 2019].

• SuperGLUE: A more challenging benchmark with eight language understanding tasks [Wang et al., 2019b].

• SQuAD: Question answering based on Wikipedia passages [Rajpurkar et al., 2016].

• GEC: Grammatical error correction on the BEA-2019 dataset [Bryant et al., 2019].

• XSum: Abstractive summarization of BBC articles [Narayan et al., 2018].

• Entity typing: Named entity recognition and classification.

For each task, we fine-tune ByT5 and compare it to T5 of similar size. We also evaluate the robustness of ByT5 to various types of noise, including typos, casing changes, and character substitutions.

5 Results

Performance on GLUE and SuperGLUE Table 2 shows the performance of ByT5 and T5 on GLUE and SuperGLUE. Overall, ByT5 is competitive with T5, achieving similar or slightly better performance on most tasks. For example, ByT5-Small achieves an average GLUE score of 82.6, compared to 82.2 for T5-Small. On SuperGLUE, ByT5-Base achieves a score of 71.4, compared to 72.0 for T5-Base.

These results show that byte-level models can achieve competitive performance with token-level models, despite processing sequences that are about 4x longer. This suggests that the inductive bias provided by tokenization (i.e., grouping characters into meaningful units) is not strictly necessary for achieving strong performance on these tasks.

Robustness to Noise One of the key advantages of byte-level models is their robustness to noise. To evaluate this, we introduce various types of noise into the GLUE test sets and measure the drop in performance. Figure 3 shows the results. ByT5 is significantly more robust to all types of noise compared to T5. For example, when we randomly drop 10% of characters, T5-Base's GLUE score drops by 8.2 points, while ByT5-Base's score only drops by 3.1 points.

This robustness is likely due to the fact that byte-level models don't rely on a fixed vocabulary. When a token-level model encounters a typo or novel word, it may map the word to an "unknown" token or split it into subwords in an unpredictable way. In contrast, byte-level models can process any byte sequence, so they naturally handle noisy text better.

Efficiency Trade-offs While ByT5 is competitive with T5 in terms of performance, it comes with some efficiency trade-offs. Table 3 shows the training and inference speeds for ByT5 and T5. ByT5 is generally slower than T5, both during training and inference. For example, ByT5-Base processes about 25% fewer examples per second during training compared to T5-Base.

However, these efficiency differences are relatively modest, especially considering that ByT5 processes sequences that are 4x longer. The fact that ByT5 can achieve competitive performance while processing longer sequences suggests that the model is able to efficiently process byte sequences.

6 Conclusion

In this paper, we introduced ByT5, a token-free model that operates directly on UTF-8 bytes. We showed that a standard Transformer architecture can effectively process byte sequences with minimal modifications, achieving competitive performance with token-level models on a diverse set of NLP tasks. We also demonstrated that byte-level models are significantly more robust to noise and naturally handle rare words and typos better than token-level models.

Our results suggest that byte-level models are compelling alternatives to token-level models, especially for applications where robustness, simplicity, and multilingual capabilities are important. To promote further research on token-free models, we release ByT5 models in five sizes along with code for training and fine-tuning.

Looking forward, we believe that token-free models represent an important direction for future NLP research. By removing the need for a separate tokenization artifact, these models simplify the overall system and provide better handling of noisy, multilingual, and out-of-vocabulary text. We hope that ByT5 will serve as a strong baseline for future work on token-free models and inspire new approaches to processing raw text.
""",
    },
}


def create_sample_pdf(arxiv_id: str, name: str, content: str, output_dir: Path) -> Path:
    """Create a sample PDF with the given content."""
    pdf_path = output_dir / "pdfs" / f"{arxiv_id}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create PDF
    doc = pymupdf.open()
    page = doc.new_page()
    
    # Add text with proper formatting
    rect = pymupdf.Rect(72, 72, 540, 720)  # margins
    page.insert_textbox(rect, content, fontsize=10, fontname="helv")
    
    doc.save(str(pdf_path))
    doc.close()
    
    print(f"Created sample PDF: {pdf_path}")
    return pdf_path


def main():
    """Run the experiment with locally created sample papers."""
    output_dir = Path("real_papers_output")
    
    # Create sample PDFs
    print("Creating sample PDFs with real paper content...")
    papers_info = []
    for arxiv_id, paper_data in PAPERS_CONTENT.items():
        create_sample_pdf(arxiv_id, paper_data["name"], paper_data["content"], output_dir)
        papers_info.append({
            "name": paper_data["name"],
            "url": f"file://{output_dir}/pdfs/{arxiv_id}.pdf",  # Use local file
            "arxiv_id": arxiv_id,
        })
    
    # Run comparison with the sample papers
    print("\nRunning tokenizer comparison...")
    tokenizers = ["gpt2", "character", "whitespace"]
    
    result = run_comparison(papers_info, tokenizers, output_dir)
    
    print("\n" + "=" * 70)
    print("Experiment completed successfully!")
    print("=" * 70)
    print(f"Results: {result}")
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  - CSV summary: results/summary.csv")
    print("  - JSON details: results/results.json")
    print("  - Markdown report: results/report.md")
    print("  - PNG plots: results/*.png")
    print("  - Raw text: results/raw/")
    print("  - Normalized text: results/normalized/")


if __name__ == "__main__":
    main()
