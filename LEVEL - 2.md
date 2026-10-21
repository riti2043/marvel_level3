## TASK 1 : Naive Bayes Classifier

Naive Bayes is a **probabilistic classifier** that uses Bayes' theorem to predict the class with the highest **posterior probability**. It assumes the features are **conditionally independent** given the class, which is rarely true for real text but makes the model very fast. The **Multinomial** variant works on word counts, and **Laplace smoothing** stops an unseen word from zeroing out a class.

![](https://github.com/riti2043/Level1images/blob/main/1task1.jpeg?raw=true)
![](https://github.com/riti2043/Level1images/blob/main/1task2.jpeg?raw=true)

### Dataset Used :
The **IMDB Movie Reviews dataset** contains **50,000 reviews** labeled **positive** or **negative**. The classes are balanced, so accuracy is a fair measure, though precision, recall and F1-score were also reported.

### Process followed:
- Loaded the dataset and checked its shape, missing values and duplicate reviews. Explored the class distribution, review lengths, the most frequent words in each class, and sample reviews.
- Removed duplicate reviews so the same review could not appear in both the training and test sets.
- Cleaned the text by lowercasing, removing HTML tags and replacing everything except letters and apostrophes with spaces. Stopwords were **not** removed, since words like *not* and *no* carry sentiment.
- Encoded the labels (positive = 1, negative = 0) and split the data into **80% training and 20% test** using **stratified sampling**.
- Converted the reviews into a **bag-of-words** count matrix with `CountVectorizer` (**10,000-word vocabulary**), fitted on the training set only to avoid data leakage.
- Implemented **Multinomial Naive Bayes from scratch** using NumPy. It computes log priors and Laplace-smoothed log likelihoods, and predicts with one matrix multiplication plus the log-sum-exp trick for stable probabilities.
- Wrote **accuracy, precision, recall, F1-score and the confusion matrix from scratch**.
- Trained scikit-learn's `MultinomialNB` on the **identical feature matrix with the same smoothing**, so any difference would come from the classifier and not the preprocessing.
- Compared the two implementations on metrics, prediction agreement, learned parameters, log probabilities, and fit and prediction time.
- Concluded that the from-scratch model matched scikit-learn exactly, with **84.1% accuracy** and **100% prediction agreement**. The tiny remaining differences are floating-point rounding, which confirms the implementation is correct. Its main limitation is the independence assumption, which ignores word order and negation.

### Implementation:
-   Understood Bayes’ theorem, the conditional independence assumption, Laplace smoothing and why the calculation is done in log space.
-   Implemented Multinomial Naive Bayes **from scratch** in NumPy, along with accuracy, precision, recall, F1-score and the confusion matrix.
-   Implemented the same model with **scikit-learn** on identical features and smoothing, and compared the two. They gave identical results.

[Link to notebook](https://github.com/riti2043/marvel_level3/blob/main/task1/naivebayes.ipynb)

![](ADD_RESULTS_IMAGE_LINK_HERE)


## TASK 2 : Ensemble Learning and Tree-Based Models

Ensemble learning combines several models so that the group predicts better than any single model. This task compares **Random Forest** (bagging), **Gradient Boosting** and **XGBoost** (boosting) on the Titanic survival dataset.

[What I learnt about Ensemble Learning](https://github.com/riti2043/Concepts_for_level2/blob/master/ensemble_learning.md)

### Dataset Used :
The Titanic survival dataset contains 891 passengers with details such as class, sex, age, fare and family members aboard. The target is whether the passenger survived or did not survive.

### Process followed:
- Loaded the dataset and checked its shape, data types, missing values (mainly *Age*, *Cabin* and *Embarked*) and survival rate. Explored survival by sex and passenger class.
- Engineered new features: **FamilySize**, **IsAlone**, **Title** (extracted from the name, with uncommon titles grouped as *Rare*) and **HasCabin**. Dropped identifier-like columns that carry no general signal.
- Split the data into **80% training and 20% test** using **stratified sampling**, leaving **179 test rows**.
- Built a preprocessing pipeline **fitted on the training set only** to avoid data leakage: median imputation for numeric columns (no scaling, since tree models do not need it) and most-frequent imputation with one-hot encoding for categorical columns.
- Trained **Random Forest, Gradient Boosting and XGBoost** with untuned baseline settings, since hyperparameter tuning is the next task.
- Evaluated each model on **accuracy, precision, recall and F1-score**, along with the train-test gap, fit time and confusion matrices.
- Checked stability with **repeated stratified cross-validation** on the training set, because a single 179-row test split is noisy.
- Compared feature importance using two methods: **impurity-based importance** and **permutation importance** (the drop in F1 when a feature is shuffled on the test set).
- Concluded that XGBoost scored the highest F1 (**0.768**) on the single split, but Gradient Boosting was the most stable in cross-validation (accuracy **0.822**) and overfit least, while Random Forest overfit most (train-test gap **0.177**). Permutation importance ranked Title and Sex highest, whereas impurity importance favored continuous features such as Fare and Age.

### Implementation:
- Understood the difference between **bagging and boosting**, and how overfitting shows up as a gap between training and test accuracy.
- Trained the three tree-based ensembles in identical pipelines, so differences come from the model and not the preprocessing.
- Interpreted feature importance with two methods and compared where they agree and disagree.

[Link to notebook](https://github.com/riti2043/marvel_level3/blob/main/task2/ensemble.ipynb)

![](https://github.com/riti2043/Level1images/blob/main/task2.jpeg?raw=true)

## TASK 3 : Hyperparameter Tuning

Hyperparameter tuning finds the best external configuration for a model by searching the parameter space using cross-validation, without ever touching the test set. Three search strategies were compared — **Grid Search**, **Randomized Search**, and **Bayesian Optimization (Optuna)** — across three model families to understand their trade-offs in coverage, speed, and sample efficiency.

[What I learnt about Hyperparameter Tuning](https://github.com/riti2043/Concepts_for_level2/blob/master/ensemble_learning.md)

### Dataset Used :
The **Titanic survival dataset** (891 passengers) with a ~38% survival rate. **F1-score** was the primary optimization metric due to mild class imbalance.

### Process followed:
- Engineered features from Task 2 (*FamilySize*, *IsAlone*, *Title*, *HasCabin*) and applied an 80/20 stratified split (712 train, 179 test rows).
- Built a shared preprocessing pipeline fitted on training data only — median imputation + StandardScaler for numerics; mode imputation + one-hot encoding for categoricals. Scaling is mandatory for SVM and harmless for tree models, so one pipeline served all three.
- Evaluated unconstrained **baseline models** for **Random Forest**, **XGBoost** and **SVM** using Repeated Stratified K-Fold (5 folds × 2 repeats) CV.
- Applied all three search strategies to each model and tracked CV F1, standard deviation, test F1, Train-CV gap, candidates evaluated and runtime.
- Diagnosed overfitting via the Train-CV gap: default Random Forest had Train F1 = **0.9834** vs CV F1 = **0.7562** (gap **0.2272**). Constraining `max_depth=3` and `min_samples_leaf` collapsed the gap to **0.0549**.
- Concluded that **Random Forest via Grid Search** was the overall winner with a **CV F1 of 0.7835 (±0.0223)** and held-out **Test F1 of 0.7552**. Randomized Search nearly matched it (**0.7792 CV F1**) at a quarter of the compute cost (7.24s vs 28.68s). Optuna was the most sample-efficient, converging in under **5s** for Random Forest and under **1s** for XGBoost.

### Implementation:
- Understood the difference between learned parameters and user-defined hyperparameters, and how restricting complexity (depth, regularization, subsampling) controls the bias-variance tradeoff.
- Built `Pipeline` + `ColumnTransformer` architectures that encapsulate preprocessing and model in one object, ensuring zero leakage during cross-validation.
- Implemented **Grid Search** over a handcrafted discrete grid, understanding how boundary-hitting results signal the need to extend the search space.
- Implemented **Randomized Search** with `scipy.stats.randint` distributions, learning how sampling from continuous ranges covers more ground per candidate than a fixed grid.
- Implemented **Bayesian Optimization with Optuna** using the TPE sampler, understanding how a probabilistic surrogate model directs the search toward promising regions after each trial.
- Ran all 12 model-method combinations (3 models × 4 methods) in identical pipelines so differences reflect the search strategy, not the preprocessing.
- Learnt that selecting the final model strictly by CV score — not test score — is critical to maintain an unbiased final evaluation, especially on a small 179-row test set where 1–2 wrong predictions shift F1 by ~1%.

[Link to notebook](ADD_LINK_HERE)

![](ADD_RESULTS_IMAGE_LINK_HERE)

## TASK 4 : Image Clustering using K-Means

**Clustering** is an unsupervised learning technique that groups data points by similarity, **without using labels**. **K-Means** splits the data into K groups, each represented by its **centroid** (the mean of its points). Every point is assigned to its nearest centroid by **Euclidean distance**, and the algorithm minimizes the **within-cluster variance (inertia)**. It always stops at a **local minimum**, so the starting centroids matter, which is why **K-Means++** initialization is used.

![](ADD_MATH_IMAGE_LINK_HERE)

### Dataset Used :
The **MNIST handwritten digits dataset**, with **70,000 images** (the standard 60,000 training and 10,000 test images combined) of the digits 0 to 9. Each image is 28 × 28 pixels, flattened into **784 features**. Clustering needs no train/test split, so the **true labels were used only for evaluation**, never for fitting.

### Process followed:
- Loaded the four raw MNIST (IDX) files, combined the training and test images into **70,000 images**, and checked the shapes, pixel range and missing values.
- Explored the data through the class distribution, sample images of each digit, and the **mean image per digit** (what a perfect centroid would look like).
- Scaled the pixel values from 0-255 to **0-1**, and studied the concepts behind image clustering: flattening, the curse of dimensionality, PCA, t-SNE, and the methods for choosing K.
- Implemented **K-Means++ from scratch** with NumPy, computing squared distances through one matrix multiplication so that the full 70,000 × 784 data runs quickly.
- Trained scikit-learn's `KMeans` with the same settings (**K = 10, K-Means++, `n_init=3`**) and compared the two on inertia, iterations, fit time, and the ARI between their clusterings.
- Evaluated the clusters against the true digits using **majority-vote accuracy (purity)**, **Hungarian accuracy** (best one-to-one matching), **ARI** and **NMI**, with a contingency heatmap, a confusion matrix, per-digit recall and the most frequent confusions.
- Visualized the clusters with **centroid images**, a cluster summary table, random images from every cluster, and **PCA and t-SNE 2-D plots** colored by cluster and by true digit.
- Concluded that the from-scratch model matched scikit-learn (inertia within **0.017%**). The clusters reached **54 to 58% accuracy**, far above random guessing (10%) but well below supervised classifiers. Digits 0, 2 and 6 formed clean clusters, while digit 1 split in two and digit 5 got none, because pixel-by-pixel distance treats shifted or thickened copies of a digit as far apart.

### Implementation:
- Understood clustering, centroids, Euclidean distance, inertia, K-Means++ and why K-Means only reaches a local minimum.
- Implemented K-Means++ **from scratch** in NumPy, along with scikit-learn's `KMeans` on MNIST, and compared the two.
- Evaluated the clusters against the true labels and visualized them with centroids, PCA and t-SNE.

[Link to notebook](ADD_LINK_HERE)

![](ADD_RESULTS_IMAGE_LINK_HERE)

## TASK 5 : Introduction to PyTorch

PyTorch is an open-source deep learning framework developed by Meta AI. It is built around three core components: **Tensors** (multi-dimensional arrays that run on GPU), **Autograd** (an automatic differentiation engine that computes gradients via backpropagation), and **torch.nn / torch.optim** (pre-built layers, loss functions and optimizers). Its dynamic computation graphs make it easy to write and debug models.

### What I covered:
- **Tensors** — creating, indexing, slicing, reshaping and operating on multi-dimensional arrays. Understood shape, dtype, device and the difference between element-wise and matrix operations.
- **Broadcasting and NumPy interoperability** — how PyTorch stretches smaller tensors to match shapes, and how CPU tensors share memory with NumPy arrays.
- **GPU acceleration** — moved tensors with `.to(device)` and benchmarked matrix multiplication across matrix sizes, observing significant speed-ups on a Tesla T4 GPU.
- **Autograd** — understood how PyTorch builds a computational graph and calls `.backward()` to apply the chain rule. Verified autograd derivatives against hand-computed values.
- **Linear Regression in PyTorch** — trained a model on synthetic data using `nn.Linear`, `MSELoss` and SGD. The learned weight and bias closely matched the true values, confirming correct training.

### Implementation:
- Understood tensors as the core data structure and practiced all key operations from creation to reduction.
- Implemented gradient descent manually before switching to `torch.optim`, to understand what the optimizer does under the hood.
- Verified autograd gradients against hand-computed derivatives to confirm the engine is working correctly.
- Understood why gradients must be zeroed before each backward pass with `optimizer.zero_grad()`.
- Understood GPU asynchrony and why `torch.cuda.synchronize()` is needed for accurate timing.

[Link to notebook](ADD_LINK_HERE)

![](ADD_IMAGE_LINK_HERE)
## TASK 6 : Deep Learning Fundamentals

Deep neural networks learn by repeatedly cycling through four steps: forward pass, loss computation, backpropagation and weight update. Four design choices govern how well this cycle works — **activation functions**, **loss functions**, **optimizers**, and **regularization** — and each was isolated in a controlled experiment on Fashion-MNIST.

[What I learnt about Deep Learning Fundamentals](ADD_ARTICLE_LINK_HERE)

### Dataset Used :
**Fashion-MNIST** contains **70,000 grayscale images** of 10 clothing categories — 60,000 for training and 10,000 for testing. It is a harder drop-in replacement for MNIST, making differences between design choices more visible. Pixels were scaled and standardized using training-set statistics.

### Process followed:
- Studied four **activation functions** — Sigmoid, Tanh, ReLU and Leaky ReLU — understanding their saturation behavior, gradient strength and zero-centering properties.
- Studied three **loss functions** — MSE, Binary Cross Entropy and Categorical Cross Entropy — and verified CCE manually against `nn.CrossEntropyLoss`.
- Studied five **optimizers** — Gradient Descent, SGD, SGD with Momentum, RMSProp and Adam — and compared their convergence behavior.
- Built a reusable `FeedforwardNet` class that supports swappable activations, dropout and batch normalization, then ran five controlled experiments changing one factor at a time:
  - **Experiment A (Activations):** ReLU and Leaky ReLU converged fastest; Sigmoid was slowest due to vanishing gradients.
  - **Experiment B (Optimizers):** Full-batch Gradient Descent stalled significantly; Adam and SGD with Momentum reached higher accuracy in the same number of epochs.
  - **Experiment C (Dropout):** Without dropout, the model heavily overfit a small training subset — train accuracy climbed while validation accuracy stagnated. Dropout progressively narrowed this gap.
  - **Experiment D (MSE vs BCE):** BCE converged faster for binary classification, consistent with it penalizing confident wrong predictions more severely than MSE.
  - **Experiment E (Batch Norm + Early Stopping):** Batch Norm improved early-epoch convergence. Early stopping halted training before validation loss climbed and restored the best weights automatically.

### Implementation:
- Understood the vanishing gradient problem and why ReLU is the default choice for hidden layers.
- Implemented and compared all five optimizers, understanding momentum as a running gradient average and Adam as momentum combined with adaptive per-parameter scaling.
- Built Dropout and Batch Normalization into the same model class and understood why both behave differently in `model.train()` vs `model.eval()` mode.
- Designed experiments so only one variable changes per run, making the cause of each difference unambiguous.

[Link to notebook](ADD_LINK_HERE)

![](ADD_IMAGE_LINK_HERE)
## TASK 7 : CNN from Scratch and Transfer Learning

[INTRO - to be discussed]

### Dataset Used :
The **CIFAR-10 dataset** contains **60,000 colour images** of size **32×32 pixels** in **10 balanced classes** (airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck). It is split into **50,000 training** and **10,000 test** images. Since the classes are balanced, accuracy is a fair measure, and precision, recall, F1-score and a confusion matrix were also used to see which classes get confused.

### Process followed:
- Loaded the dataset, checked its shape and class balance, and visualised a grid of sample images from each class.
- Held out **5,000 of the training images as a validation set**, giving a **45,000 / 5,000 / 10,000** train, validation and test split, with the test set used only for final evaluation.
- **Part A, CNN from scratch:** Normalised the images using the per-channel mean and standard deviation of the training set and applied **data augmentation** (random crop with 4-pixel padding and random horizontal flip) to the training data only.
- Built a CNN in PyTorch with **three convolution blocks** (Conv → ReLU → MaxPool, with **32, 64 and 128 filters**, 3×3 kernels), followed by flattening, a **256-unit dense layer with 0.5 dropout** and a **10-way output layer**, giving roughly **0.6 million parameters**.
- Trained it with the **Adam** optimizer (learning rate **0.001**), batch size **128**, **20 epochs** and cross-entropy loss, recording training and validation loss and accuracy every epoch.
- **Part B, transfer learning:** Loaded **ResNet18 pretrained on ImageNet**, resized the images to **224×224** and normalised them with the ImageNet mean and standard deviation, as the pretrained network expects.
- **Froze all feature-extraction layers**, replaced the final layer with a new **10-class classifier**, and trained only that classifier (about **5,000 trainable parameters** out of roughly 11 million) with Adam (learning rate **0.001**), batch size **128** and **5 epochs**.
- Evaluated both models on the test set using **accuracy, precision, recall and F1-score**, and examined **confusion matrices** and **per-class accuracy**.
- Compared the two approaches on test performance, training time, number of trainable parameters, how quickly each converged and the **train-validation gap**.

### Implementation:
-   Understood how **convolution** extracts local patterns into **feature maps**, how **pooling** reduces spatial size while keeping the strongest responses, and how stacked layers build up **representations** from edges to textures to object parts.
-   Understood **transfer learning**: a network pretrained on a large dataset already provides general-purpose features, so only a small classifier on top needs training.
-   Implemented a three-block **CNN from scratch** in PyTorch with data augmentation and dropout to reduce overfitting.
-   Adapted a **pretrained ResNet18** by freezing its feature extractor and training a new classifier.
-   Compared the from-scratch and transfer-learning approaches on accuracy, cost and training behaviour.

[Link to notebook](ADD_LINK_HERE)

![](ADD_RESULTS_IMAGE_LINK_HERE)

## TASK 9 : Transformer-Based NLP

[INTRO - to be discussed]

### Dataset Used :
The same **IMDB Movie Reviews dataset** from Task 8 was used, with the **same 80/10/10 train, validation and test splits**, so that DistilBERT and the LSTM are evaluated on exactly the same data. The reviews are **positive** or **negative**, and accuracy, precision, recall and F1-score were reported.

### Process followed:
- Loaded the data and reused the splits from Task 8 (same random seed of **42**) so results can be compared directly.
- Kept the review text largely unprocessed, since the pretrained tokenizer is designed to handle raw text. Only HTML tags were removed.
- Tokenized the reviews with the **`distilbert-base-uncased` tokenizer** (WordPiece subwords), adding the special tokens and **truncating or padding** to a maximum length of **256 tokens**, and created attention masks so that padding is ignored.
- Loaded the pretrained **DistilBERT** model with a **classification head** on top.
- **Fine-tuned** the whole model on the training set using the **AdamW** optimizer with a small learning rate of **2e-5**, a batch size of **16**, **3 epochs** and a linear learning-rate schedule with warm-up, on a GPU, while monitoring **training and validation loss and accuracy** after each epoch.
- Evaluated the fine-tuned model on the test set using **accuracy, precision, recall and F1-score**, and examined the **confusion matrix**.
- Compared DistilBERT with the Task 8 **LSTM** on performance, training time, number of parameters and handling of **long reviews**, and looked at examples where the two models disagreed.

### Implementation:
-   Understood the main components of a transformer: **tokenization**, **embeddings**, **positional encoding** (which restores word order, since attention alone has none) and **self-attention**, which lets every token look at every other token directly instead of passing information step by step.
-   Understood **transfer learning and fine-tuning**: DistilBERT is a smaller, faster model distilled from BERT, pretrained on large amounts of text and adapted here to sentiment classification.
-   Fine-tuned DistilBERT with **PyTorch and Hugging Face Transformers** on the same data used for the LSTM.
-   Compared the fine-tuned transformer with the LSTM on performance, cost and long-range context handling.

[Link to notebook](ADD_LINK_HERE)

![](ADD_RESULTS_IMAGE_LINK_HERE)

## TASK 11 : PDF Question Answering using RAG

Large Language Models (LLMs) are trained on fixed datasets - they have no knowledge of documents
outside their training data. When asked about such documents, they either **fail to answer** or 
**hallucinate**, generating confident but factually incorrect responses.

**Retrieval-Augmented Generation (RAG)** addresses this by combining information retrieval with 
language generation. Instead of relying on the model's memory, RAG first retrieves the most 
relevant passages from an external knowledge source, then passes them to the LLM as grounded 
context. The model generates its answer strictly from what was retrieved, making responses 
**accurate, traceable and document-specific.**

![](https://github.com/riti2043/Level1images/blob/main/How-rag-pipeline-works-1024x614.webp?raw=true)

RAG systems generally fall into three types:

- **Naive RAG:** The simplest form — chunk the document, embed the chunks, retrieve the top-K 
  most similar ones at query time, and pass them directly to the LLM. Fast to build but limited 
  by retrieval quality.

- **Advanced RAG:** Adds pre-retrieval steps (query rewriting, hypothetical document embeddings) 
  and post-retrieval steps (re-ranking, context compression) to improve the relevance of what 
  gets passed to the model.

- **Modular RAG:** A flexible pipeline where each component (retriever, reranker, generator) can 
  be swapped or extended independently. Suited for production systems with complex requirements.

This project implements a **Naive RAG pipeline** with conversational memory, applied to 
PDF question answering.

### What I built :

- **PDF Parsing:** Documents are loaded and parsed page by page using pypdf.
- **Chunking:** Text is split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter` (1000 chars, 150 overlap).
- **Embeddings:** Each chunk is converted into a dense vector using Google's `text-embedding-004` model.
- **Vector Store:** Chunk embeddings are indexed in a FAISS in-memory store for fast similarity search.
- **Query Retrieval:** At query time, the user's question is embedded and the top 4 most similar chunks are retrieved via cosine similarity.
- **Grounded Generation:** Retrieved chunks are injected into a system prompt that constrains Gemini 1.5 Flash to answer only from the provided context.
- **Conversational Memory:** The last 4 chat turns are appended to every prompt, enabling coherent follow-up questions.
- **Persistent History:** Conversation threads and messages are stored in Supabase (PostgreSQL), with a local session-state fallback.
- **Streamlit Interface:** A multi-thread sidebar for switching between past conversations, with token-by-token streamed responses and page-number citations.

### Tech Stack :

| Tool | Role |
|:---|:---|
| **pypdf** | PDF parsing and page extraction |
| **LangChain** | Text chunking, embeddings interface and vector store wrapper |
| **FAISS** | In-memory vector indexing and similarity retrieval |
| **Google Gemini** | `text-embedding-004` for embeddings, `gemini-1.5-flash` for generation |
| **Supabase** | Persistent storage for conversation threads and messages |
| **Streamlit** | Frontend UI, sidebar navigation and response streaming |

### System Architecture :
![](https://github.com/riti2043/Level1images/blob/main/rag_architecture.png?raw=true)

[GitHub Repo](https://github.com/riti2043/marvel_level3/tree/main/task_11) | [Live App](https://ragsimplechatbot.streamlit.app/)

![](ADD_IMAGE_LINK_HERE)

### Implementation:
- Understood how semantic search differs from keyword search — embeddings capture meaning, not exact words, so a question phrased differently still retrieves the right passage.
- Understood why chunking strategy and overlap matter — too large loses precision in retrieval; too small loses context needed for a coherent answer.
- Built the RAG prompt to be explicitly grounded — the LLM is constrained to the retrieved context, making hallucination structurally harder.
- Understood why responses are streamed token by token — it gives immediate feedback rather than a long wait, which is critical for UX with longer documents.
- Designed the database layer with a local fallback so the app works even without a Supabase connection.


