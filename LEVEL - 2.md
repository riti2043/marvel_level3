
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

Ensemble learning combines several models so that the group predicts better than any single model. **Bagging** (used by **Random Forest**) trains many deep trees independently on bootstrap samples with random feature subsets and averages their votes, which mainly reduces **variance**. **Boosting** (used by **Gradient Boosting** and **XGBoost**) builds shallow trees one after another, each correcting the errors of the previous ones and scaled by a **learning rate**, which mainly reduces **bias**. XGBoost extends gradient boosting with built-in **L1/L2 regularization**, native handling of missing values and speed optimizations.

### Dataset Used :
The **Titanic survival dataset** contains **891 passengers** with details such as class, sex, age, family members aboard, fare, cabin and port of embarkation. The target is whether the passenger **survived**. About 38% survived, so the classes are mildly imbalanced and precision, recall and F1-score were reported alongside accuracy, with **F1-score** used to rank the models.

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

[What I learnt about Hyperparameter Tuning](ADD_ARTICLE_LINK_HERE)

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
## Task 7 : CNN from Scratch and Transfer Learning

## Task 8 : Sequence Modeling using RNN and LSTM

## Task 9 : Transformer-Based NLP

## TASK 11 : PDF Question Answering using RAG

Retrieval-Augmented Generation (RAG) solves a core limitation of LLMs — they cannot answer questions about documents they were never trained on, and they hallucinate when they try. RAG fixes this by **retrieving relevant passages from your document first**, then handing them to the LLM as grounded context, so the answer comes from the document rather than the model's memory.

[What I learnt about RAG](ADD_ARTICLE_LINK_HERE) | [Live App](ADD_STREAMLIT_LINK_HERE)

### What I built :
- **RAG pipeline** — PDF is parsed page by page, split into overlapping chunks, embedded using Gemini `text-embedding-004`, and stored in a FAISS in-memory vector index. At query time, the question is embedded and the top-K most similar chunks are retrieved via cosine similarity.
- **Grounded generation** — Retrieved chunks are injected into a system prompt that instructs Gemini 1.5 Flash to answer strictly from the provided context, explicitly stating when the document does not contain the answer.
- **Conversational memory** — The last 4 chat turns are included in the prompt, allowing follow-up questions without losing context.
- **Persistent chat history** — Conversation threads and messages are stored in **Supabase** (PostgreSQL), with a local session-state fallback when the database is unavailable.
- **Streamlit UI** — Multi-thread sidebar for switching between conversations, token-by-token streamed responses, and page-number citations shown alongside each answer.

### Tech Stack :
- **LangChain** — document loading, chunking, embeddings and vector store interface
- **FAISS** — in-memory vector similarity search
- **Google Gemini** — `text-embedding-004` for embeddings, `gemini-1.5-flash` for generation
- **Supabase** — persistent thread and message storage
- **Streamlit** — frontend and deployment

### Implementation:
- Understood how semantic search differs from keyword search — embeddings capture meaning, not exact words, so a question phrased differently still retrieves the right passage.
- Understood why chunking strategy and overlap matter — too large loses precision in retrieval; too small loses context needed for a coherent answer.
- Built the RAG prompt to be explicitly grounded — the LLM is constrained to the retrieved context, making hallucination structurally harder.
- Understood why responses are streamed token by token — it gives immediate feedback rather than a long wait, which is critical for UX with longer documents.
- Designed the database layer with a local fallback so the app works even without a Supabase connection.

[GitHub Repo](ADD_LINK_HERE) | [Live App](ADD_STREAMLIT_LINK_HERE)

![](ADD_IMAGE_LINK_HERE)
