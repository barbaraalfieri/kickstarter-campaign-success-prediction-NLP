# Predicting Kickstarter Campaign Success with NLP

Can the language used in a crowdfunding campaign help predict whether it will succeed?

This project investigates the relationship between campaign descriptions and funding outcomes using a dataset of 7,354 Kickstarter campaigns across five categories: Technology, Games, Music, Publishing, and Film & Video. It combines natural language processing, topic modelling, supervised machine learning, explainability methods, and an exploratory generative AI experiment.

The project covers:

- Collection of campaign descriptions through web scraping
- Data cleaning and preprocessing of long-form textual content
- Exploratory data analysis and category-level bigram comparisons
- LDA topic modelling with coherence-based topic selection and interactive pyLDAvis visualisations
- Success/failure classification using TF-IDF features and multiple machine learning models
- Model evaluation through stratified five-fold cross-validation
- Interpretation of model predictions using coefficients and SHAP values
- Comparison of baseline and RAG-assisted campaign rewriting using Mistral-7B-Instruct

## Data

The analysis is based on the Kickstarter dataset published by [Web Robots](https://webrobots.io/kickstarter-datasets/), which consists of 85 CSV files containing campaign metadata. Since the full raw dataset is too large to include in this repository, download instructions are provided in the `General/` directory.

Campaign descriptions were collected from the corresponding Kickstarter pages through web scraping. After filtering, cleaning, and preprocessing, the final dataset contains 7,354 campaigns across five categories: Technology, Games, Music, Publishing, and Film & Video.

To reproduce the data preparation process:

1. Download and extract the Kickstarter dataset following the instructions in `General/`.
2. Place the extracted `Kickstarter_2026-03-12T03_20_26_556Z/` directory in the repository root.
3. Run the notebooks in the order specified in the repository structure below.

## Classification Results

Four supervised learning models were evaluated separately for each campaign category: Logistic Regression, Bagging, XGBoost, and Linear SVC. Campaign descriptions were represented using 
TF-IDF features, and model performance was estimated through stratified five-fold cross-validation.

The results indicate that campaign language contains useful predictive information, although performance varies across categories. Linear SVC achieved the best weighted F1 score in four of the five categories, while Logistic Regression performed best for Music. The strongest overall performance was obtained for Games, with a weighted F1 score of 0.845 and a ROC-AUC of 0.913.

These findings identify predictive associations and should not be interpreted as evidence that particular wording causes a campaign to succeed.

| Category | Best model | Weighted F1 | ROC-AUC |
|---|---|---:|---:|
| Technology | Linear SVC | 0.770 | 0.846 |
| Games | Linear SVC | **0.845** | **0.913** |
| Music | Logistic Regression | 0.803 | 0.834 |
| Publishing | Linear SVC | 0.774 | 0.851 |
| Film & Video | Linear SVC | 0.779 | 0.859 |

## Generative AI Experiment

An additional experiment investigated whether Retrieval-Augmented Generation (RAG) could improve the rewriting of unsuccessful campaign descriptions. A 4-bit quantized version of Mistral-7B-Instruct-v0.3 was evaluated under two settings:

1. **Baseline:** the model rewrites the original description using only an instruction prompt.
2. **RAG-assisted:** the prompt also includes the most semantically similar successful campaign from the same category, retrieved using Sentence Transformer embeddings.

The two approaches were compared on a sample of 100 unsuccessful campaigns, with 20 campaigns drawn from each category. Rewritten descriptions were evaluated using the predicted success probabilities produced by a fine-tuned RoBERTa classifier.

Under this evaluation framework, the RAG-assisted approach did not consistently outperform the baseline. The two methods produced similar mean predicted success probabilities, suggesting that retrieving a single semantically related campaign was not sufficient to generate systematically stronger rewrites.

## Repository Structure

```text
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_exploratory_analysis.ipynb
│   ├── 04_bigram_analysis.ipynb
│   ├── 05_topic_modelling.ipynb
│   ├── 06_classification.ipynb
│   └── 07_generative_ai_experiment.ipynb
├── data/
├── outputs/
├── General/
├── requirements.txt
└── README.mdes/
│   └── results/
└── LICENSE
```

## Technologies

- **Data processing and visualisation:** pandas, NumPy, Matplotlib
- **Natural language processing:** NLTK, Gensim, Sentence Transformers
- **Machine learning and explainability:** scikit-learn, XGBoost, SHAP
- **Topic modelling:** LDA, pyLDAvis
- **Generative AI:** PyTorch, Hugging Face Transformers, LangChain, bitsandbytes
- **Data collection:** Selenium, Beautiful Soup
