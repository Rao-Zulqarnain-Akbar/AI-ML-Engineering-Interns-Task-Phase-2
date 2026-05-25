# AI-ML-Engineering-Interns-Task-Phase-2
Here is my work (codes,snippet anf files)

# Task 2
Developed an end-to-end machine learning pipeline for telecom customer churn prediction.

1-The project used the Telco Customer Churn dataset containing customer demographics, billing, and service information.

2-Data preprocessing steps included handling missing values, feature encoding, scaling, and cleaning incorrect data types.

3-Logistic Regression and Random Forest models were trained and optimized using GridSearchCV with cross-validation techniques.

4-Model performance was evaluated using Accuracy, Precision, Recall, F1-score, and ROC-AUC metrics.

5-Random Forest achieved better predictive performance due to its ability to capture complex feature relationships.

6- The task demonstrated a production-ready ML workflow including preprocessing pipelines, model evaluation,
and deployment-ready model saving using joblib.


# Task 1
Developed a BERT-based News Topic Classification system using the AG News dataset for multi-class text classification.

1-Fine-tuned BERT (bert-base-uncased) to classify news headlines into four categories: World, Sports, Business, and Sci/Tech.

2-The methodology included tokenization using BertTokenizerFast, dynamic padding, model fine-tuning, early stopping, and mixed precision training for improved efficiency.

3-The AG News dataset containing approximately 120k training samples and 7.6k testing samples was used for training and evaluation.

4-Model performance was evaluated using Accuracy, Precision, Recall, F1-score, confusion matrix analysis, and confidence distribution visualizations.

5-The fine-tuned model achieved strong performance with approximately 94% Accuracy and Macro F1-score, showing balanced classification across all categories.

6- The task demonstrated a complete NLP deep learning pipeline including training, evaluation, prediction, visualization, and deployment through a Gradio web application.


# Task 3
1- Multimodal ML pipeline successfully built combining CNN image features and tabular data via a late-fusion neural network.

2- GBM baseline (tabular-only) outperformed the Multimodal CNN with MAE ~$169K and R² = 0.48, versus CNN's MAE ~$598K on the demo.

3- Performance gap is expected and explainable — only 800 synthetic images and 12 epochs were used; CNNs require far more real data to generalise.

4- Square footage was the strongest price predictor (r = 0.583), followed by bathrooms (0.478) and bedrooms (0.349).

5- Synthetic images were a key limitation — generated from price labels, they offered no genuine visual signal for the CNN to learn from.

6- The architectural objective was fully met — a working two-branch fusion model processes both modalities end-to-end with 509,761 trainable parameters.

7- With real house photos and the full 15,474-sample dataset, the multimodal model is expected to outperform tabular-only approaches by capturing visual cues like property condition, style, and exterior quality.

8- Log-price transformation, Huber loss, and cosine LR scheduling were effective design choices that stabilised training across the wide price range ($195K–$2M.)



















