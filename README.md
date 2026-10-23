#  SpamGuard AI
### Spam Detection with Explainable AI

SpamGuard AI is a machine-learning application that classifies text messages as **Spam** or **Not Spam**. It also provides word-level explanations to help users understand which words or phrases influence the model's prediction.

##  Features

- Spam and legitimate message classification
- Three machine-learning models compared during training
- TF-IDF text feature extraction
- Explainable AI using learned model coefficients or feature probabilities
- Interactive web interface built with Streamlit
- Model evaluation using accuracy, precision, recall, and F1-score

##  Machine Learning Models

The project compares three classifiers:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear Support Vector Machine (Linear SVM)

The model with the highest spam F1-score is selected and saved for use in the application.

## 📊 Model Evaluation

Results from the current held-out test set:

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1-score |
|---|---:|---:|---:|---:|
| Linear SVM | 98.12% | 92.84% | 91.94% | 92.88% |
| Logistic Regression | 97.85% | 90.85% | 93.29% | 92.05% |
| Naive Bayes | 96.23% | 100.00% | 71.81% | 83.59% |

**Best model:** Linear SVM, based on the highest spam F1-score in this evaluation.

These metrics describe the current test set and may differ on unseen messages.

##  Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit
- TF-IDF vectorization
- UCI SMS Spam Collection dataset

##  Project Structure

```text
SpamGuardAI/
├── app.py
├── train_model.py
├── data/
│   ├── SMSSpamCollection
│   └── model_results.csv
├── models/
│   └── spam_model.joblib
└── README.md
```

##  Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd SpamGuardAI
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository URL.

### 2. Create and activate a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install streamlit scikit-learn pandas joblib
```

### 4. Train the model

```bash
python train_model.py
```

The training script downloads the dataset if it is missing, evaluates the classifiers, saves the model, and generates the comparison CSV.

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal.

##  Explainable AI

The application identifies words or phrases that lean toward a spam or legitimate classification using information from the trained classifier.

The explanation method depends on the selected model. These indicators help interpret the prediction but do not prove that a particular word caused it.

##  Future Improvements

- Evaluate performance on additional, unseen datasets
- Improve handling of phishing links and suspicious URLs
- Add confidence estimates where supported by the model
- Create a more comprehensive evaluation dashboard
- Deploy the application online

## ⚠️ Limitations

SpamGuard AI is an educational machine-learning project. Its predictions are not guarantees of message safety. Legitimate messages may be flagged as spam, and spam messages may be missed.


