
from pathlib import Path

import joblib
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="🛡️",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "spam_model.joblib"


# --------------------------------------------------
# LOAD SAVED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("🛡️ SpamGuard AI")
st.header("Spam Detection with Explainable AI")

st.write(
    "Enter a message below to check whether it is "
    "likely to be spam or legitimate."
)

if not MODEL_PATH.exists():
    st.error("Trained model not found. Check the models folder.")
    st.stop()

model_data = load_model()

if not isinstance(model_data, dict) or "pipeline" not in model_data:
    st.error("The saved model does not contain the expected pipeline.")
    st.stop()

model = model_data["pipeline"]
model_name = model_data.get("model_name", "Unknown model")
dataset_name = model_data.get("dataset", "Unknown dataset")

st.caption(f"Model: {model_name} | Dataset: {dataset_name}")


# --------------------------------------------------
# MESSAGE INPUT
# --------------------------------------------------

message = st.text_area(
    "Enter your message",
    placeholder="Example: Congratulations! You have won a free prize...",
    height=160,
)

analyze = st.button("Analyze Message", type="primary")


# --------------------------------------------------
# EXPLAINABLE AI
# --------------------------------------------------

def explain_prediction(pipeline, message):
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    transformed = vectorizer.transform([message])
    feature_names = vectorizer.get_feature_names_out()
    classifier_classes = list(classifier.classes_)

    if "spam" not in classifier_classes or "ham" not in classifier_classes:
        st.info("This model does not expose the expected spam/ham classes.")
        return

    spam_index = classifier_classes.index("spam")
    ham_index = classifier_classes.index("ham")

    if transformed.nnz == 0:
        st.info(
            "The model did not recognize any words in its vocabulary "
            "from this message, so word-level explanations are unavailable."
        )
        return

    # Calculate a score for each word or phrase using the trained classifier.
    if isinstance(classifier, LogisticRegression):
        weights = classifier.coef_[0]
        if classifier_classes[1] != "spam":
            weights = -weights
        method = "Logistic Regression learned coefficients"

    elif isinstance(classifier, LinearSVC):
        weights = classifier.coef_[0]
        if classifier_classes[1] != "spam":
            weights = -weights
        method = "Linear SVM learned coefficients"

    elif isinstance(classifier, MultinomialNB):
        weights = (
            classifier.feature_log_prob_[spam_index]
            - classifier.feature_log_prob_[ham_index]
        )
        method = "Naive Bayes feature log-probability differences"

    else:
        st.info("Word-level explanations are not supported for this classifier.")
        return

    contributions = []

    for index in transformed.indices:
        value = float(transformed[0, index] * weights[index])

        contributions.append({
            "word": feature_names[index],
            "score": value,
        })

    contributions.sort(
        key=lambda item: abs(item["score"]),
        reverse=True,
    )

    spam_words = [
        item for item in contributions if item["score"] > 0
    ][:5]

    ham_words = [
        item for item in contributions if item["score"] < 0
    ][:5]

    st.caption(f"Explanation method: {method}.")
    st.write(
        "Positive scores lean toward spam; negative scores lean "
        "toward a legitimate message. These are model-based indicators, "
        "not proof that a word caused the prediction."
    )

    if spam_words:
        st.markdown("**Words leaning toward spam**")
        for item in spam_words:
            st.write(f"- `{item['word']}` — score: {item['score']:+.4f}")

    if ham_words:
        st.markdown("**Words leaning toward legitimate**")
        for item in ham_words:
            st.write(f"- `{item['word']}` — score: {item['score']:+.4f}")

    if not spam_words and not ham_words:
        st.info("No directional word indicators were found.")


# --------------------------------------------------
# PREDICTION RESULTS
# --------------------------------------------------

if analyze:
    if not message.strip():
        st.warning("Please enter a message first.")

    else:
        try:
            prediction = str(model.predict([message])[0]).strip().lower()

            st.divider()
            st.subheader("Analysis Result")

            if prediction == "spam":
                st.error("🚨 SPAM DETECTED")
                st.write("This message has been classified as spam.")

            elif prediction == "ham":
                st.success("✅ NOT SPAM")
                st.write("This message has been classified as legitimate.")

            else:
                st.warning(f"Unexpected model prediction: {prediction}")

            st.subheader("🤖 Why this prediction?")
            explain_prediction(model, message)

            st.divider()
            st.caption(
                "This is a machine-learning prediction, not a guarantee "
                "that the message is safe."
            )

        except Exception as error:
            st.error("Prediction failed. Check the Terminal for details.")
            st.exception(error)
