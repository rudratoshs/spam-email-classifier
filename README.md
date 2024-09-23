# **📧 Spam Email Classifier**

## **🚀 Overview**

This project builds an advanced **Spam Email Classifier** using the **Naive Bayes algorithm**. The goal is to categorize emails as either **spam** or **ham** (not spam) by analyzing the content of the emails. The model was trained on the **Enron Email Dataset** and achieves an impressive accuracy of **98.13%**. This classifier can be integrated into email systems to filter out unwanted spam emails, improving user experience. 

### **🔑 Key Features:**
- 📩 **Email Classification**: Predicts whether an email is spam or not.
- 📊 **CountVectorizer** for text feature extraction.
- 🤖 **Multinomial Naive Bayes** used for training the model.
- 🎯 **High accuracy** of 98.13%.
- 🖼️ Confusion matrix visualization to evaluate model performance.
- 💾 Model saved for future use as `nb_model.pkl`.

---

## **🎯 Use Case**

Spam emails are a constant nuisance, cluttering inboxes and posing security risks such as phishing. This project offers an automated solution to **classify emails as spam or ham**. By analyzing the content of the emails, it filters out irrelevant or harmful messages and ensures that important communication is not lost in the flood of junk mail.

### **Potential Applications**:
1. **📧 Email Services**: Automate spam filtering for email platforms like Gmail, Yahoo, or Outlook.
2. **🔒 Security Systems**: Use the classifier to detect phishing attempts by identifying spam patterns.
3. **💼 Business Communication**: Clean up customer support or general inquiry inboxes by automatically removing spam.
4. **📬 Personal Use**: Integrate with personal email clients to reduce unwanted messages.

---

## **📂 Project Structure**

```
spam-email-classifier/
├── data/                     # Folder for the dataset
│   ├── ham/                  # Non-spam emails
│   └── spam/                 # Spam emails
│
├── notebooks/                # Jupyter Notebook files
│   └── spam_classifier.ipynb # Main notebook for model development
│
├── models/                   # Folder to save trained models
│   ├── nb_model.pkl          # Saved Naive Bayes model
│   └── vectorizer.pkl        # Saved CountVectorizer for transforming text
│
├── visualizations/           # Folder for plots and visualizations
│   └── confusion_matrix.png  # Confusion matrix plot
│
├── requirements.txt          # Dependencies file
├── README.md                 # Project documentation (you are reading this)
└── .gitignore                # Files to ignore (e.g., data files)
```

---

## **📥 Dataset**

The project uses the **Enron Spam Dataset** 📚, which is publicly available and contains a large corpus of spam and ham emails. Download the dataset from [Enron Spam Dataset](https://www.cs.cmu.edu/~enron/).

- **`ham/` folder**: Contains legitimate (non-spam) emails.
- **`spam/` folder**: Contains spam emails.

---

## **⚙️ Setup Instructions**

### 1. **Clone the Repository**

First, clone the repository to your local machine:

```bash
git clone https://github.com/rudratoshs/spam-email-classifier.git
cd spam-email-classifier
```

### 2. **Create a Virtual Environment**

It’s recommended to use a virtual environment to isolate your dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

### 3. **Install Dependencies**

Install the required libraries by running:

```bash
pip install -r requirements.txt
```

### 4. **Download the Dataset**

Download the **Enron Spam Dataset** from [here](https://www.cs.cmu.edu/~enron/) and place the `ham/` and `spam/` folders inside the `data/` directory:

```
spam-email-classifier/
├── data/
│   ├── ham/   # Legitimate (non-spam) emails
│   └── spam/  # Spam emails
```

### 5. **Run Jupyter Notebook**

To explore and train the model, open the Jupyter notebook and run the cells in `spam_classifier.ipynb`:

```bash
jupyter notebook
```

### 6. **Train the Model**

The notebook trains the Naive Bayes classifier and generates predictions. The model and vectorizer will be saved as `nb_model.pkl` and `vectorizer.pkl` in the `models/` folder.

### 7. **Visualize the Confusion Matrix**

The confusion matrix will be saved as `confusion_matrix.png` in the `visualizations/` folder, helping you understand the model's performance.

---

## **💻 Usage Instructions**

After training the model, you can use it to classify new emails. Here’s how to load the model and make predictions:

```python
import joblib

# Load the saved model and vectorizer
nb_model = joblib.load('models/nb_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# Example: Classify a new email
new_email = ["Congratulations! You've won a prize. Claim it now!"]
new_email_vectorized = vectorizer.transform(new_email)
prediction = nb_model.predict(new_email_vectorized)

print(f"Prediction: {prediction}")  # Output will be either 'spam' or 'ham'
```

---

## **📊 Model Evaluation**

The model achieved an accuracy of **98.13%** on the test dataset. Below is the confusion matrix for performance evaluation:

```text
Confusion Matrix:
[[1107   13]
 [  16  416]]
```

- **True Positives (1107)**: Correctly identified ham emails.
- **False Positives (13)**: Misclassified ham as spam.
- **False Negatives (16)**: Misclassified spam as ham.
- **True Negatives (416)**: Correctly identified spam emails.

### 📉 Confusion Matrix Visualization

![Confusion Matrix](visualizations/confusion_matrix.png)

---

## **🌱 Future Enhancements**

- 🌐 **TF-IDF**: Implementing TF-IDF for feature extraction instead of simple word counts.
- ⚙️ **Additional Classifiers**: Try SVM or Random Forest for performance comparison.
- 💡 **Real-time Classification**: Integrate with email clients for real-time spam detection.
- 🛡️ **Phishing Detection**: Enhance to detect phishing emails using additional features like email headers.

---

## **🤝 Contributors**

- [Rudratosh](https://github.com/rudratoshs)

---

## **📄 License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.