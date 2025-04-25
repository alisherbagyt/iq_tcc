# Transport Complaints Classifier

A machine learning model that classifies public transport complaints as positive or negative based on review text.
Video demonstration: https://youtu.be/h6LEtPcEHu4

## Project Overview

This project analyzes public transport complaint texts to automatically classify them as positive or negative reviews. Using natural language processing techniques and machine learning, the model preprocesses text data, extracts features, and predicts sentiment with high accuracy.

## Features

- Text preprocessing (cleaning, lemmatization, stopword removal)
- Language detection (Russian/Kazakh)
- TF-IDF vectorization for feature extraction
- Balanced dataset creation
- Machine learning classification with Logistic Regression
- Performance evaluation with cross-validation
- Data visualization (sentiment distribution, word frequencies, model performance)

## Dataset

The project uses a dataset of transport complaints stored in `AI_dataset.xlsx`. The dataset contains text reviews in primarily Russian language.

## Model Performance

The model has been evaluated using 5-fold cross-validation and shows strong performance metrics:
- Accuracy: ~85-90%
- Balanced precision and recall for both positive and negative classes

## Technologies Used

- Python 3.x
- pandas - Data manipulation
- scikit-learn - Machine learning
- NLTK - Natural language processing
- matplotlib/seaborn - Data visualization
- Transformers (Hugging Face) - Initial sentiment labeling
- imbalanced-learn - Handling class imbalance

## Project Structure

```
├── AI_dataset.xlsx              # Dataset file
├── TCC_for_IQ_V2.ipynb          # Jupyter notebook with analysis
├── README.md                    # Project documentation
└── requirements.txt             # Required Python packages
```

## Getting Started

### Prerequisites

- Python 3.7+
- Required packages listed in requirements.txt

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/transport-complaints-classifier.git
cd transport-complaints-classifier
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage

1. The main analysis and model development can be found in the Jupyter notebook:
```bash
jupyter notebook TCC_for_IQ_V2.ipynb
```


## Results

The model successfully identifies positive and negative sentiments in transport complaints with high accuracy. Key insights include:

- Most informative words for sentiment classification
- Distribution of sentiments in the dataset
- Performance evaluation through confusion matrix and classification metrics
