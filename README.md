# Sentiment Analysis Using LSTM

The objective of this project is to perform sentiment analysis on movie reviews using a Long Short-Term Memory (LSTM)-based model. The model classifies reviews as positive or negative based on their text content.

## Table of Contents

- [Installation](#installation)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Steps Involved](#steps-involved)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Installation

To get started, you'll need to install the necessary libraries.

```bash
pip install tensorflow nltk tensorflow-datasets
```

## Project Overview

In this project, we'll be using an LSTM (Long Short-Term Memory) model for sentiment analysis on the IMDb movie reviews dataset. The main goal is to classify movie reviews as either positive or negative based on the text content of the reviews.

## Dataset

The dataset used in this project is the **IMDb movie reviews dataset** which consists of movie reviews and their corresponding sentiment labels. The dataset is available through TensorFlow Datasets (`tensorflow-datasets`), which can be easily loaded using `tfds.load()`.

The dataset is split into training and testing sets:
- **Training Set**: Contains a collection of movie reviews for training the model.
- **Testing Set**: Used to evaluate the performance of the model.

## Steps Involved

1. **Data Loading**: Load the IMDb dataset using TensorFlow Datasets (`tensorflow_datasets`).
2. **Data Preprocessing**: Tokenize the text data, remove stopwords, and pad sequences to ensure uniform input length for the LSTM model.
3. **Model Building**: Build a sequential model with an embedding layer, an LSTM layer, and dense output layers.
4. **Model Training**: Train the model using the training dataset.
5. **Evaluation**: Evaluate the model on the test dataset to check its performance.

## Model Architecture

The architecture of the LSTM model is as follows:

1. **Embedding Layer**: This layer converts words into dense vectors of fixed size.
2. **LSTM Layer**: A Long Short-Term Memory (LSTM) layer is used to process the sequences and capture long-term dependencies.
3. **Dense Layer**: The output layer with a sigmoid activation function is used for binary classification (positive or negative sentiment).

## Results

The model will output the classification of movie reviews as either positive or negative based on the sentiment expressed in the text. Accuracy is used to evaluate the performance of the model.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-lstm.git
   cd sentiment-analysis-lstm
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the training script:
   ```bash
   python train.py
   ```

   This will train the LSTM model on the IMDb dataset.

4. Evaluate the model on the test set:
   ```bash
   python evaluate.py
   ```

   This will give the performance metrics such as accuracy on the test set.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
