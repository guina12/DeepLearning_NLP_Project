# Sentiment Analysis of Product Reviews

This project implements a Sentiment Analysis model for product reviews using advanced NLP techniques. The model leverages LSTM, GRU, and Embeddings to predict the sentiment of a given product review as positive, negative, or neutral. The project focuses on preprocessing text data, creating a custom dataset using TFRecord, and building a deep learning architecture with recurrent layers.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Code Overview](#code-overview)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview

This project is designed to perform sentiment analysis on product reviews. It uses TensorFlow and Keras to build a recurrent neural network (RNN) that processes sequences of text and classifies them based on sentiment. The network is trained with product review data to identify patterns of sentiment using LSTM and GRU layers.

## Features

### Text Preprocessing:
- Custom text preprocessing using TensorFlow functions like `tf.strings.substr` and `tf.strings.regex_replace`.
- Removes HTML tags, non-alphabetic characters, and tokenizes strings.

### Custom Dataset Creation:
- Utilizes TFRecordDataset for efficient data storage and processing.
- Functions for loading CSV data into TensorFlow pipelines using `tf.io.decode_csv`.

### Neural Network Architecture:
- Incorporates embeddings for word representations.
- Sequential processing using GRU layers.
- Output layer with softmax activation to predict one of the three sentiment classes: positive, negative, neutral.

## Model Architecture

### Input Layer:
- Accepts sequences of arbitrary lengths and creates a mask for padding using the Lambda layer.

### Embedding Layer:
- Converts words into dense vectors for numerical representation.

### GRU Layers:
- Uses GRU (Gated Recurrent Units) layers to capture long-term dependencies in text sequences.

### Output Layer:
- A dense layer with softmax activation to classify sentiment into three categories.

## Code Overview

### 1. Text Preprocessing Functions

```python
@tf.function
def preprocess_strings(X_batch, y_batch):
    X_batch = tf.strings.substr(X_batch, 0, 200)
    X_batch = tf.strings.regex_replace(X_batch, b"<br\\s*/?>", b" ")
    X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z]", b" ")
    X_batch = tf.strings.split(X_batch)
    return X_batch.to_tensor(default_value=b'<pad>'), y_batch
```

- Truncates strings to 200 characters.
- Removes HTML tags and non-alphabetic characters.
- Tokenizes and pads sequences with `<pad>` tokens.

```python
@tf.function
def preprocess(lines):
    defs = [""] * 1 + [tf.constant([], dtype=tf.int64)]  # Specify column types
    fields = tf.io.decode_csv(lines, record_defaults=defs)
    X = fields[0]
    y = fields[-1]
    return X, y
```

- Decodes CSV lines into TensorFlow tensors for text and labels.

### 2. Dataset Creation

```python
dataset = tf.data.TFRecordDataset(['/content/sentiment_analysis.tfrecords']).interleave(
    lambda filepaths: tf.data.TextLineDataset(filepaths).skip(1)
)
```

- Loads data from TFRecord files for efficient training.

```python
with tf.io.TFRecordWriter('sentiment_analysis.tfrecords') as f:
    f.write('/content/sentiment_review.cv')
```

- Writes CSV data into a TFRecord file for optimized storage.

### 3. Model Architecture

```python
InputLayer = keras.layers.Input(shape=[None])  # Process sequences of arbitrary size
mask = keras.layers.Lambda(lambda inputs: k.not_equal(inputs, 0))(InputLayer)  # Mask padding
z = keras.layers.Embedding(input_dim=len(truncated_vocab) + out_of_vocabulary, output_dim=2)(InputLayer)
gruCell = keras.layers.GRU(50, return_sequences=True, use_cudnn=False)(z, mask=mask)
gruCell = keras.layers.GRU(50, use_cudnn=False)(gruCell, mask=mask)
outputLayer = keras.layers.Dense(3, activation='softmax')(gruCell)
model_without_pad = keras.Model(inputs=[InputLayer], outputs=[outputLayer])
```

The model processes sequences of text using embeddings and GRU layers, then classifies the sentiment into three categories.

## Usage

### 1. Install Dependencies

Ensure that you have the following dependencies installed:

```bash
pip install tensorflow keras
```

### 2. Running the Model

1. Preprocess your text data using the provided preprocessing functions.
2. Load your dataset into a TFRecord format.
3. Train the model using the custom architecture built with LSTM and GRU layers.

## Results

The model outputs the sentiment prediction for each input product review:

- 0: Negative Sentiment
- 1: Neutral Sentiment
- 2: Positive Sentiment

## Embeddings

![image](https://github.com/user-attachments/assets/b962af5c-675d-4f31-ae20-b678493d173d)


## Future Improvements

- **Integration with CNN**: Combining text-based models with convolutional neural networks (CNNs) to enhance the feature extraction process.
- **Hyperparameter Tuning**: Optimizing hyperparameters like embedding size, GRU/LSTM units, and learning rates.
- **Real-time Sentiment Prediction**: Deploying the model as an API for real-time sentiment analysis.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow documentation for providing extensive support for text-based models.
- Keras for making model building and experimentation easy with its high-level API.
