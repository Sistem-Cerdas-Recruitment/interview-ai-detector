# Interview AI Detector

## Overview

Interview AI Detector is a machine learning model designed to distinguish between human and AI-generated responses during interviews. The system is composed of two models:

1. **ALBERT Model**: Processes text features extracted from responses.
2. **Logistic Regression Model (LogReg)**: Utilizes the output from the ALBERT model along with additional behavioral features to make the final prediction.

The model is deployed on Google Vertex AI, with integration managed by a Kafka consumer deployed on Google Compute Engine. Both the model and Kafka consumer utilize FastAPI for API management.

## Architecture

### ALBERT Model

- **Source**: HuggingFace
- **Input**: 25 numerical features extracted from the text, including:
  - Part-of-Speech (POS) tags
  - Readability scores
  - Sentiment analysis
  - Perplexity numbers
- **Output**: Features used as input for the Logistic Regression model

### Logistic Regression Model

- **Input**: 
  - Output from the ALBERT model
  - 4 additional features, including typing behavior metrics such as backspace count and key presses per letter
- **Output**: Final prediction indicating whether the response is human or AI-generated

## Deployment

- **Model Deployment**: Vertex AI
- **Kafka Consumer Deployment**: Compute Engine
- **API Framework**: FastAPI
- **Training**: 
  - **Epochs**: 8
  - **Dataset**: 2000 data points (1000 human responses, 1000 AI-generated responses)
  - **Framework**: PyTorch

## Usage

### API Endpoints

- **POST /predict**: 
  - **Description**: Receives a pair of question and answer, along with typing behavior metrics. Runs the prediction pipeline and returns the result.
  - **Input**:
    ```json
    {
      "question": "Your question text",
      "answer": "The given answer",
      "backspace_count": 5,
      "letter_click_counts": {"a": 27, "b": 4, "c": 9, "d": 17, "e": 54, "f": 12, "g": 4, "h": 15, "i": 25, "j": 2, "k": 2, "l": 14, "m": 10, "n": 23, "o": 23, "p": 9, "q": 1, "r": 24, "s": 19, "t": 36, "u": 9, "v": 6, "w": 8, "x": 1, "y": 7, "z": 0}
    }
    ```
  - **Output**:
    ```json
    {
      "predicted_class": "HUMAN" or "AI",
      "main_model_probability": "0.85",
      "secondary_model_probability": "0.75",
      "confidence": "High Confidence" or "Partially Confident" or "Low Confidence"
    }
    ```

## Limitations

- The model is not designed for retraining. The current implementation focuses solely on deployment and prediction.
- The repository is meant for deployment purposes only and does not support local installation for development.

## Author
Yakobus Iryanto Prasethio