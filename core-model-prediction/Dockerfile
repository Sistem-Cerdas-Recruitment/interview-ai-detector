# HF Token args
ARG HF_TOKEN

# Use an official Python runtime as a base image
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y unzip \
    && pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt wordnet averaged_perceptron_tagger

# Unzip wordnet
RUN unzip /root/nltk_data/corpora/wordnet.zip -d /root/nltk_data/corpora/

# Print masked HF Token for debugging
RUN echo ${HF_TOKEN:0:4}********${HF_TOKEN: -4}

# Download HuggingFace model
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b', token='$HF_TOKEN'); \
    model = AutoModelForCausalLM.from_pretrained('google/gemma-2b', token='$HF_TOKEN'); \
    tokenizer.save_pretrained('/app/gemma-2b'); \
    model.save_pretrained('/app/gemma-2b')"

# Model env
ENV MODEL_DIR=/app/gemma-2b

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run uvicorn
CMD ["uvicorn", "prediction:app", "--host", "0.0.0.0", "--port", "8080"]