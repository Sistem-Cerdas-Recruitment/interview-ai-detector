# Use an official Python runtime as a base image
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt wordnet averaged_perceptron_tagger

# Unzip wordnet
RUN unzip /root/nltk_data/corpora/wordnet.zip -d /root/nltk_data/corpora/

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run uvicorn
CMD ["uvicorn", "prediction:app", "--host", "0.0.0.0", "--port", "8080"]