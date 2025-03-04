# Use the existing image as a base image
FROM salma6633/salma_labidi_4ds4_mlops

# Set the working directory to /app
WORKDIR /app

# Install FastAPI and Uvicorn
RUN pip install fastapi uvicorn

# Install MLflow
RUN pip install mlflow

# Expose the ports for FastAPI and MLflow
EXPOSE 8000 5000

# Copy all your project files into the container
COPY . .

# Command to run both FastAPI and MLflow server
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 & mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root /app/mlruns"]
