#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 730335449960.dkr.ecr.ap-south-1.amazonaws.com

echo "Pulling Docker image..."
docker pull 730335449960.dkr.ecr.ap-south-1.amazonaws.com/worker-reach-time:latest

echo "Checking for existing container..."
if [ "$(docker ps -q -f name=delivery_time_pred)" ]; then
    echo "Stopping existing container..."
    docker stop delivery_time_pred
fi

if [ "$(docker ps -aq -f name=delivery_time_pred)" ]; then
    echo "Removing existing container..."
    docker rm delivery_time_pred
fi

echo "Starting new container..."
docker run -d -p 80:8000 --name delivery_time_pred -e DAGSHUB_USER_TOKEN=a35ec3cdd7d358499449b0a38fcbc9297561f7a8 730335449960.dkr.ecr.ap-south-1.amazonaws.com/worker-reach-time:latest

echo "Container started successfully."