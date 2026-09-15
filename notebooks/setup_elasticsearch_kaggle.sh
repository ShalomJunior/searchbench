#!/bin/bash

# ==============================================================================
# SearchBench: Kaggle Elasticsearch Setup Script
# ==============================================================================
# Kaggle notebooks run as root, but Elasticsearch refuses to run as root.
# This script downloads Elasticsearch, creates a non-root user, and starts the 
# Elasticsearch daemon in the background so you can run the optimized benchmarks.
# ==============================================================================

echo "Downloading Elasticsearch 8.12.0..."
wget -q https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.12.0-linux-x86_64.tar.gz

echo "Extracting Elasticsearch..."
tar -xzf elasticsearch-8.12.0-linux-x86_64.tar.gz
rm elasticsearch-8.12.0-linux-x86_64.tar.gz

echo "Configuring permissions for non-root execution..."
useradd -m elasticuser
chown -R elasticuser:elasticuser elasticsearch-8.12.0

echo "Starting Elasticsearch daemon in the background..."
su - elasticuser -c "./elasticsearch-8.12.0/bin/elasticsearch -d -E xpack.security.enabled=false -E discovery.type=single-node"

echo "Waiting for Elasticsearch to boot (15 seconds)..."
sleep 15

echo "Elasticsearch Status:"
curl -X GET "localhost:9200/"
