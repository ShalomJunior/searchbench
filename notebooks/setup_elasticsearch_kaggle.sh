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
# We set Java memory to 1GB to prevent Kaggle out-of-memory errors
su - elasticuser -c "export ES_JAVA_OPTS='-Xms1g -Xmx1g'; ./elasticsearch-8.12.0/bin/elasticsearch -d -E xpack.security.enabled=false -E discovery.type=single-node > /tmp/es.log 2>&1"

echo "Waiting for Elasticsearch to boot (this can take up to 60 seconds)..."
for i in {1..30}; do
    if curl -s http://localhost:9200/ > /dev/null; then
        echo "Elasticsearch is UP and RUNNING!"
        curl -s http://localhost:9200/
        exit 0
    fi
    echo -n "."
    sleep 2
done

echo ""
echo "ERROR: Elasticsearch failed to start in time. Here are the logs:"
tail -n 20 /tmp/es.log
