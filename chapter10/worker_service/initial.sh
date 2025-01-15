#!/bin/bash
sleep 30s
cd /worker-service
celery -A worker.tasks worker --concurrency=${1}
