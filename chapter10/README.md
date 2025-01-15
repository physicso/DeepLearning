## DeepGo

Deep Learning Inference Infrastructure.

Directory Tree:

```
├── README.md
├── log-service
│   ├── Dockerfile
│   ├── config
│   │   ├── __init__.py
│   │   └── env.py
│   └── save_log.py
├── scheduler-service
│   ├── Dockerfile
│   ├── app
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── celery_app.py
│   │   └── job
│   │       ├── __init__.py
│   │       ├── job_processing.py
│   ├── config
│   │   ├── __init__.py
│   │   ├── celery_config.py
│   │   ├── env.py
│   │   ├── func.py
│   │   ├── var.py
│   ├── requirements.txt
│   ├── run.py
│   └── utils
│       ├── __init__.py
│       ├── kafka_client.py
│       ├── logger.py
│       └── redis_client.py
└── worker_service
    ├── Dockerfile
    ├── __init__.py
    ├── app
    │   ├── __init__.py
    │   └── celery_app.py
    ├── config
    │   ├── __init__.py
    │   ├── env.py
    │   └── var.py
    ├── initial.sh
    ├── model
    │   └── tfserving.py
    ├── requirements.txt
    ├── utils
    │   ├── __init__.py
    │   ├── error.py
    │   ├── kafka_client.py
    │   ├── log_info.py
    │   ├── logger.py
    │   ├── redis_client.py
    │   └── template.py
    └── worker
        ├── __init__.py
        ├── dogcat.py
        └── tasks.py
```
