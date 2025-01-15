#!/usr/bin/env python
from __future__ import absolute_import
from app.app import start_service
import logging

logging.basicConfig(
    format='%(asctime)s.%(msecs)s:%(name)s:%(thread)d:%(levelname)s:%(process)d:%(message)s',
    level=logging.INFO
)

# Scheduler entry.
if __name__ == '__main__':
    start_service()
