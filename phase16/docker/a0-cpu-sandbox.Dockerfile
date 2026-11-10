FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install --yes --no-install-recommends bubblewrap \
    && ln -s /usr/local/bin/python3 /usr/bin/python3 \
    && ln -s /usr/local/lib/libpython3.11.so.1.0 /usr/lib/libpython3.11.so.1.0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8

WORKDIR /repo
