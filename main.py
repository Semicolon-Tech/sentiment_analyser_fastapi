from typing import Optional

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def hello_semicolon():
    return "hello, semicolon!"


@app.get("/ping")
def ping():
    return "pong"

