from fastapi import FastAPI

app = FastAPI(
    title="NextGen Bank - FastAPI",
    description="Fully features banking API built with FastAPI",
)


@app.get("/")
def home():
    return {"message": "Welcome to the NextGen Bank API"}
