"""

"""

from ._auto import auto

app = auto.fastapi.FastAPI()


@app.get("/")
def index():
    return {"Hello": "World"}


if __name__ == "__main__":
    auto.uvicorn.run(
        app,
        host="127.0.0.1",
        port=5000,
    )
