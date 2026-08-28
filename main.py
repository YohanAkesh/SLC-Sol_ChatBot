"""
Root FastAPI Entrypoint for S.L.C Solutions ChatBot.
Allows running with:
  - fastapi dev
  - fastapi run
  - uvicorn main:app --reload
  - python main.py
"""

from Src.app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Src.app:app", host="127.0.0.1", port=8000, reload=True)
