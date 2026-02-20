
import asyncio
import os

import requests
from dotenv import load_dotenv

# dynamic import for endpoint modules
import importlib

from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

module_path = "endpoint"
for filename in os.listdir(module_path):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = f"{module_path}.{filename[:-3]}"
        module = importlib.import_module(module_name)
        if hasattr(module, "router"):
            logging.info(f"Included router from {filename}")
            app.include_router(module.router, prefix="/api")
            
@app.get("/")
async def root():
    return {"message": "Hello World, This is Rise detection application!"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8009))
    logging.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", timeout_keep_alive=-1, port=port, log_level="info")
