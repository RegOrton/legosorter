from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from train import training_manager, state
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

app = FastAPI(title="Lego Sorter Vision API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train/start")
def start_training(epochs: int = 10, batch_size: int = 32):
    success, msg = training_manager.start_training(epochs=epochs, batch_size=batch_size)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"message": msg}

@app.post("/train/stop")
def stop_training():
    success, msg = training_manager.stop_training()
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"message": msg}

@app.get("/train/status")
def get_status():
    return {
        "is_running": state.is_running,
        "epoch": state.epoch,
        "total_epochs": state.total_epochs,
        "loss": state.loss,
        "logs": state.logs[-10:] # Return last 10 logs
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
