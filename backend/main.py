import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import analyze
from backend.services.model_loader import model_manager

app = FastAPI(title="Dementia Speech Analysis Prototype")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    model_manager.initialize()

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "model_initialized": model_manager.initialized}

app.include_router(analyze.router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
