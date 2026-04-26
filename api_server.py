# api_server.py
# FastAPI wrapper for FORGE-v4 Environment (OpenEnv standard).

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Optional, List
import uvicorn

from env import FORGEEnv
from memory import CoachMemory

app = FastAPI(title="FORGE-v4 OpenEnv API")

# Persistent environment instance
memory = CoachMemory()
env = FORGEEnv(memory=memory)

class Action(BaseModel):
    coder_code: str
    coder_version: str = "unknown"
    candidate_solutions: Optional[List[str]] = None

@app.post("/reset")
async def reset():
    """Reset the environment and return the initial state."""
    state = env.reset()
    return state

@app.post("/step")
async def step(action: Action):
    """Perform a step in the environment."""
    try:
        result = env.step(action.model_dump())
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/state")
async def get_state():
    """Get the current state of the environment."""
    return env.get_state()

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "0.2.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
