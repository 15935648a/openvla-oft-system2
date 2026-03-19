import logging
import sys
import os

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Allow importing from experiments/robot/libero
sys.path.insert(0, os.path.dirname(__file__))
from experiments.robot.libero.system2_agent import System2Agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

agent: System2Agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    logger.info("Loading System2Agent...")
    agent = System2Agent()
    logger.info("System2Agent ready.")
    yield
    agent = None


app = FastAPI(lifespan=lifespan)


class SubgoalRequest(BaseModel):
    task: str
    summary: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/subgoal")
def subgoal(req: SubgoalRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not loaded")
    result = agent.next_subgoal(req.task, req.summary)
    return {"subgoal": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("system2_server:app", host="0.0.0.0", port=8000, reload=False)
