import os
import subprocess
from contextlib import asynccontextmanager
from typing import Any, Dict

import aiohttp
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper, DailyRoomParams

router = APIRouter()

MAX_BOTS_PER_ROOM = 1
bot_procs = {}
daily_helpers = {}


def cleanup():
    """Cleanup function to terminate all bot processes.

    Called during server shutdown.
    """
    for entry in bot_procs.values():
        proc = entry[0]
        proc.terminate()
        proc.wait()


def get_bot_file():
    bot_implementation = os.getenv("BOT_IMPLEMENTATION", "openai").lower().strip()
    if not bot_implementation:
        bot_implementation = "openai"
    if bot_implementation not in ["openai", "gemini"]:
        raise ValueError(
            f"Invalid BOT_IMPLEMENTATION: {bot_implementation}. Must be 'openai' or 'gemini'"
        )
    return f"bot-{bot_implementation}"


async def create_room_and_token() -> tuple[str, str]:
    room_url = os.getenv("DAILY_SAMPLE_ROOM_URL", None)
    token = os.getenv("DAILY_SAMPLE_ROOM_TOKEN", None)
    if not room_url:
        room = await daily_helpers["rest"].create_room(DailyRoomParams())
        if not room.url:
            raise HTTPException(status_code=500, detail="Failed to create room")
        room_url = room.url

        token = await daily_helpers["rest"].get_token(room_url)
        if not token:
            raise HTTPException(status_code=500, detail=f"Failed to get token for room: {room_url}")

    return room_url, token


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager that handles startup and shutdown tasks.

    - Creates aiohttp session
    - Initializes Daily API helper
    - Cleans up resources on shutdown
    """
    aiohttp_session = aiohttp.ClientSession()
    daily_helpers["rest"] = DailyRESTHelper(
        daily_api_key=os.getenv("DAILY_API_KEY", ""),
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )
    yield
    await aiohttp_session.close()
    cleanup()


@router.get("/")
async def start_agent(request: Request):
    room_url, token = await create_room_and_token()
    num_bots_in_room = sum(
        1 for proc in bot_procs.values() if proc[1] == room_url and proc[0].poll() is None
    )
    if num_bots_in_room >= MAX_BOTS_PER_ROOM:
        raise HTTPException(status_code=500, detail=f"Max bot limit reached for room: {room_url}")

    try:
        bot_file = get_bot_file()
        print(f"Starting bot process: {bot_file} -u {room_url} -t {token}")
        print(f"DIRNAME: {os.path.dirname(os.path.abspath(__file__))}")
        proc = subprocess.Popen(
            [f"python3 -m {bot_file} -u {room_url} -t {token}"],
            shell=True,
            bufsize=1,
            cwd=f"{os.path.dirname(os.path.abspath(__file__))}/bots",
        )
        bot_procs[proc.pid] = (proc, room_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start subprocess: {e}")

    return RedirectResponse(room_url)


@router.post("/connect")
async def rtvi_connect(request: Request) -> Dict[Any, Any]:
    room_url, token = await create_room_and_token()
    try:
        bot_file = get_bot_file()
        proc = subprocess.Popen(
            [f"python3 -m {bot_file} -u {room_url} -t {token}"],
            shell=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        bot_procs[proc.pid] = (proc, room_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start subprocess: {e}")

    return {"room_url": room_url, "token": token}


@router.get("/status/{pid}")
def get_status(pid: int):
    proc = bot_procs.get(pid)
    if not proc:
        raise HTTPException(status_code=404, detail=f"Bot with process id: {pid} not found")

    status = "running" if proc[0].poll() is None else "finished"
    return JSONResponse({"bot_id": pid, "status": status})
