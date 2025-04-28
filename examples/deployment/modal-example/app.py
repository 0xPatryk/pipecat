#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import importlib
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

import aiohttp
import modal
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper, DailyRoomParams

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("pipecat-ai[daily,silero,cartesia,openai]")
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App("pipecat-modal", image=image, secrets=[modal.Secret.from_dotenv()])

router = APIRouter()

MAX_BOTS_PER_ROOM = 1
bot_jobs = {}
daily_helpers = {}


def cleanup():
    """Cleanup function to terminate all bot processes.

    Called during server shutdown.
    """
    for entry in bot_jobs.values():
        func = modal.FunctionCall.from_id(entry[0])
        if func:
            func.cancel()


def get_bot_file():
    bot_implementation = os.getenv("BOT_IMPLEMENTATION", "openai").lower().strip()
    if not bot_implementation:
        bot_implementation = "openai"
    if bot_implementation not in ["openai", "gemini"]:
        raise ValueError(
            f"Invalid BOT_IMPLEMENTATION: {bot_implementation}. Must be 'openai' or 'gemini'"
        )
    return f"bot_{bot_implementation}"


def get_runner(path: str, bot_file: str):
    """Dynamically import the run_bot function based on the bot name.

    Args:
        bot_name (str): The name of the bot implementation (e.g., 'openai', 'gemini').

    Returns:
        function: The run_bot function from the specified bot module.

    Raises:
        ImportError: If the specified bot module or run_bot function is not found.
    """
    try:
        # Dynamically construct the module name
        module_name = f"{path}.{bot_file}"
        # Import the module
        module = importlib.import_module(module_name)
        # Get the run_bot function from the module
        return getattr(module, "run_bot")
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import run_bot from {module_name}: {e}")


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


@app.function(image=image)
async def launch_bot(room_url, token):
    try:
        path = "src"
        bot_file = get_bot_file()
        run_bot = get_runner(path, bot_file)

        print(f"Starting bot process: {bot_file} -u {room_url} -t {token}")
        await run_bot(room_url, token)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start bot pipeline: {e}")


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


async def start():
    room_url, token = await create_room_and_token()
    launch_bot_func = modal.Function.from_name("pipecat-modal", "launch_bot")
    function_id = launch_bot_func.spawn(room_url, token)
    bot_jobs[function_id] = (function_id, room_url)

    return room_url, token


@router.get("/")
async def start_agent(request: Request):
    room_url, token = await start()

    return RedirectResponse(room_url)


@router.post("/connect")
async def rtvi_connect(request: Request) -> Dict[Any, Any]:
    room_url, token = await start()

    return {"room_url": room_url, "token": token}


@router.get("/status/{fid}")
def get_status(fid: str):
    func = modal.FunctionCall.from_id(fid)
    if not func:
        raise HTTPException(status_code=404, detail=f"Bot with process id: {fid} not found")

    try:
        result = func.get(timeout=0)
        return JSONResponse({"bot_id": fid, "status": "finished", "code": result})
    except modal.exception.OutputExpiredError:
        return JSONResponse({"bot_id": fid, "status": "finished", "code": 404})
    except TimeoutError:
        return JSONResponse({"bot_id": fid, "status": "running", "code": 202})


@app.function(min_containers=1)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def fastapi_app():
    # Initialize FastAPI app
    web_app = FastAPI(lifespan=lifespan)

    # Include the endpoints from endpoints.py
    web_app.include_router(router)

    return web_app
