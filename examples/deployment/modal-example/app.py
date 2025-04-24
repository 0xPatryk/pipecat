#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import modal

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("pipecat-ai[daily,silero,cartesia,openai]")
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App("pipecat-modal", image=image, secrets=[modal.Secret.from_dotenv()])


@app.function(min_containers=1)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from src.endpoints import lifespan
    from src.endpoints import router as endpoints_router

    # Initialize FastAPI app
    web_app = FastAPI(lifespan=lifespan)

    # Configure CORS to allow requests from any origin
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include the endpoints from endpoints.py
    web_app.include_router(endpoints_router)

    return web_app
