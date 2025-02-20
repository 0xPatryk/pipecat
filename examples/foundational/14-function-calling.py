#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMContext, OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def start_fetch_products(function_name, llm, context):
    """Push a frame to the LLM; this is handy when the LLM response might take a while."""
    await llm.push_frame(TTSSpeakFrame("I'll take a look!"))
    logger.debug(f"Starting fetch_products_from_api with function_name: {function_name}")


async def fetch_products_from_api(function_name, tool_call_id, args, llm, context, result_callback):
    logger.debug(f"args for fetch_products_from_api: {args}")
    # In the real world you'd fetch the products from an API. We're hardcoding them here.
    product = args["product"]
    if product == "vacuums":
        await result_callback({"vacuums": ["Dyson V11", "Roomba i7"]})
    elif product == "tvs":
        await result_callback({"tvs": ["Samsung 65 inch", "LG 55 inch"]})
    else:
        await result_callback({"error": "Unknown product"})


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
        # Register a function_name of None to get all functions
        # sent to the same callback with an additional function_name parameter.
        llm.register_function(None, fetch_products_from_api, start_callback=start_fetch_products)

        tools = [
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": "get_products",
                    "description": "Get the list of products available.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product": {
                                "type": "string",
                                "enum": ["vacuums", "tvs"],
                                "description": "The type of product to show.",
                            }
                        },
                        "required": ["product"],
                    },
                },
            )
        ]
        messages = [
            {
                "role": "system",
                "content": "You are a helpful customer service agent named Hailey in a video call. Your goal is to sell vacuums or tvs. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
            },
        ]

        context = OpenAILLMContext(messages, tools)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
                report_only_initial_ttfb=True,
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            # Kick off the conversation.
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
