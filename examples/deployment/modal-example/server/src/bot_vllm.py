#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Bot Implementation.

This module implements a chatbot using OpenAI's GPT-4 model for natural language
processing. It includes:
- Real-time audio/video interaction through Daily
- Animated robot avatar
- Text-to-speech using ElevenLabs
- Support for both English and Spanish

The bot runs as part of a pipeline that processes audio/video frames and manages
the conversation flow.
"""

import os
import sys
from typing import List

from dotenv import load_dotenv
from loguru import logger
from openai.types.chat import ChatCompletionMessageParam
from PIL import Image

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    OutputImageRawFrame,
    SpriteFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

try:
    logger.remove(0)
    logger.add(sys.stderr, level="DEBUG")
except ValueError:
    # Handle the case where logger is already initialized
    pass

# REPLACE WITH YOUR MODAL URL ENDPOINT
modal_url = "https://<Modal workspace>--example-vllm-openai-compatible-serve.modal.run"
api_key = "super-secret-key"  # os.getenv("OPENAI_API_KEY")


class SelfHostedLLMService(OpenAILLMService):
    """A service for interacting with self-hosted LLM using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to self hosted API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.

    Args:
        api_key (str): The API key for accessing your custom LLM
        model (str, optional): The model identifier to use. Defaults to "accounts/fireworks/models/firefunction-v2"
        base_url (str, optional): The base URL for Fireworks API. Defaults to "https://api.fireworks.ai/inference/v1"
        **kwargs: Additional keyword arguments passed to OpenAILLMService
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16",
        base_url: str = f"{modal_url}/inference/v1",
        **kwargs,
    ):
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Fireworks API endpoint."""
        logger.debug(f"Creating Fireworks client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)

    async def get_chat_completions(
        self, context: OpenAILLMContext, messages: List[ChatCompletionMessageParam]
    ):
        """Get chat completions from Fireworks API.

        Removes OpenAI-specific parameters not supported by Fireworks.
        """
        print(f"SelfHostedLLMService:get_chat_completions")
        params = {
            "model": self.model_name,
            "stream": True,
            "messages": messages,
            "tools": context.tools,
            "tool_choice": context.tool_choice,
            "frequency_penalty": self._settings["frequency_penalty"],
            "presence_penalty": self._settings["presence_penalty"],
            "temperature": self._settings["temperature"],
            "top_p": self._settings["top_p"],
            "max_tokens": self._settings["max_tokens"],
        }

        params.update(self._settings["extra"])

        print(f"send: {params}")
        chunks = await self._client.chat.completions.create(**params)
        print(f"done")
        return chunks


sprites = []
script_dir = os.path.dirname(__file__)

# Load sequential animation frames
for i in range(1, 26):
    # Build the full path to the image file
    full_path = os.path.join(script_dir, f"assets/robot0{i}.png")
    # Get the filename without the extension to use as the dictionary key
    # Open the image and convert it to bytes
    with Image.open(full_path) as img:
        sprites.append(OutputImageRawFrame(image=img.tobytes(), size=img.size, format=img.format))

# Create a smooth animation by adding reversed frames
flipped = sprites[::-1]
sprites.extend(flipped)

# Define static and animated states
quiet_frame = sprites[0]  # Static frame for when bot is listening
talking_frame = SpriteFrame(images=sprites)  # Animation sequence for when bot is talking


class TalkingAnimation(FrameProcessor):
    """Manages the bot's visual animation states.

    Switches between static (listening) and animated (talking) states based on
    the bot's current speaking status.
    """

    def __init__(self):
        super().__init__()
        self._is_talking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and update animation state.

        Args:
            frame: The incoming frame to process
            direction: The direction of frame flow in the pipeline
        """
        await super().process_frame(frame, direction)

        # Switch to talking animation when bot starts speaking
        if isinstance(frame, BotStartedSpeakingFrame):
            if not self._is_talking:
                await self.push_frame(talking_frame)
                self._is_talking = True
        # Return to static frame when bot stops speaking
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self.push_frame(quiet_frame)
            self._is_talking = False

        await self.push_frame(frame, direction)


async def run_bot(room_url: str, token: str):
    """Main bot execution function.

    Sets up and runs the bot pipeline including:
    - Daily video transport
    - Speech-to-text and text-to-speech services
    - Language model integration
    - Animation processing
    - RTVI event handling
    """
    # Set up Daily transport with video/audio parameters
    transport = DailyTransport(
        room_url,
        token,
        "Chatbot",
        DailyParams(
            audio_out_enabled=True,
            camera_out_enabled=True,
            camera_out_width=1024,
            camera_out_height=576,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
            #
            # Spanish
            #
            # transcription_settings=DailyTranscriptionSettings(
            #     language="es",
            #     tier="nova",
            #     model="2-general"
            # )
        ),
    )

    # Initialize text-to-speech service
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        #
        # English
        #
        voice_id="pNInz6obpgDQGcFmaJgB",
        #
        # Spanish
        #
        # model="eleven_multilingual_v2",
        # voice_id="gD1IexrzCvsXPHUuT0s3",
    )

    # Initialize LLM service
    # llm = SelfHostedLLMService(api_key="super-secret-key")  # os.getenv("OPENAI_API_KEY"))
    llm = OpenAILLMService(
        # To use OpenAI
        api_key=api_key,
        # Or, to use a local vLLM (or similar) api server
        model="neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16",
        base_url=f"{modal_url}/v1",
    )

    messages = [
        {
            "role": "system",
            #
            # English
            #
            "content": "You are Chatbot, a friendly, helpful robot. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by introducing yourself.",
            #
            # Spanish
            #
            # "content": "Eres Chatbot, un amigable y útil robot. Tu objetivo es demostrar tus capacidades de una manera breve. Tus respuestas se convertiran a audio así que nunca no debes incluir caracteres especiales. Contesta a lo que el usuario pregunte de una manera creativa, útil y breve. Empieza por presentarte a ti mismo.",
        },
    ]

    # Set up conversation context and management
    # The context_aggregator will automatically collect conversation context
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    ta = TalkingAnimation()

    #
    # RTVI events for Pipecat client UI
    #
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            context_aggregator.user(),
            llm,
            tts,
            ta,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )
    await task.queue_frame(quiet_frame)

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()
        # Kick off the conversation
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        await transport.capture_participant_transcription(participant["id"])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        print(f"Participant left: {participant}")
        await task.cancel()

    runner = PipelineRunner()

    await runner.run(task)
