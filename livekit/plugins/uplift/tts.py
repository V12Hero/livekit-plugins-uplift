from __future__ import annotations

import asyncio
import os
import weakref
from dataclasses import dataclass
from typing import List, Optional

import aiohttp
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)

from .log import logger
from .models import TTSEncoding, TTSDefaultVoiceId, AvailableVoices

def _sample_rate_from_format(output_format: TTSEncoding) -> int:
    """
    Extracts the sample rate from the given output format string.

    Args:
        output_format (TTSEncoding): The encoding format string, e.g., 'MP3_22050_128'.

    Returns:
        int: The sample rate extracted from the format string. Defaults to 22050 Hz if not specified.
    """
    parts = output_format.split("_")
    return int(parts[1]) if len(parts) > 1 else 22050

@dataclass
class _TTSOptions:
    api_key: str
    voice: str
    output_format: TTSEncoding
    base_url: str

class TTS(tts.TTS):
    """
    Provides an interface to the Uplift AI Text-to-Speech (TTS) service.

    Attributes:
        _opts (_TTSOptions): Configuration options for the TTS service.
        _session (aiohttp.ClientSession): HTTP session for making API requests.
        _streams (weakref.WeakSet): A set of active synthesis streams.
    """
    def __init__(
        self,
        *,
        voice: str = TTSDefaultVoiceId,
        output_format: TTSEncoding = "MP3_22050_128",
        api_key: str | None = None,
        base_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Initializes the TTS client with specified parameters.

        Args:
            voice (str): The voice ID to use for synthesis. Defaults to TTSDefaultVoiceId.
            output_format (TTSEncoding): The desired audio output format. Defaults to 'MP3_22050_128'. Find more options here: https://docs.upliftai.org/api-reference/endpoint/text-to-speech-stream
            api_key (str, optional): The API key for authenticating with the Uplift AI service. If not provided, it is read from the 'UPLIFT_AI_API_KEY' environment variable.
            base_url (str, optional): The base URL for the Uplift AI API. Defaults to 'https://api.upliftai.org/v1'.
            http_session (aiohttp.ClientSession, optional): An existing HTTP session to use for requests. If not provided, a new session is created.

        Raises:
            ValueError: If no API key is provided or found in the environment variables.

        Note:
            The api_key must be set either through the constructor argument or by setting the UPLIFT_AI_API_KEY environmental variable.
        """
        sample_rate = _sample_rate_from_format(output_format)
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )

        api_key = api_key or os.environ.get("UPLIFT_AI_API_KEY")
        if not api_key:
            raise ValueError("Uplift API key is required, either as an argument or set in UPLIFT_AI_API_KEY environment variable")

        self._opts = _TTSOptions(
            api_key=api_key,
            voice=voice,
            output_format=output_format,
            base_url=base_url or "https://api.upliftai.org/v1",
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()

    def _ensure_session(self) -> aiohttp.ClientSession:
        """
        Ensures that an HTTP session is available.

        Returns:
            aiohttp.ClientSession: The HTTP session to use for API requests.
        """
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def list_voices(self) -> List[str]:
        """
        Retrieves a list of available voice IDs from the Uplift AI service.

        Returns:
            List[str]: A list of available voice identifiers.
        """
        return AvailableVoices

    def synthesize(
        self,
        text: str,
        *,
        conn_options: Optional[APIConnectOptions] = None,
    ) -> ChunkedStream:
        """
        Initiates a non-streaming text-to-speech synthesis request.

        Args:
            text (str): The text to be synthesized into speech.
            conn_options (Optional[APIConnectOptions]): Connection options for the API request.

        Returns:
            ChunkedStream: An object representing the synthesis process, yielding audio chunks as they are received.
        """
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
            session=self._ensure_session(),
        )

    # Updated stream() method: no text parameter required.
    def stream(self, *, conn_options: Optional[APIConnectOptions] = None) -> SynthesizeStream:
        """
        Initiates a streaming text-to-speech synthesis session.

        Returns:
            SynthesizeStream: An object representing the streaming synthesis process.
        """
        stream = SynthesizeStream(
            tts=self,
            opts=self._opts,
            conn_options=conn_options,
            session=self._ensure_session(),
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        """
        Closes all active synthesis streams and performs necessary cleanup.
        """
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await super().aclose()


class ChunkedStream(tts.ChunkedStream):
    """
    Handles non-streaming text-to-speech synthesis using the Uplift AI API.

    Attributes:
        _tts (TTS): The TTS client instance.
        _opts (_TTSOptions): Configuration options for the TTS service.
        _session (aiohttp.ClientSession): HTTP session for making API requests.
    """
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        opts: _TTSOptions,
        conn_options: Optional[APIConnectOptions] = None,
        session: aiohttp.ClientSession,
    ) -> None:
        """
        Initializes the ChunkedStream with the provided parameters.

        Args:
            tts (TTS): The TTS client instance.
            input_text (str): The text to be synthesized.
            opts (_TTSOptions): Configuration options for the TTS service.
            conn_options (Optional[APIConnectOptions]): Connection options for the API request.
            session (aiohttp.ClientSession): HTTP session for making API requests.
        """
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts
        self._opts = opts
        self._session = session

    async def _run(self) -> None:
        """
        Executes the synthesis process by sending a request to the Uplift AI API and processing the response.
        """
        request_id = utils.shortuuid()
        data = {
            "voiceId": self._opts.voice,
            "text": self._input_text,
            "outputFormat": self._opts.output_format,
        }
        headers = {
            "Authorization": f"Bearer {self._opts.api_key}",
            "Content-Type": "application/json",
        }
        decoder = utils.codecs.AudioStreamDecoder(
            sample_rate=self._tts.sample_rate,
            num_channels=1,
        )
        decode_task: asyncio.Task | None = None
        try:
            async with self._session.post(
                _synthesize_url(self._opts),
                headers=headers,
                json=data,
            ) as resp:
                if not resp.content_type.startswith("audio/"):
                    content = await resp.text()
                    logger.error("Uplift returned non-audio data: %s", content)
                    return

                async def _decode_loop():
                    try:
                        async for chunk in resp.content.iter_chunked(1024):
                            decoder.push(chunk)
                    finally:
                        decoder.end_input()

                decode_task = asyncio.create_task(_decode_loop())
                emitter = tts.SynthesizedAudioEmitter(
                    event_ch=self._event_ch,
                    request_id=request_id,
                )
                async for frame in decoder:
                    emitter.push(frame)
                emitter.flush()
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=request_id,
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            if decode_task:
                await utils.aio.gracefully_cancel(decode_task)
            await decoder.aclose()

class SynthesizeStream(tts.SynthesizeStream):
    """Streamed API using Uplift streaming endpoint (requires full text upfront)"""

    def __init__(
        self,
        *,
        tts: TTS,
        session: aiohttp.ClientSession,
        opts: _TTSOptions,
        conn_options: Optional[APIConnectOptions] = None,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts = tts
        self._opts = opts
        self._session = session
        self._input_text = ""
        self._input_ended = False
        self._input_received_event = asyncio.Event()

    def push_text(self, text: str) -> None:
        if not self._input_ended:
            self._input_text += text
            logger.debug("SynthesizeStream pushed text chunk")
        else:
             logger.warning("Attempted to push text after input ended.")


    def end_input(self) -> None:
        logger.debug("SynthesizeStream input ended.")
        self._input_ended = True
        self._input_received_event.set()

    async def _run(self) -> None:
        logger.debug("SynthesizeStream run waiting for input to end.")
        try:
            await self._input_received_event.wait()
        except asyncio.CancelledError:
            logger.warning("SynthesizeStream run cancelled while waiting for input.")
            raise

        if not self._input_text:
             logger.warning("SynthesizeStream run: No text received before input ended.")
             await self._event_ch.aclose()
             return


        logger.debug(f"SynthesizeStream run: Input finished. Synthesizing text: '{self._input_text[:50]}...'") # Log start of synthesis


        request_id = utils.shortuuid()
        data = {
            "voiceId": self._opts.voice,
            "text": self._input_text,
            "outputFormat": self._opts.output_format,
        }
        headers = {
            "Authorization": f"Bearer {self._opts.api_key}",
            "Content-Type": "application/json",
        }

        decoder = utils.codecs.AudioStreamDecoder(
            sample_rate=self._tts.sample_rate,
            num_channels=1,
        )
        decode_task: asyncio.Task | None = None
        emitter = tts.SynthesizedAudioEmitter(
                event_ch=self._event_ch,
                request_id=request_id,
        )

        try:
            logger.debug(f"SynthesizeStream run: Making POST request to {_stream_url(self._opts)} (req_id: {request_id})")
            async with self._session.post(
                _stream_url(self._opts),
                headers=headers,
                json=data,
            ) as resp:
                logger.debug(f"SynthesizeStream run: Received response status {resp.status} (req_id: {request_id})")
                resp.raise_for_status()

                if not resp.content_type.startswith("audio/"):
                    content = await resp.text()
                    logger.error(f"Uplift stream returned non-audio data (type: {resp.content_type}): {content} (req_id: {request_id})")
                    raise APIStatusError(f"Expected audio response, got {resp.content_type}", resp.status, request_id, content)


                logger.debug(f"SynthesizeStream run: Starting decode loop (req_id: {request_id})")
                async def _decode_loop():
                    chunk_count = 0
                    total_bytes = 0
                    try:
                        async for chunk in resp.content.iter_chunked(4096):
                            chunk_count += 1
                            total_bytes += len(chunk)
                            if not chunk:
                                logger.debug(f"SynthesizeStream decode: Received empty chunk, breaking loop (req_id: {request_id})")
                                break
                            decoder.push(chunk)
                        logger.debug(f"SynthesizeStream decode loop finished. Chunks: {chunk_count}, Total Bytes: {total_bytes} (req_id: {request_id})")
                    except Exception as e:
                         logger.error(f"SynthesizeStream decode loop error: {e} (req_id: {request_id})", exc_info=True)
                         raise
                    finally:
                        logger.debug(f"SynthesizeStream decode loop finally block: Ending decoder input (req_id: {request_id})")
                        decoder.end_input()

                decode_task = asyncio.create_task(_decode_loop())

                frame_count = 0
                async for frame in decoder:
                    frame_count += 1
                    emitter.push(frame)

                logger.debug(f"SynthesizeStream run: Finished iterating decoder. Total frames: {frame_count}. Flushing emitter (req_id: {request_id})")
                emitter.flush()
                logger.debug(f"SynthesizeStream run: Emitter flushed. Waiting for decode task. (req_id: {request_id})")

                await decode_task
                logger.debug(f"SynthesizeStream run: Decode task completed. (req_id: {request_id})")


        except asyncio.TimeoutError as e:
            logger.error(f"SynthesizeStream run: API Timeout Error (req_id: {request_id})", exc_info=True)
            raise APITimeoutError(f"API request timed out for request {request_id}") from e
        except aiohttp.ClientResponseError as e:
            body_text = await e.response.text() if hasattr(e, 'response') and e.response else None
            logger.error(f"SynthesizeStream run: API Status Error {e.status}: {e.message} - Body: {body_text} (req_id: {request_id})", exc_info=True)
            raise APIStatusError(
                message=f"{e.message} - Check Uplift API Key and Voice ID.",
                status_code=e.status,
                request_id=request_id,
                body=body_text,
            ) from e
        except Exception as e:
            logger.error(f"SynthesizeStream run: Unexpected Error (req_id: {request_id})", exc_info=True)
            raise APIConnectionError(f"An unexpected error occurred: {e}") from e
        finally:
            logger.debug(f"SynthesizeStream run: Main try block finished or exception occurred. Cleaning up decoder. (req_id: {request_id})")
            await decoder.aclose()
            logger.debug(f"SynthesizeStream run: Cleanup complete. (req_id: {request_id})")

def _synthesize_url(opts: _TTSOptions) -> str:
    return f"{opts.base_url}/synthesis/text-to-speech"

def _stream_url(opts: _TTSOptions) -> str:
    return f"{opts.base_url}/synthesis/text-to-speech/stream"