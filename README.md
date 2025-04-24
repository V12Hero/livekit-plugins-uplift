# LiveKit Plugins - Uplift

This plugin provides TTS integration for [Uplift AI](https://upliftai.org) into the LiveKit Agent Framework.

## Installation

```bash
pip install livekit-plugins-uplift
```

## Pre-requisites

You'll need an API key from UpliftAI. It can be set as an environment variable: `UPLIFT_AI_API_KEY`

## Urdu Script Requirements

The Uplift TTS is designed to work with Urdu script. The TTS should only receive text in Urdu script - not Roman Urdu. This can be achieved via basic prompt engineering to ensure your LLM generates output in proper Urdu script.

### Example

```python
from livekit.plugins import uplift

async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You're an Urdu speaking livebot on a LiveKit call. "
            "No matter what, you should generate Urdu responses, in urdu script, no matter what. "
            "It should not be in Roman Urdu, it should be in Urdu script. "
        ),
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    assistant = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=openai.STT(language="ur"),
        llm=openai.LLM(),
        tts=uplift.TTS(voice="v_30s70t3a"),
        chat_ctx=initial_ctx,
        fnc_ctx=fnc_ctx,
    )
    assistant.start(ctx.room)
    await assistant.say("ہیلو، آپ کیسے ہیں؟", allow_interruptions=True)
```
