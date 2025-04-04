from dataclasses import dataclass
from typing import Literal

TTSEncoding = Literal[
    "WAV_22050_16",
    "WAV_22050_32",
    "MP3_22050_32",
    "MP3_22050_64",
    "MP3_22050_128",
    "OGG_22050_16",
    "ULAW_8000_8",
]

TTSDefaultVoiceId = "v_30s70t3a"
AvailableVoices = [ "v_kwmp7zxt", "v_yypgzenx", "v_30s70t3a" ]

