import os
import json
import base64
import asyncio
import traceback
import websockets
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Connect
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Audio Transcoders (Twilio 8kHz mu-law <-> Gemini PCM) ---
def _ulaw2lin(u):
    u = ~u & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    sample = ((mantissa << 3) + 132) << exponent
    sample -= 132
    return -sample if sign else sample

MULAW_TO_PCM16 = [_ulaw2lin(i) for i in range(256)]

def mulaw_to_pcm16_upsample(mulaw_bytes):
    pcm = bytearray()
    for b in mulaw_bytes:
        s = MULAW_TO_PCM16[b]
        low, high = s & 0xFF, (s >> 8) & 0xFF
        pcm.extend((low, high, low, high))
    return bytes(pcm)

def pcm24k_to_mulaw8k(pcm_bytes):
    mulaw = bytearray()
    for i in range(0, len(pcm_bytes) - 5, 6):
        sample = int.from_bytes(pcm_bytes[i:i+2], byteorder="little", signed=True)
        sign = 0x80 if sample < 0 else 0
        if sample < 0:
            sample = -sample
        if sample > 32635:
            sample = 32635
        sample += 0x84
        exponent = 7
        for exp in range(7, -1, -1):
            if sample & (1 << (exp + 7)):
                exponent = exp
                break
        mantissa = (sample >> (exponent + 3)) & 0x0F
        mulaw.append((~(sign | (exponent << 4) | mantissa)) & 0xFF)
    return bytes(mulaw)

# --- FastAPI Server ---
app = FastAPI()

@app.api_route("/incoming-call", methods=["GET", "POST"])
@app.api_route("/outbound-call", methods=["GET", "POST"])
async def call_instructions(request: Request):
    host = request.headers.get("host")
    vr = VoiceResponse()
    vr.say("Connecting to your assistant.")
    connect = Connect()
    connect.stream(url=f"wss://{host}/media-stream")
    vr.append(connect)
    vr.pause(length=30)
    return Response(content=str(vr), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    await websocket.accept()
    print("Twilio WebSocket connected!")
    stream_sid = None

    gemini_url = (
        f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
        f"?key={GEMINI_API_KEY}"
    )

    try:
        async with websockets.connect(gemini_url) as gemini_ws:
            print("Connected to Gemini Live WebSocket!")

            setup_message = {
                "setup": {
                    "model": "models/gemini-2.0-flash-exp",
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                        "speechConfig": {
                            "voiceConfig": {
                                "prebuiltVoiceConfig": {"voiceName": "Puck"}
                            }
                        }
                    },
                    "systemInstruction": {
                        "parts": [{"text": "You are a warm, polite AI phone assistant. Keep all responses very brief, natural, and under 1-2 sentences."}]
                    }
                }
            }
            await gemini_ws.send(json.dumps(setup_message))
            setup_ack = await gemini_ws.recv()
            print("Gemini setup acknowledged:", setup_ack)

            async def from_twilio():
                nonlocal stream_sid
                try:
                    while True:
                        msg = await websocket.receive_text()
                        data = json.loads(msg)
                        if data["event"] == "start":
                            stream_sid = data["start"]["streamSid"]
                            print(f"Call started with SID: {stream_sid}")
                        elif data["event"] == "media":
                            raw_mulaw = base64.b64decode(data["media"]["payload"])
                            pcm16 = mulaw_to_pcm16_upsample(raw_mulaw)
                            gemini_msg = {
                                "