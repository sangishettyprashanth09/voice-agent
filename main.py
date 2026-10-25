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

# Read the environment variable by its key name
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Audio Transcoders (Twilio 8kHz mu-law <-> Gemini PCM 16kHz/24kHz) ---
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
        # Upsample 8kHz -> 16kHz by sample doubling
        pcm.extend((low, high, low, high))
    return bytes(pcm)

def pcm24k_to_mulaw8k(pcm_bytes):
    # Downsample 24kHz -> 8kHz (decimate by 3)
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

# --- FastAPI App ---
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
    vr.pause(length=60)
    return Response(content=str(vr), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    await websocket.accept()
    print("Twilio WebSocket connected!")
    stream_sid = None

    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY environment variable is missing!")
        await websocket.close()
        return

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
                        "parts": [{
                            "text": "You are a helpful, conversational phone assistant. Keep answers concise, natural, and under two sentences."
                        }]
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
                        event = data.get("event")

                        if event == "start":
                            stream_sid = data["start"]["streamSid"]
                            print(f"Call streaming started. SID: {stream_sid}")
                        elif event == "media":
                            raw_mulaw = base64.b64decode(data["media"]["payload"])
                            pcm16 = mulaw_to_pcm16_upsample(raw_mulaw)
                            gemini_msg = {
                                "realtimeInput": {
                                    "mediaChunks": [{
                                        "mimeType": "audio/pcm;rate=16000",
                                        "data": base64.b64encode(pcm16).decode("utf-8")
                                    }]
                                }
                            }
                            await gemini_ws.send(json.dumps(gemini_msg))
                        elif event == "stop":
                            print("Twilio sent stop event.")
                            break
                except Exception as e:
                    print(f"Twilio sender error: {e}")

            async def from_gemini():
                nonlocal stream_sid
                try:
                    async for raw_msg in gemini_ws:
                        resp = json.loads(raw_msg)
                        server_content = resp.get("serverContent", {})
                        model_turn = server_content.get("modelTurn", {})
                        parts = model_turn.get("parts", [])

                        for part in parts:
                            inline_data = part.get("inlineData", {})
                            if inline_data.get("mimeType", "").startswith("audio/pcm") and stream_sid:
                                pcm24_bytes = base64.b64decode(inline_data["data"])
                                mulaw_bytes = pcm24k_to_mulaw8k(pcm24_bytes)
                                twilio_payload = base64.b64encode(mulaw_bytes).decode("utf-8")
                                await websocket.send_text(json.dumps({
                                    "event": "media",
                                    "streamSid": stream_sid,
                                    "media": {"payload": twilio_payload}
                                }))

                        if server_content.get("interrupted") and stream_sid:
                            await websocket.send_text(json.dumps({
                                "event": "clear",
                                "streamSid": stream_sid
                            }))
                except Exception as e:
                    print(f"Gemini receiver error: {e}")

            await asyncio.gather(from_twilio(), from_gemini())

    except Exception as e:
        print(f"Connection error: {e}")
        traceback.print_exc()