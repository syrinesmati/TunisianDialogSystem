from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "CohereLabs/aya-expanse-8b")
ADAPTER_DIR = os.environ.get(
    "ADAPTER_DIR",
    str(Path(__file__).resolve().parent / "outputs" / "checkpoints" / "aya-expanse-8b-cpt-tunisian"),
)


def _resolve_device_dtype() -> torch.dtype:
    return torch.float16 if torch.cuda.is_available() else torch.float32


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=160, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.0)
    do_sample: bool = Field(default=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("🔧 Loading model at startup...")
    
    dtype = _resolve_device_dtype()
    adapter_path = Path(ADAPTER_DIR)

    logger.info(f"📦 Base model: {BASE_MODEL_ID}")
    logger.info(f"📦 Adapter dir: {ADAPTER_DIR}")
    logger.info(f"🎯 Device dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("✅ Tokenizer loaded")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    logger.info("✅ Base model loaded")

    if adapter_path.exists():
        logger.info(f"📂 Loading adapter from local path: {adapter_path}")
        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
            device_map="auto",
            local_files_only=True,
        )
    else:
        logger.info(f"🌐 Loading adapter from HF Hub: {ADAPTER_DIR}")
        model = PeftModel.from_pretrained(
            base_model,
            ADAPTER_DIR,
            device_map="auto",
        )

    model.eval()
    logger.info("✅ Adapter loaded and model set to eval mode")

    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.base_model = base_model
    app.state.device_dtype = dtype
    
    logger.info("🎉 Server ready! Waiting for requests...")
    yield
    logger.info("👋 Shutting down...")


app = FastAPI(title="Tunisian LLM Server", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    logger.info("💚 Health check request")
    return {"status": "ok"}


@app.post("/generate")
def generate(request: GenerateRequest):
    tokenizer = app.state.tokenizer
    model = app.state.model

    logger.info(f"📥 Received request: prompt='{request.prompt[:50]}...' max_tokens={request.max_new_tokens}")

    def event_stream():
        try:
            messages = [{"role": "user", "content": request.prompt}]
            logger.info(f"📝 Messages before chat template: {messages}")
            
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            
            logger.info(f"✅ Chat template applied. Input IDs shape: {inputs['input_ids'].shape}")

            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            inputs = {k: v.to(device) for k, v in inputs.items()}
            logger.info(f"🎯 Inputs moved to device: {device}")

            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            generation_kwargs = {
                **inputs,
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "do_sample": request.do_sample,
                "streamer": streamer,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }

            logger.info(f"🚀 Starting generation with kwargs: max_new_tokens={request.max_new_tokens}, temp={request.temperature}")
            
            thread = threading.Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
            thread.start()

            full_response = []
            yield f"data: {json.dumps({'event': 'start'}, ensure_ascii=False)}\n\n"

            for i, token_text in enumerate(streamer):
                full_response.append(token_text)
                if i % 10 == 0:
                    logger.info(f"📤 Streamed {i} tokens...")
                yield f"data: {json.dumps({'token': token_text}, ensure_ascii=False)}\n\n"

            thread.join()
            response_text = "".join(full_response)
            logger.info(f"✨ Generation complete! Total tokens: {len(full_response)}")
            logger.info(f"📄 Final response: '{response_text[:100]}...'")
            
            yield f"data: {json.dumps({'response': response_text}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'event': 'end'}, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            logger.error(f"❌ Error during generation: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
