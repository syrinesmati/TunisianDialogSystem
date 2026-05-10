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
from fastapi.responses import JSONResponse, StreamingResponse
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "CohereLabs/aya-expanse-8b")
FINETUNED_MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    str(Path(__file__).resolve().parent / "outputs" / "checkpoints" / "aya-expanse-8b-tunisian-sft"),
)

STOP_MARKER = "<END_TIGANI>"

SYSTEM_PROMPT = """أنت "التيجاني"، مساعد ذكاء اصطناعي تونسي 100%. تحكي مع العباد بلهجة تونسية دارجة.

### القاعدة الأهم (CRITICAL):
ممنوع منعاً باتاً تطرح سؤال على روحك وتجاوب عليه. ممنوع تواصل الكلام بعد ما تجاوب المستخدم. بمجرد ما تعطي الإجابة المطلوبة، قص الكلام فوراً (STOP generating).

### القواعد الصارمة:
1. **اللغة:** احكي بالتونسي الدارجة فقط. ممنوع منعاً باتاً استعمال اللغة العربية الفصحى أو الكلمات "البيضاء". استعمل قاموسنا (مثال: "إي"، "نجم"، "فمة"، "برشة"، "كيما").
2. **جاوب وقص:** كي تجاوب على قد السؤال، اسكت وديريكت قص الكلام. ممنوع تزيد حرف واحد بعد الإجابة، وممنوع تولد نصوص وهمية أو أخبار قديمة.
3. **ممنوع الهلوسة:** إذا سألوك على معلومة حينة (طقس، أخبار) أو حاجة ما تعرفهاش، قول ديريكت: "ما عنديش معلومة" أو "ما نعرفش". لا تخترع إجابات ولا تجبد مواضيع سياسية بايتة.
4. **ادخل في الموضوع:** ما تعاودش سؤال المستخدم، ما تعتذرش بلا سبب، وما تفسرش شكونك إلا إذا سألك. جاوب بوضوح واختصار.
5. **الشخصية:** أنت ذكي، خفيف روح، ومتربي. تحكي بلهجة تونسية يفهموها التوانسة الكل.

### أمثلة للالتزام بالنمط:
المستخدم: عالسلامة تحكي تونسي؟
التيجاني: عالسلامة! إي نعم، نحكي تونسي ونفهمك بالباهي. شنوة حاجتك؟

المستخدم: شنوة الطقس اليوم؟
التيجاني: سامحني، ما عنديش معلومة حينة على الطقس توة.

المستخدم: شكونك؟
التيجاني: أنا التيجاني، مساعدك التونسي. موجود هنا باش نعاونك في اللي تحب بالتونسي.

قاعدة تقنية نهائية: كي تكمل إجابتك مباشرة، كتب الرمز هذا وحدو في الآخر: <END_TIGANI>
"""

def _resolve_device_dtype() -> torch.dtype:
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _load_model_from_dir(model_dir: Path, base_model_id: str, dtype: torch.dtype):
    """Load either a merged model directory or a PEFT adapter directory."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # If the directory contains a full model config, load it directly.
    if (model_dir / "config.json").exists():
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        return tokenizer, model

    # Otherwise, treat it as a PEFT adapter on top of the base model.
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        str(model_dir),
        device_map="auto",
        local_files_only=True,
    )
    return tokenizer, model


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=220, ge=1, le=768)
    min_new_tokens: int = Field(default=24, ge=0, le=256)
    temperature: float = Field(default=0.5, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0)
    do_sample: bool = Field(default=False)


def _build_eos_ids(tokenizer) -> list[int] | int | None:
    eos_ids: list[int] = []
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos_ids.append(int(tokenizer.eos_token_id))

    # Aya/cohere chat models frequently use END_OF_TURN token.
    for token in ("<|END_OF_TURN_TOKEN|>", "<|eot_id|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            eos_ids.append(token_id)

    # Deduplicate while preserving order.
    unique_ids = list(dict.fromkeys(eos_ids))
    if not unique_ids:
        return None
    return unique_ids if len(unique_ids) > 1 else unique_ids[0]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("🔧 Loading model at startup...")
    
    dtype = _resolve_device_dtype()
    model_path = Path(FINETUNED_MODEL_DIR)

    logger.info(f"📦 Base model: {BASE_MODEL_ID}")
    logger.info(f"📦 Fine-tuned model dir: {FINETUNED_MODEL_DIR}")
    logger.info(f"🎯 Device dtype: {dtype}")

    tokenizer, model = _load_model_from_dir(model_path, BASE_MODEL_ID, dtype)
    logger.info("✅ Tokenizer loaded")
    logger.info("✅ Model loaded")

    model.eval()
    logger.info("✅ Model set to eval mode")

    app.state.tokenizer = tokenizer
    app.state.model = model
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
        """Stream response tokens as SSE data lines."""
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.prompt},
            ]
            logger.info("📝 Applying chat template for generation")

            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            inputs = {k: v.to(device) for k, v in inputs.items()}
            eos_ids = _build_eos_ids(tokenizer)

            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            generation_kwargs = {
                **inputs,
                "max_new_tokens": request.max_new_tokens,
                "min_new_tokens": request.min_new_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "repetition_penalty": request.repetition_penalty,
                "do_sample": request.do_sample,
                "no_repeat_ngram_size": 4,
                "pad_token_id": tokenizer.eos_token_id,
                "streamer": streamer,
            }
            if eos_ids is not None:
                generation_kwargs["eos_token_id"] = eos_ids

            logger.info("🚀 Starting streaming generation")
            
            # Run generation in background thread
            thread = threading.Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
            thread.start()

            # Stream tokens as SSE data lines
            full_response = []
            yield f"data: {json.dumps({'event': 'start'}, ensure_ascii=False)}\n\n"

            for token_text in streamer:
                if STOP_MARKER in token_text:
                    before_marker = token_text.split(STOP_MARKER, 1)[0]
                    if before_marker:
                        full_response.append(before_marker)
                        yield f"data: {json.dumps({'token': before_marker}, ensure_ascii=False)}\n\n"
                    logger.info("🛑 Stop marker detected, ending stream")
                    break

                if token_text:
                    full_response.append(token_text)
                    yield f"data: {json.dumps({'token': token_text}, ensure_ascii=False)}\n\n"

            thread.join(timeout=0.2)
            response_text = "".join(full_response)
            logger.info(f"✨ Streaming complete: {len(full_response)} tokens")
            yield f"data: {json.dumps({'event': 'end', 'response': response_text}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"❌ Error during streaming: {e}", exc_info=True)
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
