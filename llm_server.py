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
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.prompt}
            ]
            logger.info(f"📝 Messages before chat template: {len(messages)} messages (system + user)")
            
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

            eos_ids = _build_eos_ids(tokenizer)

            generation_kwargs = {
                **inputs,
                "max_new_tokens": request.max_new_tokens,
                "min_new_tokens": request.min_new_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "repetition_penalty": request.repetition_penalty,
                "do_sample": request.do_sample,
                "no_repeat_ngram_size": 4,
                "streamer": streamer,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if eos_ids is not None:
                generation_kwargs["eos_token_id"] = eos_ids

            logger.info(f"🚀 Starting generation with kwargs: max_new_tokens={request.max_new_tokens}, temp={request.temperature}")
            
            thread = threading.Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
            thread.start()

            full_response = []
            pending = ""
            yield f"data: {json.dumps({'event': 'start'}, ensure_ascii=False)}\n\n"

            for i, token_text in enumerate(streamer):
                pending += token_text

                # Stop at explicit marker if present.
                if STOP_MARKER in pending:
                    before_marker = pending.split(STOP_MARKER, 1)[0]
                    if before_marker:
                        full_response.append(before_marker)
                        yield f"data: {json.dumps({'token': before_marker}, ensure_ascii=False)}\n\n"
                    logger.info("🛑 Stop marker detected, ending stream early.")
                    pending = ""
                    break

                # Emit everything except a small tail to safely detect marker across chunk boundaries.
                keep_tail = max(0, len(STOP_MARKER) - 1)
                if len(pending) > keep_tail:
                    emit_text = pending[:-keep_tail]
                    pending = pending[-keep_tail:]
                    if emit_text:
                        full_response.append(emit_text)
                        yield f"data: {json.dumps({'token': emit_text}, ensure_ascii=False)}\n\n"

                if i % 10 == 0:
                    logger.info(f"📤 Streamed {i} tokens...")

            # Flush any remaining non-marker tail.
            if pending and STOP_MARKER not in pending:
                cleaned = pending.replace(STOP_MARKER, "")
                if cleaned:
                    full_response.append(cleaned)
                    yield f"data: {json.dumps({'token': cleaned}, ensure_ascii=False)}\n\n"

            thread.join(timeout=0.2)
            if thread.is_alive():
                logger.info("ℹ️ Generation thread still running in background after early stop.")
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
