from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

app = FastAPI(title="Disaster Forecast Explanation API (Adapted)")

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "lab8_adapted_model"

# Load Model
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

adapted_base = AutoModelForCausalLM.from_pretrained(
    DEFAULT_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)
if os.path.exists(OUTPUT_DIR):
    model = PeftModel.from_pretrained(adapted_base, OUTPUT_DIR)
else:
    model = adapted_base # Fallback if model not found
model.eval()

class QueryRequest(BaseModel):
    instruction: str
    input: str

class QueryResponse(BaseModel):
    explanation: str

@app.post("/generate", response_model=QueryResponse)
def generate_explanation(request: QueryRequest):
    prompt = f"Instruction: {request.instruction}\nInput: {request.input}\nResponse:"
    tokens = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        device = next(model.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        out = model.generate(**tokens, max_new_tokens=100, do_sample=False)

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    explanation = decoded.split("Response:", 1)[-1].strip() if "Response:" in decoded else decoded.strip()
    
    return QueryResponse(explanation=explanation)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
