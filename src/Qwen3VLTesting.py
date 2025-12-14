import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

print("=== 0. Detecting device ===")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model_id = "Qwen/Qwen3-VL-2B-Instruct" 

print("\n=== 1. Loading model (no device_map) ===")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
model.to(device)
model.eval()
print("Model loaded and moved to", device)

print("\n=== 2. Loading processor ===")
processor = AutoProcessor.from_pretrained(model_id)
print("Processor loaded.")

print("\n=== 3. Building messages ===")
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
print("Messages ready.")

print("\n=== 4. Applying chat template ===")
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
print("Template applied.")

print("\n=== 5. Cleaning token_type_ids & moving inputs to device ===")
inputs.pop("token_type_ids", None)
# if `inputs` is a BatchEncoding, this works:
inputs = inputs.to(device)
print("Inputs moved to", device)

print("\n=== 6. Running generation (no_grad) ===")
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.1,
    )
print("Generation complete.")

print("\n=== 7. Trimming prompt tokens ===")
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
]
print("Trimming done.")

print("\n=== 8. Decoding ===")
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)
print("Decoding done.")

print("\n=== 9. Final output ===")
print(output_text[0])
