import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_name = "Zeanb/ticket-classifier-lora"

tokenizer = AutoTokenizer.from_pretrained(adapter_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(base_model, adapter_name)

def classify(ticket_text):
    prompt = (
        "Classify this customer support ticket into exactly one category.\n\n"
        f"Ticket: {ticket_text}\n\n"
        "Category:"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    result = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return result.strip()

if __name__ == "__main__":
    while True:
        text = input("\nEnter a support ticket (or 'quit'): ")
        if text.lower() == "quit":
            break
        print("Predicted category:", classify(text))