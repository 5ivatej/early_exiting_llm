import os, torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set environment variable for safety
os.environ.setdefault("HF_DISABLE_FLEX_ATTENTION", "1")

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map={"": device},
)

# Define early-exit classifier
class EarlyExitClassifier(nn.Module):
    def __init__(self, hidden_size, vocab_size, dtype=torch.float16):
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size, dtype=dtype)
        self.to(device)

    def forward(self, hidden_states):
        return self.linear(hidden_states)

# Wrap model with early-exit logic
class EarlyExitModelWrapper(nn.Module):
    def __init__(self, model, confidence_threshold=0.9, num_layers_to_check=None):
        super().__init__()
        self.model = model
        self.config = model.config
        self.confidence_threshold = confidence_threshold
        # Default to all layers if not specified
        num_layers_to_check = num_layers_to_check or len(model.model.layers)
        # Add classifiers for each layer, ensuring float16
        self.classifiers = nn.ModuleList([
            EarlyExitClassifier(self.config.hidden_size, self.config.vocab_size, dtype=torch.float16)
            for _ in range(num_layers_to_check)
        ])
        self.to(device)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Get hidden states from the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        hidden_states = outputs.hidden_states  # List of [batch, seq_len, hidden_size]

        # Process each layer for early exit
        for layer_idx, (h_state, classifier) in enumerate(zip(hidden_states[1:], self.classifiers)):
            # Compute logits for the last token
            logits = classifier(h_state[:, -1, :].to(torch.float16))  # Ensure float16
            probs = F.softmax(logits, dim=-1)  # [batch, vocab_size]
            max_prob = probs.max(dim=-1)[0]  # Max probability for confidence

            # Check if we can exit early
            if max_prob >= self.confidence_threshold:
                return {
                    "logits": logits,
                    "exit_layer": layer_idx + 1,
                    "hidden_states": h_state,
                }

        # Fallback to final layer output
        final_hidden = self.model.model.norm(hidden_states[-1].to(torch.float16))  # Apply final layer norm
        final_logits = self.model.lm_head(final_hidden[:, -1, :])
        return {
            "logits": final_logits,
            "exit_layer": len(hidden_states) - 1,
            "hidden_states": hidden_states[-1],
        }

# Wrap the model
wrapped_model = EarlyExitModelWrapper(model, confidence_threshold=0.9)

# Custom generation function to handle early exits
@torch.no_grad()
def generate_with_early_exit(prompt, model, tokenizer, max_new_tokens=64, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]  # Keep as torch.long
    attention_mask = inputs.get("attention_mask", None)  # Keep default type (usually torch.long)
    generated_ids = input_ids.clone()

    for _ in range(max_new_tokens):
        outputs = model(input_ids=generated_ids, attention_mask=attention_mask)
        logits = outputs["logits"] / temperature

        # Apply top-p sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float("-inf")

        # Sample next token
        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        
        # Append to generated sequence
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        
        # Update attention mask
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=-1)
        
        # Check for EOS token
        if next_token_id.item() == tokenizer.eos_token_id:
            break
        
        print(f"Exited at layer {outputs['exit_layer']} for token {next_token_id.item()}")

    return generated_ids

# Run inference
prompt = "### User:\nExplain early-exit for LLMs in one tweet."
generated_ids = generate_with_early_exit(
    prompt,
    wrapped_model,
    tokenizer,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
)
output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(output)