import torch
from model import GPTConfig, GPT

# Load the config from train_shakespeare_char
config = GPTConfig(
    block_size=512,  
    n_layer=6,
    n_head=6,
    n_embd=576,
    dropout=0.2,
    block_type='cortex'  # default, cortex, or cortex_x
)

print("\n\nCreating model with config:")
print(f"block_type: {config.block_type}")
print(f"n_embd: {config.n_embd}")
print(f"n_head: {config.n_head}")
print(f"n_layer: {config.n_layer}")

# Create the model
model = GPT(config)

# Print the model structure
print("\nModel summary:")
for name, module in model.named_children():
    print(f"{name}: {type(module)}")
    if name == "transformer":
        for subname, submodule in module.named_children():
            if subname == "h":
                print(f"  {subname}: {type(submodule)} with {len(submodule)} blocks")
                for i, block in enumerate(submodule):
                    print(f"    block {i}: {type(block)}") 