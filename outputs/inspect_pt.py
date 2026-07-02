import torch
import json

# Load the Step 3 intervention file
pt_file = "m=CCBD-Interns--multiligual-arithmetic-final_model_p=200_int=True_seed=42.pt"
data = torch.load(pt_file, weights_only=False)

print("=" * 60)
print("STEP 3 RESULTS OVERVIEW")
print("=" * 60)

# Get metadata
metadata = data["metadata"]
print(f"\nModel: {metadata['model_path']}")
print(f"Prompts tested: {metadata['num_prompts']}")
print(f"Layers: {metadata['layer_indices']}")
print(f"Attention heads: {metadata['num_attention_heads']}")

# Baseline accuracy
baseline_acc = data["baseline_accuracy"]
print(f"\n📊 Baseline Accuracy: {baseline_acc:.1%}")

# Ablations
ablations = data["ablations"]
print(f"💥 Total Components Tested: {len(ablations)}")

print("\n" + "=" * 60)
print("TOP 10 MOST IMPORTANT COMPONENTS")
print("=" * 60)

# Sort by impact
ablations_sorted = sorted(ablations, key=lambda x: x["accuracy_drop"], reverse=True)

for i, ab in enumerate(ablations_sorted[:10], 1):
    layer = ab["layer_idx"]
    feat_type = ab["type"]
    local_idx = ab["local_idx"]
    drop = ab["accuracy_drop"]
    final_acc = ab["accuracy"]
    
    # Create readable label
    if feat_type == "mlp":
        label = f"L{layer}MLP{local_idx}"
    else:
        label = f"L{layer}HEAD{local_idx}"
    
    print(f"{i:2d}. {label:12s} | Drop: {drop:6.1%} | Final Acc: {final_acc:5.1%}")

print("\n" + "=" * 60)