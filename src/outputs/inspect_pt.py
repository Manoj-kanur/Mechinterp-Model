import torch

# Replace with your actual file path
file_path = "m=CCBD-Interns--multiligual-arithmetic-final_model_p=20_int=False_seed=42.pt"

data = torch.load(file_path, map_location="cpu")

print(type(data))

if isinstance(data, dict):
    print("\nTop-level keys:")
    print(data.keys())

    for key, value in data.items():
        print(f"\n===== {key} =====")
        print(type(value))

        if isinstance(value, dict):
            print(value.keys())

        elif isinstance(value, list):
            print(f"Length: {len(value)}")
            if len(value) > 0:
                print("First item:")
                print(value[0])