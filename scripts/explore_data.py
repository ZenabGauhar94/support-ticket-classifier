from datasets import load_dataset

# Load the Banking77 dataset from Hugging Face
dataset = load_dataset("banking77")

print("Train examples:", len(dataset["train"]))
print("Test examples:", len(dataset["test"]))
print("\nExample ticket:")
print(dataset["train"][0])

print("\nAll 77 possible intent labels:")
label_names = dataset["train"].features["label"].names
for i, name in enumerate(label_names):
    print(i, name)