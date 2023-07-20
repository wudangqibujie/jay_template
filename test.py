from datasets import load_dataset

# Print all the available datasets
from huggingface_hub import list_datasets
print([dataset.id for dataset in list_datasets()])


