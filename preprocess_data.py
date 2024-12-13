from datasets import load_dataset
from transformers import GPT2Tokenizer

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set or add a pad token
tokenizer.pad_token = tokenizer.eos_token  # or use `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`

# Load the tiny_shakespeare dataset
dataset = load_dataset('Elriggs/openwebtext-100k')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

# Apply the tokenize function to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=8)

# Convert the tokenized dataset to a pandas DataFrame
tokenized_df = tokenized_dataset['train'].to_pandas()  # Use 'train', 'test', or 'validation' as needed
#tokenized_df = tokenized_df.head(15000)
tokenized_df['input_ids'] = tokenized_df['input_ids'].apply(lambda arr: list([float(x) for x in arr]))

# Save the DataFrame to a CSV file
tokenized_df.to_csv('openwebtext6.csv', index=False)
