import argparse
import torch
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument("--query", type=str, required=True)

word2int = {}
with open("vocab/token_vocab") as f:
    words = f.read().split()
    for (i, word) in enumerate(words):
        word2int[word] = i
    word2int["<UNK>"] = len(words)

with open("vocab/intent_vocab") as f:
    intent2int = f.read().split()
print("Loaded Vocabs")


def map_token_sequence_to_ints(sequence):
    mapped = []
    for word in sequence:
        if word not in word2int:
            word = "<UNK>"
        mapped.append(word2int[word])

    return mapped


model_inference = torch.load("saved_model.pkl")

args = parser.parse_args()

query = args.query
query = query.lower().strip()
query = query.split()
query = map_token_sequence_to_ints(query)
query = torch.tensor(query, dtype=torch.long)

model_inference.eval()
pred = model_inference(query)
pred_class = pred.argmax(dim=-1).numpy()[0]

print(intent2int[pred_class])