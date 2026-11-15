
import re

# Import the Data files
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])

# Split them away from punctuation
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])

# Create a sorted set from the text remaining. 
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)


# Give each vocabulary an interger label
vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

# Tokenize, or the process of mapping each word (word fragment) to a number so as to use this as an tensor later for LLM training. So "The large fox" would be turned into something like: [123, 222,1, 889902903102938102382] Just as an example shape of the data.   
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab            #1
        self.int_to_str = {i:s for s,i in vocab.items()}        #2

    def encode(self, text):         #3
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):         #4
        text = " ".join([self.int_to_str[i] for i in ids]) 

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)    #5
        return text
    
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," 
       Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))


# Adding in the unknown and end of text tokens, otherwise we would see issues with text where we have not seen the tokens, eg, its outside of the training data. 
# Issue here mainly is that we would get a key error on the tokenizer.encode calls
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer, token in enumerate(all_tokens)}

print(len(vocab.items()))
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)',text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int
                        else"<|unk|>" for item in preprocessed]
        
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text ="".join([self.int_to_str[i]for i in ids])
        
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)   
        return text

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.decode(tokenizer.encode(text)))


# So typically people use more advanced tokenization schemes, like bitwise pair encoding. Which we can simply import here. Basically it assigns codes to sections of text. 
# Like: foobar -> [fo][o][bar] Something like this. This allows it to understand any text in tokens. 
from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
# Basic exercise
cooltext = '“Akwirw'

ints = tokenizer.encode(cooltext)
print(ints)

print(tokenizer.decode(ints))

# Now we are going to work with the text, and then set it up to be future predictors. 
# Effectively this feels somewhat like a linked list, like 2->201,201->5555 etc... 
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]

context_size = 4         #1
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")


for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)

desired = enc_sample[i]
print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

import torch 
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)    #1

        for i in range(0, len(token_ids) - max_length, stride):     #2
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):    #3
        return len(self.input_ids)

    def __getitem__(self, idx):         #4
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")                         #1
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)   #2
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,     #3
        num_workers=num_workers     #4
    )

    return dataloader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)      #1
first_batch = next(data_iter)
print(first_batch)

second_batch = next(data_iter)
print(second_batch)
# Exercise, create different cases
diffLoader = create_dataloader_v1(raw_text, batch_size=1, max_length=8, stride=4, shuffle=False)
first_iter = iter(diffLoader)
new_batch = next(first_iter)
print(new_batch)
second_new_batch = next(first_iter)
print(second_new_batch)

### End exercise 
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4,
    shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

# Embedding creation process 
input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)

print(embedding_layer(torch.tensor([3])))

print(embedding_layer(input_ids))


#  - Positonal Embedding work - 

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
   stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
# 8x4 matrix, 8 text samples, 4 tokens each
print("\nInputs shape:\n", inputs.shape)

# Embedding creation
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)


context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)


## End of Chapter 2 Notes 

# LLMs must be fed data that is converted into tokens via a "Vocabulary" 
# These tokens are then transformed into an embedding, where there position as an embeeding is added to it as well
# Embeddings are required because the token's Id itself if used for training would not allow updates to the weight or value of the token
# relative to other tokens around it. 

# Tokenization schemes are varied, the simple one we have here, and the bytewise pair encoding used for GPT2. There are many ways to slice this.
# We must end up with a way to map units of data to an int, and then that int to an array of numbers for its value as an embedding.

# During the tokenziation process, we want to create an input and target pair for training. 
# This approach we used is called the "sliding window" approach, there are probably others. 
# See the following output from claude on the other approaches

'''
  Here are the main ways to build training pairs, each with a tiny example using the sentence "the cat sat on the mat".

  Pretraining (learning language from raw text)

  1. Next-token prediction (what you have now). Predict the next token from everything before it. Used by GPT, Llama, Claude.
  input:  the cat sat on the
  target: cat sat on the mat

  2. Masked language modeling (BERT). Hide random tokens and predict them, using context on both sides. Good for understanding tasks like classification
  and search, not for generating text.
  input:  the [MASK] sat on the [MASK]
  target: cat, mat

  3. Span corruption (T5). Like masking, but whole chunks are hidden behind placeholders and the model writes the missing chunks back out.
  input:  the <X> on the mat
  target: <X> cat sat

  4. Fill-in-the-middle (code models). Cut out a middle piece, move it to the end, and train on next-token prediction as usual. This teaches
  autocomplete in the middle of a file.
  input:  <prefix> the cat <suffix> the mat <middle>
  target: sat on

  5. Multi-token prediction. At each position, predict the next several tokens, not just one. Some recent models (e.g. DeepSeek-V3) add this as an extra
  training signal.
  at "the" → predict: cat, sat

  6. Skip-gram / CBOW (word2vec, older). Predict nearby words from a center word, or the reverse. This trains only an embedding table, with no deep
  model.
  input: sat  →  targets: cat, on

  After pretraining (teaching the model to behave)

  7. Instruction tuning / SFT. Pairs of prompt → good answer. It's still next-token prediction, but loss is counted only on the answer tokens.
  input:  "Translate to French: cat"
  target: "chat"

  8. Preference pairs (RLHF, DPO). One prompt with two answers: a better one and a worse one. The model learns to favor the better answer.
  prompt: "Explain gravity"
  chosen: clear answer   rejected: rambling answer

  For embedding / search models

  9. Contrastive pairs (sentence embedders, CLIP). Pairs of things that belong together get pulled close in vector space, and mismatched pairs get
  pushed apart. This is how vector-search embeddings are trained.
  positive: ("how to bake bread", "bread recipe")
  negative: ("how to bake bread", "car repair guide")

  The pattern

  All of these do the same thing: take data, hide or hold back part of it, and make the model predict the hidden part. What changes is what gets hidden,
  and that decides what the model gets good at:

  - Hide the future, and you get a generator (GPT).
  - Hide random words, and you get an understander (BERT).
  - Hide which pairs match, and you get a search embedder.
'''