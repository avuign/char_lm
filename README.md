# char_lm

A character-level language model that generates plausible human names, built with PyTorch.

### What is this project about ?

This is a learning project I built to understand how language models work. The model learns the statistical patterns of English names — which characters tend to follow which — and then generates new names that sound plausible but don't exist.

The core concept is the same as large language models (GPT, Claude, etc.): given a sequence of tokens, predict the next one. Here the "tokens" are individual characters and the "text" is a name, but the architecture and training pipeline are fundamentally identical.

### How does it work ?

The model sees a sliding window of characters and predicts the next character. For example, given the name "emma":

```
"..." → "e"    (predict the first letter)
"..e" → "m"
".em" → "m"
"emm" → "a"
"mma" → "."    (predict end of name)
```

The architecture is an MLP (multi-layer perceptron) with a learned embedding layer:

- **Embedding** — each of the 27 characters (a–z + start/end token) is mapped to a learned vector of dimension 30
- **Hidden layer** — Dense(512) + ReLU
- **Output layer** — Dense(27) producing a probability distribution over the next character

Training uses cross-entropy loss and the Adam optimizer. After 40 epochs the model reaches a loss of ~1.47, meaning it effectively narrows down the next character to about 4 plausible options on average.

### Example output

```
axel
aryen
james
mavory
heitham
braham
daedyn
aslea
mose
```

### Training data

The model trains on ~32,000 common names from [Andrej Karpathy's makemore dataset](https://github.com/karpathy/makemore).

### How to run

```bash
curl -o names.txt https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
pip install torch
python main.py
```

### Project structure

- `config.py` — hyperparameters (context size, embedding dimension, learning rate, etc.)
- `data.py` — character encoding, context window extraction from names
- `model.py` — embedding + MLP architecture
- `train.py` — training loop with cross-entropy loss and Adam
- `main.py` — orchestrates training and generates new names

### Dependencies

- PyTorch
