# char_lm

A character-level language model that generates plausible human names, built with PyTorch.

### What is this project about ?

This is a learning project I built to understand how language models work. The model learns the statistical patterns of English names — which characters tend to follow which — and then generates new names that sound plausible but don't exist.

The core concept is the same as large language models (GPT, Claude, etc.): given a sequence of tokens, predict the next one. Here the "tokens" are individual characters and the "text" is a name, but the architecture and training pipeline are fundamentally identical.

### How does it work ?

The model sees a sliding window of characters and predicts the next character. For example, given the name "emma":

```
"....." → "e"    (predict the first letter)
"....e" → "m"
"...em" → "m"
"..emm" → "a"
".emma" → "."    (predict end of name)
```

The architecture is an MLP (multi-layer perceptron) with a learned embedding layer:

- **Embedding** — each of the 27 characters (a–z + start/end token) is mapped to a learned vector of dimension 20
- **Hidden layers** — one or more Dense + ReLU layers
- **Output layer** — Dense(27) producing logits, converted to a probability distribution via softmax

Training uses cross-entropy loss and the Adam optimizer.

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

### Knowledge distillation experiment

I implemented knowledge distillation to explore whether a small model can learn better by imitating a large model's soft probability distribution rather than training on hard labels alone.

**Setup:**
- Teacher: 2 hidden layers of 512 neurons each
- Student: 1 hidden layer of 64 neurons
- Distillation temperature T = 2, mixing weight α = 0.8
- Both models share the same embedding dimension (20) and context size (5)

The distillation loss combines the standard cross-entropy with the true label and the KL divergence between the teacher's and student's softened distributions:

```
L = α · CE(student, label) + (1 − α) · T² · KL(teacher_soft ‖ student_soft)
```

**Results** (evaluated with cross-entropy on the full dataset):

| Model | Eval loss |
|---|---|
| Teacher [512, 512] | 1.96 |
| Student [64] (hard labels) | 2.08 |
| Student [64] (distilled) | 2.09 |

**Finding:** Distillation did not improve the student's performance. The likely explanation is that with only 27 characters in the vocabulary, the teacher's soft distribution carries very little "dark knowledge" beyond the hard label. In real-world distillation (e.g. GPT-4 → GPT-4-mini), the vocabulary has ~100,000 tokens, so the teacher's relative rankings of unlikely tokens encode rich semantic structure. With 27 characters, there simply isn't enough structure to transfer. The distilled student also showed a significant train-eval gap (1.69 training vs 2.09 eval), suggesting it was overfitting to the teacher's idiosyncrasies rather than learning generalizable patterns.

### Training data

The model trains on ~32,000 common names from [Andrej Karpathy's makemore dataset](https://github.com/karpathy/makemore). The dataset is included in the repo for reproducibility but can also be downloaded:

```bash
curl -o names.txt https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
```

### How to run

```bash
pip install torch
python train_teacher.py       # train the large teacher model
python train_student.py       # train the small student on hard labels
python distill.py             # train the small student via distillation
python evaluate.py            # compare all three models
python main.py                # generate new names
```

### Project structure

- `config.py` — hyperparameters (context size, embedding dimension, hidden dims, learning rate, distillation temperature, etc.)
- `data.py` — character encoding, context window extraction from names
- `model.py` — flexible MLP architecture with learned embeddings, supporting arbitrary hidden layer configurations
- `train.py` — training loop, distillation loss function, and distillation training loop
- `train_teacher.py` — trains the teacher model, saves weights to `teacher.pt`
- `train_student.py` — trains the student model on hard labels, saves to `student.pt`
- `distill.py` — trains the student via distillation using the teacher's outputs, saves to `student_distill.pt`
- `evaluate.py` — compares all three models on the same data with the same metric
- `main.py` — generates new names from a trained model

### Dependencies

- PyTorch
