# char_lm

A character-level language model that generates plausible human names, built with PyTorch.

### What is this project about ?

This is a learning project I built to understand how language models work. The model learns the statistical patterns of English names — which characters tend to follow which — and then generates new names that sound plausible but don't exist.

The core concept is the same as large language models (GPT, Claude, etc.): given a sequence of tokens, predict the next one. Here the "tokens" are individual characters and the "text" is a name, but the architecture and training pipeline are fundamentally identical.

### How does it work ?

The model sees a sliding window of 5 characters and predicts the next character. For example, given the name "emma":

```
"....." → "e"    (predict the first letter)
"....e" → "m"
"...em" → "m"
"..emm" → "a"
".emma" → "."    (predict end of name)
```

The architecture is an MLP (multi-layer perceptron) with a learned embedding layer:

- **Embedding** — each of the 27 characters (a–z + start/end token) is mapped to a learned vector of dimension 30
- **Hidden layers** — one or more Dense + ReLU layers (configurable)
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

I implemented [knowledge distillation](https://arxiv.org/abs/1503.02531) to explore whether a small student model can learn better by imitating a large teacher model's soft probability distribution, rather than training on hard labels alone.

**Setup:**
- Teacher: 2 hidden layers of 512 neurons each
- Student: 1 hidden layer of 16 neurons
- Distillation temperature T = 2, mixing weight α = 0.2
- Both models share the same embedding (dim 30) and context size (5)

The distillation loss combines the standard cross-entropy with the KL divergence between the teacher's and student's softened output distributions:

```
L = α · CE(student, label) + (1 − α) · T² · KL(softmax(teacher/T) ‖ softmax(student/T))
```

**Results** (cross-entropy evaluated on the full dataset):

| Model | Parameters | Eval loss |
|---|---|---|
| Teacher [512, 512] | ~420k | 1.91 |
| Student [16] (hard labels) | ~5k | 2.25 |
| Student [16] (distilled) | ~5k | 2.36 |

**Finding:** Distillation did not improve the student — the distilled version performed slightly worse than the one trained on hard labels. After experimenting with various hyperparameters (temperature, mixing weight, student size, epochs), the result was consistent. The most likely explanation is that with only 27 characters in the vocabulary, the teacher's soft distribution carries very little information beyond the hard label. In production distillation (e.g. GPT-4 → GPT-4-mini), the vocabulary has ~100,000 tokens and the teacher's relative rankings of unlikely tokens encode rich semantic relationships — the "dark knowledge." With 27 characters, that information channel is too narrow for distillation to help.

This is a known limitation: distillation benefits scale with the complexity of the output space.

### Training data

The model trains on ~32,000 common names from [Andrej Karpathy's makemore dataset](https://github.com/karpathy/makemore). The dataset is included in the repo for reproducibility.

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

- `config.py` — hyperparameters (architecture, learning rate, distillation temperature, etc.)
- `data.py` — character encoding, context window extraction from names
- `model.py` — flexible MLP with learned embeddings, supporting arbitrary hidden layer configurations via `nn.ModuleList`
- `train.py` — training loop, distillation loss (cross-entropy + KL divergence), and distillation training loop
- `train_teacher.py` — trains the teacher model, saves weights to `teacher.pt`
- `train_student.py` — trains the student on hard labels, saves to `student.pt`
- `distill.py` — trains the student via distillation, saves to `student_distill.pt`
- `evaluate.py` — compares all three models with the same cross-entropy metric
- `main.py` — generates new names from a trained model

### Dependencies

- PyTorch
