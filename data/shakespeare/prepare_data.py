import numpy as np
text = open('tiny_shakespeare.txt').read()
chars = sorted(set(text))
vocab_size = len(chars)




# 65 pour TinyShakespeare
print(f"vocab size: {vocab_size}")
stoi = {c: i for i, c in enumerate(chars)}

data = np.array([stoi[c] for c in text], dtype=np.int32)
print(data.shape)
# split 90/10
n = int(0.9 * len(data))
train = data[:n]
val = data[n:]

# pour l'entraînement : séquences de longueur (seq_len+1)

# input = seq[0:seq_len]
# target = seq[1:seq_len+1] ← next token prediction
SEQ_LEN = 64
# créer les batchs : à toi de coder le découpage et l'écriture en .bin
# hint : np.lib.stride_tricks.as_strided ou juste une boucle
# format final : X.bin shape (N_samples, SEQ_LEN) int32
#
#y.bin shape (N_samples, SEQ_LEN) int32
