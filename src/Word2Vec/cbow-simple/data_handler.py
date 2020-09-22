
CONTEXT_SIZE = 2

text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

split_ind = (int)(len(text) * 0.8)

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(text)
vocab_size = len(vocab)
print('vocab_size:', vocab_size)

w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for i, w in enumerate(vocab)}


# context window size is two
def create_cbow_dataset(text):
    data = []
    for i in range(CONTEXT_SIZE, len(text) - CONTEXT_SIZE):
        context = [text[i - CONTEXT_SIZE], 
                   text[i - (CONTEXT_SIZE - 1)],
                   text[i + (CONTEXT_SIZE - 1)],
                   text[i + CONTEXT_SIZE]]
        target = text[i]
        data.append((context, target))
    return data