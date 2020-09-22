import random


CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right

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


w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for i, w in enumerate(vocab)}


def create_skipgram_dataset(text):
    data = []
    # CONTEXT_SIZE에 맞게 학습 데이터 생성
    for i in range(CONTEXT_SIZE, len(text) -CONTEXT_SIZE):
        
        # sampling
        data.append((text[i], text[i - CONTEXT_SIZE], 1))
        data.append((text[i], text[i - (CONTEXT_SIZE - 1)], 1))
        data.append((text[i], text[i + (CONTEXT_SIZE - 1)], 1))
        data.append((text[i], text[i + CONTEXT_SIZE], 1))

        # negative sampling: 정답이랑 똑같은 만큼의 
        for _ in range(CONTEXT_SIZE * 2):
            '''
            실제 정답을 제외한 오답을 학습 데이터로 생성
            0 ~ i - 3 < i - 2 ~ i + 2 < i + 3 ~ total_len
            '''

            # 50% 확률로 이전껄로 샘플링 or 
            # 마지막 중심단어일 경우 무조건 이전껄로
            if (random.random() < 0.5 or i >= len(text) - (CONTEXT_SIZE + 1)) \
                and i >= (CONTEXT_SIZE + 1):
                rand_id = random.randint(0, i - (CONTEXT_SIZE + 1))
            else:
                rand_id = random.randint(i + (CONTEXT_SIZE + 1), len(text) - 1)
            data.append((text[i], text[rand_id], 0))
    return data


if __name__ == '__main__':
    skipgram_train = create_skipgram_dataset(text)
    print('skipgram sample', skipgram_train)