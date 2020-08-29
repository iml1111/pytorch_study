from data_loader import DataLoader

loaders = DataLoader(
    train_fn='./review.sorted.uniq.refined.tok.shuf.train.tsv',
    batch_size=256,
    valid_ratio=.2,
    device=-1,
    max_vocab=999999,
    # 최소 5번 이상 등장하지 않으면 단어 사전에 등록 안됨
    min_freq=5)

print("# loader 체크")
print("|train|=%d" % len(loaders.train_loader.dataset))
print("|valid|=%d" % len(loaders.valid_loader.dataset))
print("|vocab|=%d" % len(loaders.text.vocab))
print("|label|=%d" % len(loaders.label.vocab), "\n")

print("# 미니 배치 하나 꺼내기")
data = next(iter(loaders.train_loader))
print(data.text.shape)
print(data.label.shape, "\n")

print("# 단어 사전 (vocabulary)")
# 빈도수에 기반하여 사전이 만들어지므로
# 같은 데이터셋이라면 만들어지는 단어 사전은 같음
print(loaders.text.vocab.stoi['배송'])
print(loaders.text.vocab.itos[18], "\n")

print("# 단어 빈도수 출력")
# 각종 특수 토큰은 무조건 상위에 차지하도록 함
for i in range(10):
    word = loaders.text.vocab.itos[i]
    print('%5d: %s\t%d' % (i, word, loaders.text.vocab.freqs[word]))
print()


print("# 텍스트 복원해보기")

print("## 해당 미니배치의 마지막 텐서")
x = data.text[-1]
print(x)

print("## 해당 텐서를 이용해서 원문 복원")
line = []
for x_i in x:
    line += [loaders.text.vocab.itos[x_i]]
print(" ".join(line))
