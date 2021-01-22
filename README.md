# pytorch_study
제가 Pytorch(주로 자연어처리)를 공부하면서 다룬 실습 코드들을 모아둔 repo입니다.

해당 실습 코드를 구현하기 위해 참고한 References는 맨 아래의 링크들을 참고해주세요.



# Indexes

1. linear_regression_model
2. mnist_tutorial
3. mnist_autoencoder
4. advanced_pytorch
5. cnn
6. transfer_learning
7. rnn
8. preprocessing
9. word_embedding
10. text_classification
12. Seq2Seq
13. Transformer

Appendix-1. LDA

Appendix-2. Word2Vec



## Linear Regression

파이토치의 가장 기본적인 사용 방법에 대한 이해를 위해, 간단한 선형 회귀 예측을 구현한 코드입니다.

data set 또한 굉장히 작기 때문에, 01_data.csv로 디렉터리에 함께 동봉되어 있습니다.

실행 코드 또한 main.py 코드 한개입니다.



## Mnist Tutorial

마찬가지로 파이토치의 기본적인 사용 방법 이해를 위한 Mnist Tutorial 코드입니다.

utils.py의 load_mnist Function을 통해 **Mnist Dataset를 자동으로 다운받아 학습을 시작**합니다.

또한, 코드 구조화를 위해 다음과 같이 코드 파일을 구분하였습니다.

- **train.py (메인 학습 실행 코드)**

- model.py (신경망 구조 클래스)
- trainer.py (학습기 클래스)
- utils.py (유틸 함수 모음)
- **predict.py (모델 예측 실행 코드)**



## Mnist Autoencoder

간단한 오토 인코더를 구현해보는 코드로, Mnist로 실습을 수행하였습니다. 기본적으로 위의 Mnist Tutorial 코드와 같은 구조를 가집니다.



## Advanced Pytorch

### 1. Custom Dataset

Pytorch에 지원하는 커스텀 데이터셋 클래스를 사용해보는 실습입니다. Dataset으로는 Sklearn의 유방암 데이터셋을 사용하였습니다.

### 2. Pytorch ignite

위에서 사용한 커스텀 데이터셋과 더불어, Pytorch 확장 프레임워크인 Ignite를 사용하여 코드 구조화에 사용하였습니다. Dataset은 Mnist이며, 이후에 모든 코드는 Ignite를 통해 구현되었습니다.



## CNN

Mnist를 CNN을 사용하여 구현한 코드입니다.



## Transfer Learning

torchvision에서 제공하는 model을 기반으로 전이 학습을 실시하는 코드입니다. Dataset은 Mnist이며, **"resnet", "alexnet", "vgg", "squeezenet", "densenet"**을 이용해 전이학습을 수행합니다.

전이 학습시, 사전에 기록된 기존 모델의 parameter을 동결시킬 것인지 말지를 결정할 수 있으며, 마지막 Output 단의 Classification의 Class 수를 재정의하여 전이학습을 수행합니다.



## RNN

Mnist를 RNN을 사용하여 구현한 코드입니다.



## Preprocessing

해당 챕터는 모델 학습 관련 코드가 아닙니다. 자연어처리에서 데이터 전처리시에 사용되는**, Tokenize, Regular Expression, Subword Segmentation** 등에 대하여 다룹니다.

또한, **torchtext**에 제공하는 Dataset 기능을 사용해보았습니다.



## Word Embedding

Word Embedding 실습을 위해, Facebook의 FastText를 통해 임베딩 벡터를 구하는 실습입니다.

(FastText의 경우, Gensim 라이브러리를 통해서 Python 단에서 학습시킬 수 있습니다) 



## Text Classification

**영화 리뷰 데이터를 기반으로 감정 분석 (긍정/부정)을 수행하는 Text Classification 학습 코드**입니다.

Dataset인 영화 리뷰 데이터의 경우, 저작권 문제로 올리지 않았습니다.

(Dataset 규격은 TSV로 각 라인당, 아래와 같이 구성되어 있습니다.)

```sh
# label: 1 or 0
# text: 영화 리뷰 텍스트
<label>\t<text>
<label>\t<text>
<label>\t<text>
...
```

전이 학습을 사용하지 않을 경우, 사용 가능한 모델은 위에서 실습했던 **CNN, RNN**입니다.

전이 학습을 사용할 경우, BERT를 이용한 finetuning이 가능합니다.

default로 사용된 BERT 모델은 **"beomi/kcbert-base"** 입니다.



## Seq2Seq

시퀀스 투 시퀀스 모델을 이용하여 기계 번역기를 구현한 코드입니다.

학습에 사용된 Dataset은 AI-HUB에서 제공된 Open Data를 사용하였습니다.

https://www.aihub.or.kr/aidata/87

Model에는 **Tearch Forcing으로 인하여 학습(foward)과 추론(search) 함수가 존재**하며, 추론시 성능 향상을 도울 수 있는 **Beam Search**까지 구현되어 있습니다.

그 밖에 Trainer에는 **Gradient Accumulation** 및 **Gradient Clipping**이 구현이 되어 있습니다.



## Transformer

트랜스포머 모델을 이용하여 기계 번역기를 구현한 코드입니다.

학습된 데이터를 포함하여, 모델의 Input Output까지 위의 Seq2Seq와 구조가 일치하기 때문에 신경망을 제외한 대부분의 코드가 일치합니다.

단, 기존 Paper 방식인 Post-LN 방식의 경우, 하이퍼 파라미터 튜닝에 까다로운 점이 있기 때문에 **Pre-LN을 기반하여 구현**하였습니다.




# References

https://github.com/kh-kim/nlp_with_pytorch_examples

https://github.com/kh-kim/simple-nmt

[PyTorch - Word Embedding](https://www.tutorialspoint.com/pytorch/pytorch_word_embedding.htm)

[jojonki/word2vec-pytorch](https://github.com/jojonki/word2vec-pytorch)

[PengFoo/word2vec-pytorch](https://github.com/PengFoo/word2vec-pytorch)

[ray1007/pytorch-word2vec](https://github.com/ray1007/pytorch-word2vec)

https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-00-references/

https://wikidocs.net/52460
