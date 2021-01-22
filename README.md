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

해당 챕터는 모델 학습 관련 코드가 아닙니다. 자연어처리에서 데이터 전처리시에 사용되는, Tokenize, Regular Expression, Subword Segmentation 등에 대하여 다룹니다.

또한, torchtext에 제공하는 Dataset 기능을 사용해보았습니다.








# References

https://github.com/kh-kim/nlp_with_pytorch_examples

https://github.com/kh-kim/simple-nmt

[PyTorch - Word Embedding](https://www.tutorialspoint.com/pytorch/pytorch_word_embedding.htm)

[jojonki/word2vec-pytorch](https://github.com/jojonki/word2vec-pytorch)

[PengFoo/word2vec-pytorch](https://github.com/PengFoo/word2vec-pytorch)

[ray1007/pytorch-word2vec](https://github.com/ray1007/pytorch-word2vec)

https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-00-references/

https://wikidocs.net/52460
