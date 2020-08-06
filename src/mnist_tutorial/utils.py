def load_mnist(is_train=True, flatten=True):
    '''
    Mnist 예제 학습용 데이터 호출
    '''
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        '../data', train=is_train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    x = dataset.data.float() / 255.
    y = dataset.targets

    # flatten 차원을 하나 낮추고, 평평하게 나열함
    if flatten:
        x = x.view(x.size(0), -1)

    return x, y
