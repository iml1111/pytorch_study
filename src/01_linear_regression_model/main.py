import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

# 데이터 준비
data = pd.read_csv('01_data.csv')
# Avoid copy data, just refer
x = torch.from_numpy(data['x'].values).unsqueeze(dim=1).float()
y = torch.from_numpy(data['y'].values).unsqueeze(dim=1).float()

# plt.xlim(0, 11);    plt.ylim(0, 8)
# plt.title('02_Linear_Regression_Model_Data')
# plt.scatter(x, y)
# plt.show()
#----------------------------------------------#
# 모델 선언 및 불러오기
model = nn.Linear(in_features=1, out_features=1, bias=True)
# print(model)
# print(model.weight)
# print(model.bias)

"""
매번 랜덤으로 결정되서 시작
Linear(in_features=1, out_features=1, bias=True)
Parameter containing:
tensor([[-0.9360]], requires_grad=True)
Parameter containing:
tensor([0.7960], requires_grad=True)
"""
#----------------------------------------------#
# 로스 펑션 및 옵티마이저 결정
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

print(model(x))

"""
초기 예측 값 - 당연히 굉장히 잘못된 값이 나옴
tensor([[-0.1399],
        [-1.0759],
        [-2.0119],
        [-2.9478],
        [-3.8838],
        [-4.8197],
        [-5.7557],
        [-6.6917],
        [-7.6276],
        [-8.5636]], grad_fn=<ThAddmmBackward>)
"""
#----------------------------------------------#
# 모델 학습시키기
for step in range(500):
    prediction = model(x) # 모델 예측하기
    # 예측값과 실제값의 에러 값 계산
    loss = criterion(input=prediction, target=y)

    # 옵티마이저 grad 초기화 / 매번 루프마다 해주어야 함
    optimizer.zero_grad()
    # 에러 값을 통한 역전파 개시
    loss.backward()
    # 계산된 grad 값을 토대로 파라미터 갱신
    optimizer.step()

    if step % 20 == 0:
        """
        Show your intermediate results
        중간 과정을 출력할때 사용
        """
        pass
#----------------------------------------------#
# 모델 결과 시각화 및 저장
def display_results(model, x, y):
    prediction = model(x)
    loss = criterion(input=prediction, target=y)
    
    plt.clf()
    plt.xlim(0, 11);    plt.ylim(0, 8)
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'b--')
    plt.title('loss={:.4}, w={:.4}, b={:.4}'.format(loss.data.item(), model.weight.data.item(), model.bias.data.item()))
    plt.show()
    # plt.savefig('results/02_Linear_Regression_Model_trained.png')

display_results(model, x, y)

#-------------------------------------------------#
# 모델 저장하기
# torch.save(obj=model, f='02_Linear_Regression_Model.pt')
#-------------------------------------------------#
# 모델 불러오기
# loaded_model = torch.load(f='02_Linear_Regression_Model.pt')
# display_results(loaded_model, x, y)

#-------------------------------------------------#
#-------------------------------------------------#