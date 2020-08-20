import requests 
from os import listdir, mkdir, rename
from os.path import isfile, isdir, join
from zipfile import ZipFile


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
    print('Downloading zip file completed.')


def unzip(zip_path, dataset_path):
    zf = ZipFile(zip_path)
    zf.extractall(path=dataset_path)
    zf.close()
    print('Unzipping completed.')


def restructure_dir(data_path, is_train=True):
    files = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    if is_train:
        for file in files:
            if not isdir(join(data_path, file.split('.')[0])):
                mkdir(join(data_path, file.split('.')[0]))
            rename(
                join(data_path, file), join(data_path, file.split('.')[0], file)
            )
    else:
        for file in files:
            if not isdir(join(data_path, 'dummy')):
                mkdir(join(data_path, 'dummy'))
            rename(
                join(data_path, file), join(data_path, 'dummy', file)
            )
    print('Resturcturing completed.')


if __name__ == '__main__':

    # make dataset directory
    dataset_path = './dataset'
    if not isdir(dataset_path):
        print('Making dataset directory on {}'.format(dataset_path))
        mkdir(dataset_path)

    # set hymenoptera dataset
    '''
    개미 / 벌 분류기 데이터셋
    그냥 해당 코드 실행하면 자동으로 다 분류해줌
    '''
    hymenoptera_url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
    hymenoptera_path = './hymenoptera.zip'

    download_url(hymenoptera_url, hymenoptera_path)
    unzip(hymenoptera_path, dataset_path)
    rename(join(dataset_path, 'hymenoptera_data'), join(dataset_path, 'hymenoptera'))
    rename(join(dataset_path, 'hymenoptera', 'val'), join(dataset_path, 'hymenoptera', 'test'))

    # set catdog train dataset
    '''
    고양이 / 개 분류기 데이터셋
    압축파일을 받을 수 있는 URL이 소실되서 직접 다운받아야함.
    https://www.kaggle.com/c/dogs-vs-cats/data?select=train.zip
    여기서 zipfile 직접 다운받고 zip 파일 이름 바꿔서 같은 경로에 놓으면 됨
    
    테스트의 경우, 분류가 되어있지 않으므로 그냥 위에서 다운받고 /catdog/test/dummy 안에 다 구겨 넣으면 됨
    '''
    catdog_path = join(dataset_path, 'catdog')
    catdog_train_path = join(catdog_path, 'train')
    catdog_train_zip = './catdog.zip'

    unzip(catdog_train_zip, catdog_path)
    restructure_dir(catdog_train_path, is_train=True)

