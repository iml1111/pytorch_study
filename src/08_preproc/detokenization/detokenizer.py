'''
디토크나이제이션 순서

1. 공백 제거 (서브워드 분절 정보 제거)
2. 언더바 2개를 공백으로 치환(기존문장의 띄어쓰기 + 미캡의 띄어쓰기 -> 공백 복원)
3. 언더바 1개를 제거 (단독으로 미캡의 띄어쓰기된 정보 제거)
'''
import sys

STR = '▁'
TWO_STR = '▁▁'


def detokenization(line):
    if TWO_STR in line:
        line = line.strip().replace(' ', '').replace(TWO_STR, ' ').replace(STR, '').strip()
    else:
        line = line.strip().replace(' ', '').replace(STR, ' ').strip()

    return line


if __name__ == "__main__":
    for line in sys.stdin:
        if line.strip() != "":
            buf = []
            for token in line.strip().split('\t'):
                buf += [detokenization(token)]

            sys.stdout.write('\t'.join(buf) + '\n')
        else:
            sys.stdout.write('\n')
