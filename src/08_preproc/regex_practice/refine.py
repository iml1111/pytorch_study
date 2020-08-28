import sys
import re


def read_regex(fn):
    regexs = []
    f = open(fn, 'r', encoding='UTF8')

    for line in f:
        # 주석 탐지를 방지하기 위함
        if not line.startswith('#'):
            tokens = line.split('\t')

            if len(tokens) == 1:
                tokens += [' ']

            # 마지막 공백 문자 제거
            if tokens[1].endswith("\n"):
                tokens[0] = tokens[0][:-1]
            if tokens[1].endswith("\n"):
                tokens[1] = tokens[1][:-1]

            regexs += [(tokens[0], tokens[1])]
    
    f.close()
    return regexs


if __name__ == '__main__':
    fn = sys.argv[1]
    target_index = int(sys.argv[2])

    regexs = read_regex(fn)

    for line in sys.stdin:
        if line.strip() != "":
            columns = line.strip().split("\t")

            for r in regexs:
                columns[target_index] = re.sub(r'%s' % r[0],
                                               r[1],
                                               columns[target_index].strip())

            sys.stdout.write('\t'.join(columns) + "\n")
        else:
            sys.stdout.write("\n")