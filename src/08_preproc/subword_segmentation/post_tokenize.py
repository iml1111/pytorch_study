'''
# Tokenization
기존 문장의 띄어쓰기를 언더바(_)로 바꾸고,
미캡에 의해 띄워진 걸 그대로 유지함.

# Subword Segmentation
미캡에 의해 띄워진걸 언더바(_)로 바꾸고,
서브워드 분절에 의해 띄워진걸 그대로 유지함
'''
import sys

STR = '▁'

if __name__ == "__main__":
    ref_fn = sys.argv[1]

    f = open(ref_fn, 'r')

    for ref in f:
        # 미캡으로 분절한 텍스트 : stdin > input_line > tokens
        # 원래 텍스트 : ref > ref_tokens
        ref_tokens = ref.strip().split(' ')
        input_line = sys.stdin.readline().strip()
        if input_line != "":
            tokens = input_line.split(' ')

            idx = 0
            buf = []

            # We assume that stdin has more tokens than reference input.
            # tokens: 실제로 output에 작성할 토큰 리스트
            # ref_tokens: 원본 문장의 텍스트
            for ref_token in ref_tokens:
                tmp_buf = []

                while idx < len(tokens):
                    if tokens[idx].strip() == '':
                        idx += 1
                        continue

                    tmp_buf += [tokens[idx]]
                    idx += 1
                    '''
                    당연히 미캡의 토큰이 원본 문장보다 더 쪼개져 있음.
                    그걸 리스트로 합치면서 원본 문장과 같아질때까지 
                    분절된 상태로 수집
                    '''
                    if ''.join(tmp_buf) == ref_token:
                        break

                # tmp 버퍼의 맨처음 토큰은 실제 원본 띄어쓰기일 것이므로
                # 스폐셜 토큰을 앞에다 삽입
                # 그 후, 실제 버퍼에 수집 -> 이게 진짜 토큰임
                if len(tmp_buf) > 0:
                    buf += [STR + tmp_buf[0].strip()] + tmp_buf[1:]

            sys.stdout.write(' '.join(buf) + '\n')
        else:
            sys.stdout.write('\n')

    f.close()
