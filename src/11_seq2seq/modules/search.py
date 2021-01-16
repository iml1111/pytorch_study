from operator import itemgetter

import torch
import torch.nn as nn

import modules.data_loader as data_loader

LENGTH_PENALTY = .2
MIN_LENGTH = 5


class SingleBeamSearchBoard():

    def __init__(
        self,
        device,
        prev_status_config,
        beam_size=5,
        max_length=255,
    ):
        self.beam_size = beam_size
        self.max_length = max_length

        self.device = device
        # 각 타임 스텝의 Word Index(즉, 최종 예측 단어들) * beam_size
        # 처음에는 모두 <BOS>므로 초기화
        self.word_indice = [torch.LongTensor(beam_size).zero_().to(self.device) + data_loader.BOS]
        # 각 타임 스텝의 Word들이 선정된 Beam Index
        # 처음에는 아무것도 선정되지 않았기에 -1로 초기화
        self.beam_indice = [torch.LongTensor(beam_size).zero_().to(self.device) - 1]
        # 각 Beam들의 누적 확률 값
        # 처음에는 [0, -inf, -inf, ...]로 초기화
        self.cumulative_probs = [torch.FloatTensor([.0] + [-float('inf')] * (beam_size - 1)).to(self.device)]
        # 각 빔이 현재 EOS에 도달했는지 여부
        # 1 if it is done else 0
        self.masks = [torch.BoolTensor(beam_size).zero_().to(self.device)]
        # We don't need to remember every time-step of hidden states:
        #       prev_hidden, prev_cell, prev_h_t_tilde
        # What we need is remember just last one.

        '''
        각 빔의 이전 hidden, cell, h_tilde를 저장해두는 공간
        항상 마지막 타임스텝만 보관하면 됨

        단 처음에는 그냥 넘겨받은 hidden, cell, h_tilde를 beam_size만큼 늘려줌
        h_tilde의 경우, 처음에 None이므로 예외처리
        '''
        self.prev_status = {}
        self.batch_dims = {}
        for prev_status_name, each_config in prev_status_config.items():
            init_status = each_config['init_status']
            batch_dim_index = each_config['batch_dim_index']
            if init_status is not None:
                self.prev_status[prev_status_name] = torch.cat([init_status] * beam_size,
                                                               dim=batch_dim_index)
            else:
                self.prev_status[prev_status_name] = None
            self.batch_dims[prev_status_name] = batch_dim_index

        self.current_time_step = 0
        self.done_cnt = 0

    def get_length_penalty(
        self,
        length,
        alpha=LENGTH_PENALTY,
        min_length=MIN_LENGTH,
    ):
        # Calculate length-penalty,
        # because shorter sentence usually have bigger probability.
        # In fact, we represent this as log-probability, which is negative value.
        # Thus, we need to multiply bigger penalty for shorter one.
        p = ((min_length + 1) / (min_length + length))**alpha

        return p

    def is_done(self):
        # Return 1, if we had EOS more than 'beam_size'-times.
        if self.done_cnt >= self.beam_size:
            return 1
        return 0

    def get_batch(self):
        '''
        현재 빔에서 가장 마지막 스텝의 워드 인덱스들을 가져옴
        처음에는, 당연히 모두 BOS 일것임
        그 후로는 이전에 예측했던 TopK의 단어들을 주게 될것임
        '''
        y_hat = self.word_indice[-1].unsqueeze(-1)
        # |y_hat| = (beam_size, 1)
        # if model != transformer:
        #     |hidden| = |cell| = (n_layers, beam_size, hidden_size)
        #     |h_t_tilde| = (beam_size, 1, hidden_size) or None
        # else:
        #     |prev_state_i| = (beam_size, length, hidden_size),
        #     where i is an index of layer.
        return y_hat, self.prev_status

    #@profile
    def collect_result(self, y_hat, prev_status):
        '''
        y_hat: 현재 타입스텝의 각 beam마다 예측한 단어
        pre_status: 현재 타입스텝에서 함께 나왔던 hidden, cell, h_tilde
        넣을때, beam 채로 넣었으므로 그대로 다시 나오게 됨
        '''

        # |y_hat| = (beam_size, 1, output_size)
        # |hidden| = |cell| = (n_layers, beam_size, hidden_size)
        # |h_t_tilde| = (beam_size, 1, hidden_size)
        output_size = y_hat.size(-1)

        self.current_time_step += 1

        # 누적 확률 값을 계산함
        # (beam_size) --> (beam_size, 1, 1) --> (beam_size, 1, output_size)
        # 이미 예측이 끝난 경우, 즉 EOS인 경우, 확률값에 -inf을 덮어씀
        cumulative_prob = self.cumulative_probs[-1].masked_fill_(self.masks[-1], -float('inf'))
        # 각 단어(output_Size)만큼의 누적확률 값을 계산하기 위해
        # (beam_size, 1, output_size)의 크기로 늘려줌
        # 그 후, 입력받은 y_hat과 더해서 최종 누적 확률 값 산출
        # 하지만 이떄, 맨처음 cumulative_prob가 (0, -inf, -inf)이므로
        # 처음에는 첫 번째 빔에서만 모든 결과가 나오게 될 것임
        cumulative_prob = y_hat + cumulative_prob.view(-1, 1, 1).expand(self.beam_size, 1, output_size)
        # |cumulative_prob| = (beam_size, 1, output_size)

        # cumulative_prob를 (beam_size * output_size,)로 
        # flatten 해준후 확률이 높은 순으로 정렬
        # top_indice에는 원래 정렬되기전 index가 유지됨
        top_log_prob, top_indice = cumulative_prob.view(-1).sort(descending=True)
        # 그후, TopK개만큼 잘라냄
        top_log_prob, top_indice = top_log_prob[:self.beam_size], top_indice[:self.beam_size]
        # |top_log_prob| = (beam_size,)
        # |top_indice| = (beam_size,)
        # top_log_prob: 각 단어에 대한 확률 값
        # top_indice: 각 단어들의 index -> 해당 인덱스를 이용해서
        # 어느 빔의 어느 단어가 인지를 추적할 수 있음
       

        # 모든 top_indice를 output_size로 나눈 나머지를 구함으로써
        # 각 top_indice가 원래 가르키던 word_Index가 튀어나오게 됨
        self.word_indice += [top_indice.fmod(output_size)]
        # 모든 top_indice를 output_size로 나눔으로써
        # 각 top_indice가 원래 가르키던 Beam_index가 나오게 됨
        # 이로써, 최종적으로 topK에 (어떤 빔)에서 나와서 (어떤 단어)가 선정되었는지 식별
        self.beam_indice += [top_indice.div(float(output_size)).long()]

        # 이번 스텝에서 구한 누적 확률값을 객체에 갱신
        self.cumulative_probs += [top_log_prob]
        # 이번 결과를 보며, EOS가 나온 곳을 mask 처리
        self.masks += [torch.eq(self.word_indice[-1], data_loader.EOS)] 
        # 마스크 결과를 바탕으로 done_cnt 캐싱
        self.done_cnt += self.masks[-1].float().sum()

        # 현재 타임스텝에서 도출된 각종 hidden, cell, h_tilde 값을
        # 객체에 저장해야 함. -> 이후 get_batch에서 호출될때 사용
        # 단 이때, topK로 선정된 Beam_index의 hidden, cell, h_tilde만 가지고감
        for prev_status_name, prev_status in prev_status.items():
            self.prev_status[prev_status_name] = torch.index_select(
                prev_status,
                dim=self.batch_dims[prev_status_name],
                index=self.beam_indice[-1]
            ).contiguous()

    def get_n_best(self, n=1, length_penalty=.2):
        '''
        이때까지의 Beam Board를 찾아보며,
        가장 확률 값이 높았던 N개의 문장 추출
        '''
        sentences, probs, founds = [], [], []
        
        '''
        mask 여부를 통해, EOS 즉, 온전히 끝난 문장을 탐색
        찾았다면, 해당 문장의 EOS(끝) 인덱스와 마지막으로 나왔던 beam 인덱스,
        그리고 그 당시에 누적 확률값을 저장
        '''
        for t in range(len(self.word_indice)):  # for each time-step,
            for b in range(self.beam_size):  # for each beam,
                if self.masks[t][b] == 1:  # if we had EOS on this time-step and beam,
                    # Take a record of penaltified log-proability.
                    probs += [self.cumulative_probs[t][b] * self.get_length_penalty(t, alpha=length_penalty)]
                    founds += [(t, b)]

        # 만약에, EOS는 아니지만, max_length에 도달해버려 끊겨버린 경우도 수집해옴
        # Also, collect log-probability from last time-step, for the case of EOS is not shown.
        for b in range(self.beam_size):
            if self.cumulative_probs[-1][b] != -float('inf'): # If this beam does not have EOS,
                if not (len(self.cumulative_probs) - 1, b) in founds:
                    probs += [self.cumulative_probs[-1][b] * self.get_length_penalty(len(self.cumulative_probs),
                                                                                     alpha=length_penalty)]
                    founds += [(t, b)]

        # Sort and take n-best.
        # 갖고온 문장의 EOS 인덱스를 확률과 묶어서 내림차순 정렬후, N개를 자름
        sorted_founds_with_probs = sorted(
            zip(founds, probs),
            key=itemgetter(1),
            reverse=True,
        )[:n]
        probs = []

        '''
        정렬된 각 인덱스(EOS)부터 문장을 역으로 내려가며 단어를 수집함
        이때, 단어가 beam을 계속해서 옮겨다니며 선정했을 것이기에
        반대로, 자신이 나왔던 beam의 단어를 하나씩 추적하며 내려가야 함
        '''
        for (end_index, b), prob in sorted_founds_with_probs:
            sentence = []

            # Trace from the end.
            for t in range(end_index, 0, -1):
                sentence = [self.word_indice[t][b]] + sentence
                b = self.beam_indice[t][b]

            sentences += [sentence]
            probs += [prob]

        return sentences, probs
