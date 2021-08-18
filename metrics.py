import numpy as np
import json
from collections import Counter
np.random.seed(0)

class Metrics(object):

    def __init__(self, score_file_path:str, type_file_path: str):
        super(Metrics, self).__init__()
        self.score_file_path = score_file_path
        self.type_file_path = type_file_path
        self.segment = 10

    def __read_socre_file(self, score_file_path):
        sessions = []
        session_text = []
        one_sess = []
        one_sess_text = []
        candidate_type = json.loads(open(self.type_file_path).read())
        with open(score_file_path, 'r') as infile:
            i = 0
            for line in infile.readlines():
                tokens = line.strip().split('\t')
                one_sess.append((float(tokens[0]), int(tokens[1])))
                one_sess_text.append([candidate_type[0][i], candidate_type[1][i], float(tokens[0])])
                i += 1
                if i % self.segment == 0:
                    one_sess_tmp = np.array(one_sess)
                    if one_sess_tmp[:, 1].sum() > 0:
                        sessions.append(one_sess)
                        session_text.append(one_sess_text)
                    one_sess = []
                    one_sess_text = []
        return sessions, session_text

    def cal_ndcg(self, golden, scores, n = -1):
        def dcg_at_n(rel,n):
            rel = np.asfarray(rel)[:n]
            dcg = sum(np.divide(rel, log2_table[:rel.shape[0]]))
            return dcg
        log2_table = np.log2(np.arange(2, 102))
        scores = np.array(scores)
        sorted_list =  sorted(list(zip(scores, golden)), key=lambda x: x[0], reverse=True)
        rel_score = [m[1] for m in sorted_list]
        #rel_score  = [x if x != 2 else 10 for x in rel_score]
        #golden = [x  if x != 2 else 10 for x in golden ]
        k = len(golden) if n == -1 else n
        idcg = dcg_at_n(sorted(golden, reverse=True), n=k)
        dcg = dcg_at_n(rel_score, n=k)
        return dcg/idcg


    def __mean_average_precision(self, sort_data):
        #to do
        count_1 = 0
        sum_precision = 0
        for index in range(len(sort_data)):
            if sort_data[index][1] == 1:
                count_1 += 1
                sum_precision += 1.0 * count_1 / (index+1)
        return sum_precision / count_1


    def __mean_reciprocal_rank(self, sort_data):
        sort_lable = [s_d[1] for s_d in sort_data]
        assert 1 in sort_lable
        return 1.0 / (1 + sort_lable.index(1))

    def __precision_at_position_1(self, sort_data):
        if sort_data[0][1] == 1:
            return 1
        else:
            return 0

    def __recall_at_position_k_in_10(self, sort_data, k):
        sort_label = [s_d[1] for s_d in sort_data]
        select_label = sort_label[:k]
        return 1.0 * select_label.count(1) / sort_label.count(1)


    def evaluation_one_session(self, data):
        '''
        :param data: one conversion session, which layout is [(score1, label1), (score2, label2), ..., (score10, label10)].
        :return: all kinds of metrics used in paper.
        '''
        np.random.shuffle(data)
        sort_data = sorted(data, key=lambda x: x[0], reverse=True)
        m_a_p = self.__mean_average_precision(sort_data)
        m_r_r = self.__mean_reciprocal_rank(sort_data)
        p_1   = self.__precision_at_position_1(sort_data)
        r_1   = self.__recall_at_position_k_in_10(sort_data, 1)
        r_2   = self.__recall_at_position_k_in_10(sort_data, 2)
        r_5   = self.__recall_at_position_k_in_10(sort_data, 5)
        return m_a_p, m_r_r, p_1, r_1, r_2, r_5

    def evaluate_all_metrics(self):
        sum_m_a_p = 0
        sum_m_r_r = 0
        sum_p_1 = 0
        sum_r_1 = 0
        sum_r_2 = 0
        sum_r_5 = 0
        sum_ndcg = 0

        sessions, session_text = self.__read_socre_file(self.score_file_path)
        total_s = len(sessions)
        predict_answer = []
        for i, session in enumerate(sessions):
            m_a_p, m_r_r, p_1, r_1, r_2, r_5 = self.evaluation_one_session(session)
            true_labels = [s[1] for s in session_text[i]]
            scores = [s[-1] for s in session_text[i]]
            ndcg = self.cal_ndcg(true_labels, scores, n=5)
            best_answer = max(session_text[i], key=lambda x: x[-1])[0]
            predict_answer.append(best_answer)

            sum_m_a_p += m_a_p
            sum_m_r_r += m_r_r
            sum_p_1 += p_1
            sum_r_1 += r_1
            sum_r_2 += r_2
            sum_r_5 += r_5
            sum_ndcg += ndcg
        return (sum_m_a_p/total_s,
                sum_m_r_r/total_s,
                  sum_p_1/total_s,
                  sum_r_1/total_s,
                  sum_r_2/total_s,
                  sum_r_5/total_s,
                  sum_ndcg/total_s)


if __name__ == '__main__':
    metric = Metrics(f'/home/qhj/math/output_weibo/weibo.ioi.40/test.score', '/home/qhj/dw/weibo/test.type', '/home/qhj/dw/weibo/test_weibo.text')
    result = metric.evaluate_all_metrics()

    for r in result:
        print(r)










