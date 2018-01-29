    # coding: utf-8
import copy
import six
import sys
from chainer import configuration
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import function
from chainer import functions as F
from chainer import link
from chainer import reporter as reporter_module
from chainer.training import extensions

class RankingEvaluator(extensions.Evaluator):

    def __init__(self, iterator, target, args, score_abs_dest, n_qd_pairs, converter, device):
        super(RankingEvaluator, self).__init__(
            iterator=iterator, target=target, device=device)
        self.score_abs_dest = score_abs_dest
        self.converter = converter
        self.n_qd_pairs = n_qd_pairs
        self.device = device
        
        self.epoch_num = 1

    def evaluate(self):
        """Evaluates the model and returns a result dictionary.
        This method runs the evaluation loop over the validation dataset. It
        accumulates the reported values to :class:`~chainer.DictSummary` and
        returns a dictionary whose values are means computed by the summary.
        Users can override this method to customize the evaluation routine.
        Returns:
            dict: Result dictionary. This dictionary is further reported via
                :func:`~chainer.report` without specifying any observer.
        """
        # iterator = self._iterators['main']
        target = self._targets['main']
        eval_func = self.eval_func or target


        if self.eval_hook:
            self.eval_hook(self)

        summary = reporter_module.DictSummary()
        # return summary.compute_mean()

        # test data evaluation
        all_batch_count = sum([i for i in self.n_qd_pairs])
        batch_count = 0.0
        print >> sys.stderr, "\ntest set evaluation"
        iterator = self._iterators['test']
        it = copy.copy(iterator)
        score_first = []
        score_second = []
        pre_rate = 0
        for batch in it:
            observation = {}
            # trainでない時には、別処理をする分岐がNetwork.py内に必要
            with reporter_module.report_scope(observation):
                # 1インスタンスごとのスコアをリストに格納
                
                xs1, xs2, xs3, y = self.converter(batch, device=self.device)
                y_score, first_rel_score, second_rel_score = target.predictor(xs1, xs2, xs3)

                loss = F.hinge(x=y_score, t=y).data
                reporter_module.report({'loss_test': loss}, target)

                # P@Nなどのためにscoreをリストに格納
                score_first += first_rel_score.data.flatten().tolist()
                score_second += second_rel_score.data.flatten().tolist()

                batch_count += len(batch)
                rate = int(batch_count / all_batch_count * 10)
                if rate != pre_rate:
                    print >> sys.stderr, "{}%".format(int(batch_count / all_batch_count * 10) * 10),
                pre_rate = rate

            summary.add(observation)

        # development data evaluation
        print >> sys.stderr, "\ndev set evaluation"
        iterator = self._iterators['dev']
        it = copy.copy(iterator)
        batch_count = 0.0
        pre_rate = 0
        for batch in it:
            observation = {}
            # trainでない時には、別処理をする分岐がNetwork.py内に必要
            with reporter_module.report_scope(observation):
                # 1インスタンスごとのスコアをリストに格納
                xs1, xs2, xs3, y = self.converter(batch, device=self.device)
                y_score, first_rel_score, second_rel_score = target.predictor(xs1, xs2, xs3)

                loss = F.hinge(x=y_score, t=y).data
                reporter_module.report({'loss_dev': loss}, target)

                batch_count += len(batch)
                rate = int(batch_count / all_batch_count * 10)
                if rate != pre_rate:
                    print >> sys.stderr, "{}%".format(int(batch_count / all_batch_count * 10) * 10),
                pre_rate = rate

            summary.add(observation)

        current_position = 0
        score = []
        for t in self.n_qd_pairs:
            score.append(score_first[current_position])
            for s in score_second[current_position:current_position+t]:
                score.append(s)
            current_position += t

        # p@N, MAP, nDCGのためのスコア書き込み
        with open(self.score_abs_dest+"/score_epoch{}.txt".format(self.epoch_num),"w") as fo:
            for s in score:
                fo.write("{}\n".format(s))
        self.epoch_num += 1

        return summary.compute_mean()


