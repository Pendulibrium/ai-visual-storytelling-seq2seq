from nltk.translate.bleu_score import sentence_bleu
from enum import Enum
import h5py
from meteor import MeteorScorer, MeteorReference


class Score_Method(Enum):
    BLEU = "BLEU",
    METEOR = "METEOR"


class Scores:
    def __init__(self):
        self.score_method = None

    def bleu_score(self, references, hypotheses):
        scores = []
        for i in range(len(references)):
            scores.append(sentence_bleu([str(references[i]).split()], str(hypotheses[i]).split()))

        return scores, sum(scores)

    def meteor_score(self, references, hypotheses):

        scores = []
        meteor = MeteorScorer("meteor_language=en,meteor_path=nlp")
        for i in range(len(references)):
            meteor_ref = MeteorReference(str(references[i]).split(), meteor)
            scores.append(meteor_ref.score(str(hypotheses[i]).split()))

        return scores, sum(scores)

    def calculate_scores(self, score_method_name, references, hypotheses):

        self.score_method = score_method_name
        scores = []
        total_score = 0
        if score_method_name == Score_Method.BLEU:
            scores, total_score = self.bleu_score(references, hypotheses)
        elif score_method_name == Score_Method.METEOR:
            scores, total_score = self.meteor_score(references, hypotheses)

        return scores, total_score


