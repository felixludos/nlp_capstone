"""
This is a skeleton for building your own solver.

You just need to find and fix the two TODOs in this file.
"""
from typing import List

from aristomini.common.solver import SolverBase
from aristomini.common.models import MultipleChoiceQuestion, MultipleChoiceAnswer, ChoiceConfidence

import csv
import util
import nltk
import word_util as wtil
import graph_util as gtil

# TODO: replace with your solver name.
MY_SOLVER_NAME = "simple_graph_solver"

class MySolver(SolverBase):
    def solver_info(self) -> str:
        return MY_SOLVER_NAME

    def answer_question(self, question: MultipleChoiceQuestion) -> MultipleChoiceAnswer:
        # pylint: disable=unused-variable

        stem = question.stem
        choices = question.choices
        
        qbag = wtil.bag_text(stem)
        

        confidences: List[float] = []
        
        picks = {}

        for i, choice in enumerate(question.choices):
            label = choice.label
            text = choice.text
            
            abag = wtil.bag_text(text) + qbag

            picks[i] = wtil.extract_bag(abag)

        f = {}
        for i, (k, t) in enumerate(picks.items()):
            common = set()
            for j, (k2, s) in enumerate(picks.items()):
                if i == j:
                    continue
                common.update(s)
            f[k] = t.difference(common)
            
        total = sum([len(v)+1 for v in f.values()])
        
        for i in range(len(f)):
            confidences.append(f[i]/(total))
            
        print(confidences)

        return MultipleChoiceAnswer(
            [ChoiceConfidence(choice, confidence)
             for choice, confidence in zip(choices, confidences)]
        )

if __name__ == "__main__":
    solver = MySolver()  # pylint: disable=invalid-name
    solver.run()
