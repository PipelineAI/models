#!/usr/bin/env python3
# ------------- Builtin imports --------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random

# ------------- 3rd-party imports ------------------------------------------------------------------


class Model(object):

    def __init__(self):
        pass

    @staticmethod
    def predict(routes: dict) -> dict:
        """
        TODO: Implement bandit logic to optimize route weights

        :param dict routes:     existing routes by tag and weight

        :return:                dict: bandit optimized routes by tag and weight
        """

        n = len(routes)
        autoroutes = dict()
        cumulative_total = 0

        # TODO: replace deployment target cost simulator with your custom bandit logic
        # deployment target cost simulator
        for (k, v) in routes.items():

            if n == 1:
                i = 100 - cumulative_total
            else:
                i = random.randint(1, 21)*5
                if cumulative_total + i > 100:
                    i = 0

            cumulative_total += i
            autoroutes[k] = i
            n -= 1

        if cumulative_total < 100:
            autoroutes[next(iter(routes.keys()))] = 100 - cumulative_total

        return autoroutes
