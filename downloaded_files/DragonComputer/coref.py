#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
.. module:: coref
    :platform: Unix
    :synopsis: the top-level submodule of Dragonfire that aims to create corefference based dialogs.

.. moduleauthor:: Mehmet Mert Yıldıran <mert.yildiran@bil.omu.edu.tr>
"""

import itertools  # Functions creating iterators for efficient looping

import neuralcoref  # Fast Coreference Resolution in spaCy with Neural Networks


class NeuralCoref():
    """Class to provide corefference based dialogs.
    """

    def __init__(self, nlp):
        """Initialization method of :class:`dragonfire.coref.NeuralCoref` class.
        """

        self.nlp = nlp
        neuralcoref.add_to_pipe(self.nlp)
        self.coms = []

    def core(self, doc, n_sents):
        """Core resolution
        """

        resolution = doc._.coref_resolved
        chained = self.nlp(resolution)
        total_sents = sum(1 for sent in chained.sents)
        sents = itertools.islice(chained.sents, total_sents - n_sents, None)
        sents_arr = []
        for sent in sents:
            sents_arr.append(sent.text)
        return " ".join(sents_arr)

    def resolve(self, com):
        """Method to return the version of command where each corefering mention is replaced by the main mention in the associated cluster (compared to previous commands).

        Args:
            com (str):  User's command.

        Returns:
            str:  Resolution.
        """

        com_doc = self.nlp(com)
        n_sents = sum(1 for sent in com_doc.sents)

        token = None
        for token in com_doc:
            pass
        if token.tag_ not in [',', ':', "."]:
            com += '.'
        self.coms.append(com)

        if len(self.coms) > 1:
            chain = " ".join(self.coms[-2:])

            doc = self.nlp(chain)
            if doc._.has_coref:
                return self.core(doc, n_sents)

            return com

        return com

    def resolve_api(self, com, previous=None):
        if not previous:
            return com
        com_doc = self.nlp(com)
        n_sents = sum(1 for sent in com_doc.sents)

        token = None
        for token in com_doc:
            pass
        if token.tag_ not in [',', ':', "."]:
            com += '.'

        previous_doc = self.nlp(previous)
        token = None
        for token in previous_doc:
            pass
        if token.tag_ not in [',', ':', "."]:
            previous += '.'

        chain = previous + " " + com
        doc = self.nlp(chain)
        if doc._.has_coref:
            return self.core(doc, n_sents)

        return com
