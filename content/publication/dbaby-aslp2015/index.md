+++
title = "Coupled Dictionaries for Exemplar-Based Speech Enhancement and Automatic Speech Recognition"
date = 2015-11-15
draft = false

authors = ["admin", "tuomas-virtanen", "jort-gemmeke", "hugo-vanhamme"]
publication_types = ["2"]

publication = "IEEE/ACM Transactions on Audio, Speech and Language Processing"
publication_short = "IEEE/ACM-TASLP"

abstract = "Exemplar-based speech enhancement systems work by decomposing the noisy speech as a weighted sum of speech and noise exemplars stored in a dictionary and use the resulting speech and noise estimates to obtain a time-varying filter in the full-resolution frequency domain to enhance the noisy speech. To obtain the decomposition, exemplars sampled in lower dimensional spaces are preferred over the full-resolution frequency domain for their reduced computational complexity and the ability to better generalize to unseen cases. But the resulting filter may be sub-optimal as the mapping of the obtained speech and noise estimates to the full-resolution frequency domain yields a low-rank approximation. This paper proposes an efficient way to directly compute the full-resolution frequency estimates of speech and noise using coupled dictionaries: an input dictionary containing atoms from the desired exemplar space to obtain the decomposition and a coupled output dictionary containing exemplars from the full-resolution frequency domain. We also introduce modulation spectrogram features for the exemplar-based tasks using this approach. The proposed system was evaluated for various choices of input exemplars and yielded improved speech enhancement performances on the AURORA-2 and AURORA-4 databases. We further show that the proposed approach also results in improved word error rates (WERs) for the speech recognition tasks using HMM-GMM and deep-neural network (DNN) based systems."

abstract_short = ""

selected = true

projects = ["INSPIRE-ITN"]
tags = ["speech enhancement", "nmf"]

url_pdf = "https://ieeexplore.ieee.org/document/7138598"
url_preprint = ""
url_code = ""
url_dataset = ""
url_project = ""
url_slides = ""
url_video = ""
url_poster = ""
url_source = ""

doi = "10.1109/TASLP.2015.2450491"

comments = false
profile = true
+++
