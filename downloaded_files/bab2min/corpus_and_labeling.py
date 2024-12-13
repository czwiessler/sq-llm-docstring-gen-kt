import sys
import tomotopy as tp

def corpus_and_labeling_example(input_file):
    corpus = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer(), stopwords=['.'])
    # data_feeder yields a tuple of (raw string, user data) or a str (raw string)
    corpus.process(open(input_file, encoding='utf-8'))

    # make LDA model and train
    mdl = tp.LDAModel(k=20, min_cf=10, min_df=5, corpus=corpus)
    mdl.train(0)
    print('Num docs:', len(mdl.docs), ', Vocab size:', len(mdl.used_vocabs), ', Num words:', mdl.num_words)
    print('Removed top words:', mdl.removed_top_words)
    
    mdl.train(1000, show_progress=True)
    mdl.summary()

    # extract candidates for auto topic labeling
    extractor = tp.label.PMIExtractor(min_cf=10, min_df=5, max_len=5, max_cand=10000)
    cands = extractor.extract(mdl)

    labeler = tp.label.FoRelevance(mdl, cands, min_df=5, smoothing=1e-2, mu=0.25)
    for k in range(mdl.k):
        print("== Topic #{} ==".format(k))
        print("Labels:", ', '.join(label for label, score in labeler.get_topic_labels(k, top_n=5)))
        for word, prob in mdl.get_topic_words(k, top_n=10):
            print(word, prob, sep='\t')
        print()

# You can get the sample data file 'enwiki-stemmed-1000.txt'
# at https://drive.google.com/file/d/18OpNijd4iwPyYZ2O7pQoPyeTAKEXa71J/view?usp=sharing

print('Running LDA and Labeling')
corpus_and_labeling_example('enwiki-stemmed-1000.txt')
