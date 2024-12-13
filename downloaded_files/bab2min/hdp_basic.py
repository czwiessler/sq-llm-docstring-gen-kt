import sys
import tomotopy as tp

def hdp_example(input_file, save_path):
    mdl = tp.HDPModel(tw=tp.TermWeight.ONE, min_cf=3, rm_top=5)
    for n, line in enumerate(open(input_file, encoding='utf-8')):
        ch = line.strip().split()
        mdl.add_doc(ch)
    mdl.burn_in = 100
    mdl.train(0)
    print('Num docs:', len(mdl.docs), ', Vocab size:', len(mdl.used_vocabs), ', Num words:', mdl.num_words)
    print('Removed top words:', mdl.removed_top_words)
    print('Training...', file=sys.stderr, flush=True)
    mdl.train(1000, show_progress=True)
    mdl.summary()
    print('Saving...', file=sys.stderr, flush=True)
    mdl.save(save_path, True)

    important_topics = [k for k, v in sorted(enumerate(mdl.get_count_by_topics()), key=lambda x:x[1], reverse=True)]
    for k in important_topics:
        if not mdl.is_live_topic(k): continue
        print('Topic #{}'.format(k))
        for word, prob in mdl.get_topic_words(k):
            print('\t', word, prob, sep='\t')
# You can get the sample data file 'enwiki-stemmed-1000.txt'
# at https://drive.google.com/file/d/18OpNijd4iwPyYZ2O7pQoPyeTAKEXa71J/view?usp=sharing

print('Running HDP')
hdp_example('enwiki-stemmed-1000.txt', 'test.hdp.bin')
