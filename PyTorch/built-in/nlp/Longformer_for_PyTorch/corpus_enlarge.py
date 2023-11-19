wcnt = 0
max_wlen = 0
with open('corpus/train_corpus_large.txt', 'w') as wf:
    while wcnt < 1000000:
        for corpus_file in ['corpus/train_corpus.txt', 'corpus/test_corpus.txt']:
            with open(corpus_file) as f:
                for line in f.readlines():
                    llen = line.count(' ')+1
                    wlen = 0
                    for i in range(8):
                        wf.write(line.strip() + ' ')
                        wlen += llen
                    wf.write('\n')
                    max_wlen = max(max_wlen, wlen)
                    wcnt += 1
print(wcnt, max_wlen)

# check data 
rcnt = 0
max_rlen = 0
with open('corpus/train_corpus_large.txt', 'r') as rf:
    for line in rf.readlines():
        rlen = line.count(' ')
        max_rlen = max(max_rlen, rlen)
        rcnt += 1
print(rcnt, max_rlen)