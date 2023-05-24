import sys
import numpy as np

def readBIO(path):
    #read in BIO format
    #return list of lists of tags
    ents = []
    for line in open(path):
        line = line.strip()
        ents.append(line) #I don't have words here, so 0
    return ents

def toSpans(tags):
    spans = set()
    spans_list = list()
    for beg in range(len(tags)):
        if tags[beg][0] == 'B':
            for end in range(beg+1, len(tags)):
                if tags[end][0] != 'I':
                    break
            spans.add(str(beg) + '-' + str(end) + ':' + tags[beg][2:])
            spans_list.append(str(beg) + '-' + str(end) + ':' + tags[beg][2:])

    return spans, spans_list

def getInstanceScores(predPath, goldPath):
    goldEnts = readBIO(goldPath)
    predEnts = readBIO(predPath)
    tp = 0
    fp = 0
    fn = 0

    goldSpans, goldSpans_list = toSpans(goldEnts)
    predSpans, predSpans_list = toSpans(predEnts)

    np.savetxt('goldSpans.txt', goldSpans_list, fmt='%s')
    np.savetxt('predSpans.txt', predSpans_list, fmt='%s')

    overlap = len(goldSpans.intersection(predSpans))

    tp += overlap
    fp += len(predSpans) - overlap
    fn += len(goldSpans) - overlap
    #calculate accuracy

    prec = 0.0 if tp+fp == 0 else tp/(tp+fp)
    rec = 0.0 if tp+fn == 0 else tp/(tp+fn)
    f1 = 0.0 if prec+rec == 0.0 else 2 * (prec * rec) / (prec + rec)

    return {'model': predPath.split('/')[-1], 'prec': prec, 'rec': rec, 'f1': f1}
    
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('please provide path to gold file and output of your system (in same format)')
        print('for example: \npython3 eval.py opener_en-dev.conll bert_out-dev.conll')
    else:
        score = getInstanceScores(sys.argv[1], sys.argv[2])
        print(score)
