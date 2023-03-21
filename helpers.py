def load_conll(path):
    """Load a conll file. returns a list of sentences with corresponding tags.

    Args:
        path (string): path to conll file

    Returns:
        list[([],[])]: list of touples with token list and tag list
    """
    ents = []
    curSentence = ([],[])
    for line in open(path):
        line = line.strip()
        if line == '':
            ents.append(curSentence)
            curSentence = ([],[])
        else:
            curSentence[0].append(line.split('\t')[0])
            curSentence[1].append(line.split('\t')[1])

    return ents
