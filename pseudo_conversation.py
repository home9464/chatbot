"""generate pesudo conversations
"""
import os
import random
import sys
import params
INT_MAX = sys.maxsize
INT_MIN = -INT_MAX


base_dir = os.path.dirname(os.path.abspath(__file__))
datafile = os.path.join(base_dir, 'tableau.tsv')

def generate_pesudo_conversations(datafile):
    pairs = []
    q = ['what is the uptime of tableau server',
        'uptime of tableau server',
        'uptime of tableau service',
        'uptime tableau',
        'uptime of tableau',
        'tableau uptime']
    for i in range(1, 10000):
        line = "{}\t\{}\n".format(random.choice(q), 'service(uptime,tableau)')
        pairs.append(line)

    q = ['what is the uptime of gpdb server',
        'uptime of gpdb server',
        'uptime of gpdb service',
        'uptime gpdb',
        'uptime of gpdb',
        'gpdb uptime']
    for i in range(1, 10000):
        line = "{}\t\{}\n".format(random.choice(q), 'service(uptime,gpdb)')
        pairs.append(line)
    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        #writer = csv.writer(outputfile, delimiter=delimiter)
        outputfile.writelines(pairs)
#generate_pesudo_conversations(datafile)


def generate_additive(datafile):
    RANGE = 150
    duplicates = 1
    pairs = []
    for i in range(RANGE):
        for j in range(RANGE):
            x = ' '.join([_ for _ in str(i)])
            y = ' '.join([_ for _ in str(j)])
            _sum = ' '.join([_ for _ in str(i+j)])
            pairs.append('{} + {}\t{}\n'.format(x, y, _sum))
    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        #writer = csv.writer(outputfile, delimiter=delimiter)
        outputfile.writelines(pairs)
generate_additive(datafile=os.path.join(params.data_dir, "{}.tsv".format('math_add')))
