"""generate pesudo conversations
"""
import os
import random

base_dir = os.path.dirname(os.path.abspath(__file__))
datafile = os.path.join(base_dir, 'tableau.tsv')

def generate_pesudo_conversations(datafile):
    pairs = []
    q = ['what is the uptime of tableau server',
        'uptime of tableau server',
        'uptime of tableau',
        'tableau uptime',
        'tableau']
    for i in range(1, 10000):
        line = "{}\t\{}\n".format(random.choice(q), 'select uptime from table where service="tableu"')
        pairs.append(line)
    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        #writer = csv.writer(outputfile, delimiter=delimiter)
        outputfile.writelines(pairs)
generate_pesudo_conversations(datafile)