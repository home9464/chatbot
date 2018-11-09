"""

wget --recursive --no-clobber --no-parent https://fangj.github.io/friends/
"""
import os
import re
import csv
import html
import string
from bs4 import BeautifulSoup
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'data/friends/season/')


def normalize_string(text):
    text = re.sub(r'\n', r' ', text.lower())
    text = re.sub(r'\(.*\)', r' ', text)
    text = re.sub(r"([.!?])", r" \1", text)
    #text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
    text = re.sub(r"[^a-zA-Z]+", r" ", text)
    text = re.sub(r"\s+", r" ", text).strip()
    return text


def preprocess(datafile, delimiter='\t'):
    """convert raw files into paired conversations like "what is your name<TAB>Tom Hanks"
    and save it as a new file
    print(files[0])
    #files[0] = '/Users/yingsun/Desktop/workspace/chatbot/data/friends/season/0421.html'
    """

    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    pairs = []
    for htmlfile in files:
        with open(htmlfile, 'r', encoding="ISO-8859-1") as f:
            lines = []
            htmltxt = f.read()
            htmltxt = html.unescape(htmltxt)  # escape html tag to ascii, '&amp;' -> '&'
            text = re.sub(r'[^\x00-\x7f]', r'', htmltxt)  # remove non-ascii characters
            soup = BeautifulSoup(htmltxt, 'lxml')
            conversations = soup.find_all('p')
            for i in range(1, len(conversations)-1):
                text = normalize_string(conversations[i].text)
                text = ' '.join(text.split(' ')[1:])  # the 1st word is the speaker
                lines.append(text)
        for idx in range(len(lines)-1):
            if lines[idx] and lines[idx+1]:
                pairs.append("{}\t{}\n".format(lines[idx], lines[idx+1]))

    # write new csv file
    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        #writer = csv.writer(outputfile, delimiter=delimiter)
        outputfile.writelines(pairs)
        #for pair in pairs:
        #    writer.writerow(pair)

#preprocess(os.path.join(base_dir, 'friends.tsv'))
