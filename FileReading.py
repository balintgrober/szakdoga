import os
import re
import unicodedata

from Lang import Lang


def unicodetoAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodetoAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    hu_files = os.listdir("separate/hungarian")
    en_files = os.listdir("separate/english")

    en_lines = []
    hu_lines = []

    for en_file_name in en_files:
        en_file = open("separate/english/" + en_file_name).read().strip().split('\n')
        for en_line in en_file:
            en_lines.append(en_line)


    for hu_file_name in hu_files:
        hu_file = open("separate/hungarian/" + hu_file_name).read().strip().split('\n')
        for hu_line in hu_file:
            hu_lines.append(hu_line)

    zipped = list(map(list, zip(en_lines, hu_lines)))
    pairs = [[ normalizeString(s) for s in p]for p in zipped]

    norm_hun = [normalizeString(s) for s in hu_lines]
    norm_eng = [normalizeString(s) for s in en_lines]

    hun_word_counts = []
    eng_word_counts = []
    for sentence in norm_hun:
        hun_word_counts.append(len(sentence.split(' ')))

    for sentence in norm_eng:
        eng_word_counts.append(len(sentence.split(' ')))

    print("max_hun:", max(hun_word_counts))
    print("max_eng:", max(eng_word_counts))

    frequency = {}

    for i in eng_word_counts:
        if i in frequency:
            frequency[i] += 1
        else:
            frequency[i] = 1

    frequency = dict(sorted(frequency.items()))
    print("English length frequencies:", frequency)

    freq = []
    for x in list(frequency)[60:]:
        freq.append(frequency[x])

    print("Sentences with more than 60 words:", sum(freq))

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs
