import math
import os
import random
import time
import torch
from torch import optim, nn
import matplotlib.pyplot as plt

import DecoderRNN
import EncoderRNN

plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


import FileReading

SOS_token = 0
EOS_token = 1
teacher_forcing_ratio = 0.5


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = FileReading.readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    max_len = 0
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
        for sentence in pair:
            if len(sentence.split(' ')) > max_len:
                max_len = len(sentence.split(' '))

    print("The longest sentence in the corpus has", max_len, "words")
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

def indexesFromSentece(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentece(lang, sentence):
    indexes = indexesFromSentece(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device="cuda").view(-1, 1)

def tensorFromPair(pair):
    input_tensor = tensorFromSentece(input_lang, pair[0])
    target_tensor = tensorFromSentece(output_lang, pair[1])
    return (input_tensor, target_tensor)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(289, encoder.hidden_size, device="cuda")
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device="cuda")

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(encoder, decoder, epochs, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorFromPair(pair) for pair in pairs]
    criterion = nn.NLLLoss()

    for epoch in range(1, epochs + 1):
        print("Epoch%d:" % (epoch))
        for i in range(1, len(training_pairs)):
            training_pair = training_pairs[i - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if i % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (
                timeSince(start, i / len(training_pairs)), i, i / len(training_pairs) * 100, print_loss_avg))

            if i % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    showPlot(plot_losses)


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def evaluate(encoder, decoder, sentence):
    with torch.no_grad():
        input_tensor = tensorFromSentece(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(289, encoder.hidden_size, device="cuda")

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device="cuda")

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(289, 289)

        for di in range(10):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    input_lang, output_lang, pairs = prepareData('eng', 'hun', True)

    hidden_size = 256
    encoder1 = EncoderRNN.EncoderRNN(input_lang.n_words, hidden_size).to("cuda")
    attn_decoder1 = DecoderRNN.AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to("cuda")

    epochs = 10

    trainIters(encoder1, attn_decoder1, epochs, print_every=1)
    evaluateRandomly(encoder1, attn_decoder1)
    torch.save(encoder1.state_dict(), "/model/encoder.txt")
    torch.save(attn_decoder1.state_dict(), "/model/decoder.txt")