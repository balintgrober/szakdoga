import math
import os
import random
import sys
import time
import torch
from torch import optim, nn
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from nltk.translate.bleu_score import sentence_bleu
from sklearn.model_selection import train_test_split

import DecoderRNN
import EncoderRNN

import matplotlib.ticker as ticker
import numpy as np


import FileReading

SOS_token = 0
EOS_token = 1
teacher_forcing_ratio = 0.5

def filterPair(p):
    return len(p[0].split(' ')) < 30 and len(p[1].split(' ')) < 30

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = FileReading.readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    max_len = 0
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
        for sentence in pair:
            if len(sentence.split(' ')) > max_len:
                max_len = len(sentence.split(' '))

    print("The longest sentence in the teaching corpus has", max_len, "words")
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
    encoder_h_hidden = encoder.initHidden()
    encoder_c_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(30, encoder.hidden_size, device="cuda")
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hiddens = encoder(input_tensor[ei], encoder_h_hidden, encoder_c_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
        encoder_h_hidden = encoder_hiddens[0]
        encoder_c_hidden = encoder_hiddens[1]


    decoder_input = torch.tensor([[SOS_token]], device="cuda")

    decoder_h_hidden = encoder_h_hidden
    decoder_c_hidden = encoder_c_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hiddens, decoder_attention = decoder(decoder_input, decoder_h_hidden, decoder_c_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
            decoder_h_hidden = decoder_hiddens[0]
            decoder_c_hidden = decoder_hiddens[1]
    else:
        for di in range(target_length):
            decoder_output, decoder_hiddens, decoder_attention = decoder(decoder_input, decoder_h_hidden, decoder_c_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            decoder_h_hidden = decoder_hiddens[0]
            decoder_c_hidden = decoder_hiddens[1]

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

def trainIters(encoder, decoder, epochs, print_every=1000, plot_every=1000, learning_rate=0.001):

    plot_losses = []

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorFromPair(pair) for pair in train_pairs]
    criterion = nn.NLLLoss()

    for epoch in range(1, epochs + 1):
        start = time.time()
        print_loss_total = 0
        plot_loss_total = 0
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
    plt.title('Training Results')
    plt.savefig('training_loss.png')
    plt.close('all')

def evaluate(encoder, decoder, sentence):
    with torch.no_grad():
        input_tensor = tensorFromSentece(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_h_hidden = encoder.initHidden()
        encoder_c_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(30, encoder.hidden_size, device="cuda")

        for ei in range(input_length):
            encoder_output, encoder_hiddens = encoder(input_tensor[ei], encoder_h_hidden, encoder_c_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
            encoder_h_hidden = encoder_hiddens[0]
            encoder_c_hidden = encoder_hiddens[1]

        decoder_input = torch.tensor([[SOS_token]], device="cuda")

        decoder_h_hidden = encoder_h_hidden
        decoder_c_hidden = encoder_c_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(30, 30)

        for di in range(30):
            decoder_output, decoder_hiddens, decoder_attention = decoder(decoder_input, decoder_h_hidden, decoder_c_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()
            decoder_h_hidden = decoder_hiddens[0]
            decoder_c_hidden = decoder_hiddens[1]

        return decoded_words, decoder_attentions[:di + 1]


def calculate_bleu_score_test(encoder, decoder):
    sum = 0
    for i in range(len(test_pairs)):
        pair = test_pairs[i]
        candidate_words, attentions = evaluate(encoder, decoder, pair[0])
        candidate = ' '.join(candidate_words)
        reference = [pair[1]]
        BLEU_score = sentence_bleu(reference, candidate)
        sum += BLEU_score

    avg_BLEU = sum / len(pairs)
    print("BLEU test score -> {}".format(avg_BLEU))

def calculate_bleu_score_train(encoder, decoder):
    sum = 0
    for i in range(len(train_pairs)):
        pair = train_pairs[i]
        candidate_words, attentions = evaluate(encoder, decoder, pair[0])
        candidate = ' '.join(candidate_words)
        reference = [pair[1]]
        BLEU_score = sentence_bleu(reference, candidate)
        sum += BLEU_score

    avg_BLEU = sum / len(pairs)
    print("BLEU train score -> {}".format(avg_BLEU))

def evaluateRandomly(encoder, decoder, n=10):

    print("From train set:")
    for i in range(n):
        pair = random.choice(train_pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


    print("From test set:")
    for i in range(n):
        pair = random.choice(test_pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def showAttention(input_sentence, output_words, attentions, fig_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(fig_name + ".png")

    plt.close("all")

def evaluateAndShowAttention(input_sentence, encoder, decoder, fig_name):
    output_words, attentions = evaluate(
        encoder, decoder, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions, fig_name)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    input_lang, output_lang, pairs = prepareData('eng', 'hun', True)

    train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, train_size=0.8, random_state=0)

    train_mode = True
    hidden_size = 1024

    input = input("Train or load models?:")

    if input == "train":
        train_mode = True
    elif input == "load":
        train_mode = False
    else:
        sys.exit("Wrong input!")

    if train_mode:
        encoder1 = EncoderRNN.EncoderRNN(input_lang.n_words, hidden_size).to("cuda")
        attn_decoder1 = DecoderRNN.AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to("cuda")

        epochs = 10

        trainIters(encoder1, attn_decoder1, epochs, print_every=10000)
        evaluateRandomly(encoder1, attn_decoder1)
        torch.save(encoder1.state_dict(), "model/encoder.pt")
        torch.save(attn_decoder1.state_dict(), "model/decoder.pt")
        evaluateAndShowAttention(random.choice(train_pairs)[0], encoder1, attn_decoder1, "attention1")
        evaluateAndShowAttention(random.choice(train_pairs)[0], encoder1, attn_decoder1, "attention2")
        evaluateAndShowAttention(random.choice(test_pairs)[0], encoder1, attn_decoder1, "attention3")
        evaluateAndShowAttention(random.choice(test_pairs)[0], encoder1, attn_decoder1, "attention4")
        calculate_bleu_score_train(encoder1, attn_decoder1)
        calculate_bleu_score_test(encoder1, attn_decoder1)

    else:
        encoder = EncoderRNN.EncoderRNN(input_lang.n_words, hidden_size).to("cuda")
        attn_decoder = DecoderRNN.AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to("cuda")
        encoder.load_state_dict(torch.load("model/encoder.pt"))
        attn_decoder.load_state_dict(torch.load("model/decoder.pt"))
        encoder.eval()
        attn_decoder.eval()
        evaluateRandomly(encoder, attn_decoder)
        evaluateAndShowAttention(random.choice(train_pairs)[0], encoder, attn_decoder, "attention1")
        evaluateAndShowAttention(random.choice(train_pairs)[0], encoder, attn_decoder, "attention2")
        evaluateAndShowAttention(random.choice(test_pairs)[0], encoder, attn_decoder, "attention3")
        evaluateAndShowAttention(random.choice(test_pairs)[0], encoder, attn_decoder, "attention4")
        calculate_bleu_score_train(encoder, attn_decoder)
        calculate_bleu_score_test(encoder, attn_decoder)
