from dataset.ocr_dataloader import ResizeNormalize, AlignCollate, TextLineDataset, RandomSequentialSampler
from torch.utils.data import DataLoader
from dataclasses import dataclass
import torch.nn as nn
from model.cnn_model.crnn import CRNN, Decoder
from utils.parameter_initialization import weights_init
from utils.ocr_convert import ConvertBetweenStringAndLabel, SOS_TOKEN, EOS_TOKEN
import torch
from model.metric import Averager
import random


with open('mini_data/ocr_data/char_std_5990.txt', encoding="utf-8") as f:
    data = f.readlines()
    alphabet = [x.rstrip() for x in data]
    alphabet = ''.join(alphabet)

num_classes = len(alphabet) + 2


@dataclass
class Config:
    train_file: str
    test_file: str
    img_height: int = 32
    img_width: int = 280
    batch_size: int = 16
    num_worders: int = 0
    hidden_size: int = 256
    max_width: int = 71
    learning_rate: float = 0.0001
    num_epochs: int = 2
    teach_forcing_prob: float = 0.5
    model_save_dir: str = "./model/"

def load_data(v, data):
    with torch.no_grad():
        v.resize_(data.size()).copy_(data)

config = Config(train_file="mini_data/ocr_data/train_info.txt",
                test_file="mini_data/ocr_data/test_info.txt",
                batch_size=32)
converter = ConvertBetweenStringAndLabel(alphabet)
train_dataset = TextLineDataset(config.train_file)
test_dataset = TextLineDataset(text_line_file=config.test_file,
                               transform=ResizeNormalize(img_width=config.img_width,
                                                         img_height=config.img_height))

sampler = RandomSequentialSampler(train_dataset, config.batch_size)
train_loader = DataLoader(train_dataset,
                          batch_size=config.batch_size,
                          shuffle=False,
                          sampler=sampler,
                          num_workers=config.num_worders,
                          collate_fn=AlignCollate(config.img_height, config.img_width))
test_dataloader = DataLoader(test_dataset,
                             shuffle=False,
                             batch_size=1,
                             num_workers=config.num_worders)

encoder = CRNN(channel_size=3, hidden_size=config.hidden_size)
decoder = Decoder(config.hidden_size, output_size=num_classes, dropout_p=0.1, max_length=config.max_width)
encoder.apply(weights_init)
decoder.apply(weights_init)
criterion = nn.NLLLoss()


image = torch.FloatTensor(config.batch_size, 3, config.img_height, config.img_width)
text = torch.LongTensor(config.batch_size)

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
loss_avg = Averager()

#-------------------------- Traing ---------------------------
for epoch in range(config.num_epochs):
    train_iter = iter(train_loader)
    for i in range(len(train_loader)):
        cpu_images, cpu_texts = next(train_iter)
        batch_size = cpu_images.size(0)
        for encoder_param, decoder_param in zip(encoder.parameters(), decoder.parameters()):
            encoder_param.requires_grad = True
            decoder_param.requires_grad = True
        encoder.train()
        decoder.train()
        target_variable = converter.encode(cpu_texts)
        load_data(image, cpu_images)

        encoder_outputs = encoder(image)
        decoder_input = target_variable[SOS_TOKEN]
        decoder_hidden = decoder.initHidden(batch_size)

        loss = 0.0
        teach_forcing = True if random.random() > config.teach_forcing_prob else False
        if teach_forcing:
            for di in range(1, target_variable.shape[0]):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_variable[di])
                decoder_input = target_variable[di]
        else:
            for di in range(1, target_variable.shape[0]):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_variable[di])
                topv, topi = decoder_output.data.topk(1)
                ni = topi.squeeze()
                decoder_input = ni

        encoder.zero_grad()
        decoder.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        loss_avg.add(loss)
        if i % 10 == 0:
            print('[Epoch {0}/{1}] [Batch {2}/{3}] Loss: {4}'.format(epoch, config.num_epochs, i, len(train_loader),
                                                                     loss_avg.val()))
            loss_avg.reset()

        # torch.save(encoder.state_dict(), '{0}/encoder_{1}.pth'.format(config.model_save_dir, epoch))
        # torch.save(decoder.state_dict(), '{0}/decoder_{1}.pth'.format(config.model_save_dir, epoch))


#-------------------------- Testing ---------------------------
for e, d in zip(encoder.parameters(), decoder.parameters()):
    e.requires_grad = False
    d.requires_grad = False

encoder.eval()
decoder.eval()
val_iter = iter(test_dataloader)

n_correct = 0
n_total = 0
loss_avg = Averager()

for i in range(min(len(test_dataloader), 100)):
    cpu_images, cpu_texts = next(val_iter)
    batch_size = cpu_images.size(0)
    load_data(image, cpu_images)

    target_variable = converter.encode(cpu_texts)
    n_total += len(cpu_texts[0]) + 1

    decoded_words = []
    decoded_label = []
    encoder_outputs = encoder(image)
    decoder_input = target_variable[0]
    decoder_hidden = decoder.initHidden(batch_size)

    for di in range(1, target_variable.shape[0]):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.data.topk(1)
        ni = topi.squeeze(1)
        decoder_input = ni
        if ni == EOS_TOKEN:
            decoded_label.append(EOS_TOKEN)
            break
        else:
            decoded_words.append(converter.decode(ni))
            decoded_label.append(ni)

    for pred, target in zip(decoded_label, target_variable[1: , :]):
        if pred == target:
            n_correct += 1

    if i % 10 == 0:
        texts = cpu_texts[0]
        print('pred: {}, gt: {}'.format(''.join(decoded_words), texts))

accuracy = n_correct / float(n_total)
print('Test loss: {}, accuray: {}'.format(loss_avg.val(), accuracy))
