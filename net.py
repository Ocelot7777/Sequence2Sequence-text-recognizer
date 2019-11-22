import torch
import torch.nn as nn
import torch.nn.functional as F

import seq_resnet


class Encoder(nn.Module):
    '''
    Args:
        hidden_size: hidden size of LSTM
    Inputs:
        img, (N, C=3, H, W)
        # conv_feature: (N, C=2048, H=1, W)
    Outputs: 
        # output: (seq_len=W, batch_size=N, input_size=2*hidden_size), 2 for bidirectional
        output: (batch_size=N, seq_len=W, input_size=2*hidden_size), 2 for bidirectional
        h_n, c_n: (num_layers*num_directions=4, batch_size=N, hidden_size)
    '''
    def __init__(self, hidden_size=256):
        super().__init__()

        self.conv = seq_resnet.resnet50(stride=(2, 1))
        self.lstm = nn.LSTM(input_size=512*self.conv.block_expansion, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        # # need this?
        # self.embedding = nn.Embedding(input_size, hidden_size)

    def forward(self, x):
        conv_feature = self.conv(x)
        # lstm expects the input of (batch, seq_len, input_size)
        # seq_len <==> W, input_size <==> C
        # (N, C, H=1, W) --> (N, C, W) --> (W, N, C)
        conv_feature = torch.squeeze(conv_feature, 2)
        conv_feature = conv_feature.transpose(1, 2)
        # conv_feature = conv_feature.permute(2, 0, 1)
        output, (h_n, c_n) = self.lstm(conv_feature)

        return output, (h_n, c_n)


class AttentionDecoderRnn(nn.Module):
    '''
    Args:
        hidden_size: hidden size
        output_size: number of classes
        max_label_length: A pre-defined length, used for aligning variable-length sequences.
    Inputs:
        decoder_input: (batch_size, ), last output character of decoder, or the target label (for teacher forcing)
        hidden: hidden state from last output of decoder, (num_layers*num_directions=4, batch_size, hidden_size=256)
        encoder_output: context vector / encoded feature, (batch_size=N, seq_len=W, input_size=2*hidden_size), 2 for bidirectional
    Outputs:
        output: (N, output_size)
        hidden: (h_n, c_n) for LSTM, h_n for GRU
    '''

    def __init__(self, hidden_size, output_size, max_label_length):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_label_length = max_label_length
        
        # Note that we should add an BOS/SOS to start decoder, thus num_embedding=output_size+1
        self.embedding = nn.Embedding(num_embeddings=self.output_size+1, embedding_dim=self.hidden_size)
        # compute the weights
        self.attention = nn.Linear(self.hidden_size * 2, max_label_length)
        self.attention_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=2, batch_first=True)
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.with_attention = False

    def forward(self, decoder_input, hidden, encoder_outputs):
        # print('decoder_intput.shape = ', decoder_input.shape)
        # print('hidden.shape = ', hidden.shape)

        if self.with_attention:
            raise NotImplementedError

            print('encoder_outputs.shape = ', encoder_outputs.shape)
            embedded = self.embedding(decoder_input.long())        # (batch_size, embedding_dim)
            print('embedded.shape = ', embedded.shape)
            print(embedded.dtype)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            hidden = hidden.reshape(-1, hidden.size(2))     # (batch_size * T, hidden_size)
            print(hidden.dtype)
            attention_weights = F.softmax(
                self.attention(torch.cat((embedded, hidden), dim=1)), dim=1
            )
            print('attention_weights.shape = ', attention_weights.shape)
            attention_applied = torch.bmm(attention_weights, encoder_outputs)
            print('attention_applied.shape = ', attention_applied.shape)

            output = torch.cat((embedded, attention_applied), dim=1)
            print('output.shape = ', output.shape)
            output = self.attention_combine(output)
            print('output.shape = ', output.shape)

            output = F.relu(output)
            print('output.shape = ', output.shape)

        else:
            output = self.embedding(decoder_input.long()).unsqueeze(1)      # (batch_size, embedding_dim) --> (batch_size, seq_len=1, embedding_dim)
            attention_weights = None

        # output, hidden = self.gru(output, hidden)
        output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output.squeeze(dim=1)), dim=1)

        return output, hidden, attention_weights


class Seq2SeqNetwork(nn.Module):
    '''
    Args:
        hidden_size: hidden size of LSTM/GRU in both encoder and decoder
        output_size: output size of decoder, equals to num_classes, including EOS, PADDING, UNKNOWN, except BOS
        max_label_length: maximum length of input sequence
    Inputs:
        img: (N, C=3, H=32, W)
        label: (N, max_label_length)
        label_lengths: lengths for each label, (N, )
    Outputs:
        outputs: (batch_size, max(label_lengths), output_size)
    '''
    
    def __init__(self, hidden_size, output_size, max_label_length):
        super().__init__()

        self.encoder = Encoder(hidden_size)
        self.decoder = AttentionDecoderRnn(hidden_size, output_size, max_label_length)
        self.output_size = output_size
        self.teacher_forcing = True

    def forward(self, img, label, label_lengths):

        encoder_output, _ = self.encoder(img)
        batch_size, seq_len, _ = encoder_output.size()

        decoder_input, hidden_state = torch.zeros((batch_size)).fill_(self.output_size), None
        outputs = []

        for i in range(max(label_lengths)):
            decoder_output, hidden_state, attention_weights = self.decoder(decoder_input, hidden_state, encoder_output)
            outputs.append(decoder_output)
            # if self.training and self.teacher_forcing:
            #     decoder_input = label[:, i]
            # else:
            #     decoder_input = torch.argmax(decoder_output, dim=-1)
            decoder_input = label[:, i]

        outputs = torch.cat([output.unsqueeze(dim=1) for output in outputs], dim=1)
        return outputs



if __name__ == "__main__":
    # encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    # encoder_layer = Encoder()
    # decoder = AttentionDecoderRnn(hidden_size=256, output_size=28, max_length=100)
    # print(encoder_layer)
    batch_size = 5
    output_size = 28
    hidden_size = 256
    seq_len = 100
    # transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    # print(transformer_encoder)
    from torchsummaryX import summary
    import random
    img = torch.rand(batch_size, 3, 32, 128)
    labels = torch.randint(low=0, high=28, size=(batch_size, seq_len))
    label_lengths = [random.randint(0, seq_len) for _ in range(seq_len)]
    max_label_length = 64
    # summary(encoder_layer, src)
    # encoder_output, hidden = encoder_layer(src)
    # # fake_hidden = torch.randint(low=0, high=28, size=(batch_size, output_size))
    # # fake_hidden = torch.rand(hbatch_size, output_size)
    # # fake_decoder_input = torch.randint(low=0, high=28, size=(output_size, batch_size, hidden_size))
    # fake_decoder_input = torch.rand(batch_size, seq_len, hidden_size)
    # # fake_decoder_input = torch.tensor([[0]])
    # output, hidden, attention_weights = decoder(fake_decoder_input, hidden=None, encoder_outputs=encoder_output)

    model = Seq2SeqNetwork(hidden_size, output_size, max_label_length)
    output = model(img, labels, label_lengths)
    print(output.shape)