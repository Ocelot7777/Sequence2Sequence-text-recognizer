import torch
import torch.nn as nn
import torch.nn.functional as F

import seq_resnet
from temp_net import recog_net


BACKBONES = {'resnet18': seq_resnet.resnet18, 'resnet34': seq_resnet.resnet34, 'resnet50': seq_resnet.resnet50, 'temp_net': recog_net}

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
    def __init__(self, backbone, hidden_size=256):
        super().__init__()

        assert backbone in BACKBONES.keys(), 'Only {} are supported! '.format(BACKBONES.keys())
        if backbone == 'temp_net':
            self.conv = BACKBONES[backbone]()
        else:
            self.conv = BACKBONES[backbone](stride=(2, 1))
        print('Backbone: {}'.format(backbone))
        self.lstm = nn.LSTM(input_size=512*self.conv.block_expansion, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)

    def forward(self, x):
        conv_feature = self.conv(x)
        # lstm expects the input of (batch, seq_len, input_size)
        # seq_len <==> W, input_size <==> C
        # (N, C, H=1, W) --> (N, C, W) --> (W, N, C)
        conv_feature = torch.squeeze(conv_feature, 2)
        conv_feature = conv_feature.transpose(1, 2)
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

    def __init__(self, hidden_size, output_size, max_label_length, with_attention):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_label_length = max_label_length
        self.num_directions = 2
        self.num_lstm_layers = 2
        self.simple_attention = False
        
        # Note that we should add an BOS/SOS to start decoder, thus num_embedding=output_size+1
        self.embedding = nn.Embedding(num_embeddings=self.output_size+1, embedding_dim=self.hidden_size)
        self.with_attention = with_attention
        if self.with_attention:
            if self.simple_attention:
                self.attention = SimpleAttentionUnit(self.hidden_size, self.num_directions)
            else:
                self.attention = AttentionUnit(self.hidden_size * self.num_lstm_layers * self.num_directions, self.hidden_size * 2, self.hidden_size * 2)
            # self.gru = nn.GRU(input_size=self.hidden_size * 3, hidden_size=self.hidden_size, batch_first=True)
            self.lstm = nn.LSTM(input_size=self.hidden_size * 3, hidden_size=self.hidden_size, batch_first=True, num_layers=2, bidirectional=True)
        else:
            self.attention = None
            self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True, num_layers=2, bidirectional=True)
            self.num_directions = 2
            print('No attention is used.')
        self.out = nn.Linear(self.hidden_size * self.num_directions, self.output_size)

    def forward(self, decoder_input, hidden, encoder_outputs):
        if self.with_attention:
            hidden_state, cell_state = hidden
            embedded = self.embedding(decoder_input.long())        # (batch_size, embedding_dim)
            attention_weights = self.attention(encoder_outputs, hidden_state)         # (batch_size, seq_len)
            context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)       # context, (batch_size, 1, 2 * hidden_size)

            output = torch.cat((embedded, context_vector.squeeze(1)), dim=1)

            output, hidden_state = self.lstm(
                torch.cat((embedded, context_vector.squeeze(1)), dim=1).unsqueeze(1),
                hidden
            )       # output, (batch_size, 1, hidden_size)
            output = self.out(output.squeeze(dim=1))

            return output, hidden_state, attention_weights

        else:
            output = self.embedding(decoder_input.long()).unsqueeze(1)      # (batch_size, embedding_dim) --> (batch_size, seq_len=1, embedding_dim)
            attention_weights = None

            # output, hidden = self.gru(output, hidden)
            output, hidden = self.lstm(output, hidden)

            output = self.out(output.squeeze(dim=1))

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
    
    def __init__(self, hidden_size, output_size, max_label_length, with_attention=True, backbone='resnet50', teacher_forcing=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_label_length = max_label_length

        self.encoder = Encoder(backbone, hidden_size)
        self.decoder = AttentionDecoderRnn(hidden_size, output_size, max_label_length, with_attention)
        self.output_size = output_size
        self.teacher_forcing = teacher_forcing

    def forward(self, img, label, label_lengths):

        encoder_output, hidden = self.encoder(img)
        batch_size, seq_len, _ = encoder_output.size()

        # decoder_input, hidden_state = torch.zeros((batch_size)).fill_(self.output_size), torch.zeros(1, batch_size, self.hidden_size)
        decoder_input = torch.zeros((batch_size)).fill_(self.output_size)
        outputs = []

        for i in range(max(label_lengths)):
            decoder_output, hidden, attention_weights = self.decoder(decoder_input, hidden, encoder_output)
            outputs.append(decoder_output)
            if self.teacher_forcing:
                decoder_input = label[:, i]
            else:
                decoder_input = torch.argmax(decoder_output, dim=-1)


        outputs = torch.cat([output.unsqueeze(dim=1) for output in outputs], dim=1)
        return outputs


class AttentionUnit(nn.Module):
    '''
    Args:
        sDim: hidden_state dim of encoder lstm
        xDim: encoder output_size
        attDim: 
    '''
    def __init__(self, sDim, xDim, attDim):
        super(AttentionUnit, self).__init__()

        self.sDim = sDim
        self.xDim = xDim
        self.attDim = attDim

        self.sEmbed = nn.Linear(sDim, attDim)
        self.xEmbed = nn.Linear(xDim, attDim)
        self.wEmbed = nn.Linear(attDim, 1)

        # self.init_weights()

    def init_weights(self):
        init.normal_(self.sEmbed.weight, std=0.01)
        init.constant_(self.sEmbed.bias, 0)
        init.normal_(self.xEmbed.weight, std=0.01)
        init.constant_(self.xEmbed.bias, 0)
        init.normal_(self.wEmbed.weight, std=0.01)
        init.constant_(self.wEmbed.bias, 0)

    def forward(self, x, sPrev):
        batch_size, T, _ = x.size()                      # [b x T x xDim]
        x = x.reshape(-1, self.xDim)                        # [(b x T) x xDim]
        xProj = self.xEmbed(x)                           # [(b x T) x attDim]
        xProj = xProj.reshape(batch_size, T, -1)            # [b x T x attDim]

        if sPrev.shape[0] == 1:
            sPrev = sPrev.squeeze(0)
        else:
            sPrev = sPrev.permute(1, 0, 2)
            sPrev = sPrev.reshape(batch_size, -1)
        sProj = self.sEmbed(sPrev)                       # [b x attDim]
        sProj = torch.unsqueeze(sProj, 1)                # [b x 1 x attDim]
        sProj = sProj.expand(batch_size, T, self.attDim) # [b x T x attDim]

        sumTanh = torch.tanh(sProj + xProj)
        sumTanh = sumTanh.reshape(-1, self.attDim)

        vProj = self.wEmbed(sumTanh) # [(b x T) x 1]
        vProj = vProj.reshape(batch_size, T)

        alpha = F.softmax(vProj, dim=1) # attention weights for each sample in the minibatch

        return alpha


class SimpleAttentionUnit(nn.Module):
    '''A simple attention unit similar to machine translation
    Args:
        hidden_size:
        num_directions: The paramters used in the LSTM layer of encoder
    Inputs:
        hidden_state: The final hidden state of the LSTM in encoder, not including the cell state. (num_layers * num_directions, batch_size, hidden_size)
        encoder_outputs: (batch_size, seq_len, hidden_size * num_directions)
    '''
    
    def __init__(self, hidden_size, num_directions):

        super().__init__()

        self.hidden_size = hidden_size
        self.num_directions = num_directions
        self.attn = nn.Linear(self.hidden_size * (num_directions + 1), self.hidden_size * self.num_directions)
        self.attn_combine = nn.Linear(self.hidden_size * num_directions, 1)

    def forward(self, encoder_outputs, hidden_state):
        batch_size, seq_len, _ = encoder_outputs.size()
        if hidden_state.dim() == 3:
            hidden_state = hidden_state[-1]     # --> (batch_size, hidden_size)
        hidden_state = hidden_state.repeat(seq_len, 1, 1).transpose(1, 0)       # --> (batch_size, seq_len, hidden_size)

        energy = torch.tanh(self.attn(torch.cat(
            (hidden_state, encoder_outputs), dim=-1
        )))        # --> (batch_size, seq_len, hidden_size * num_directions)

        energy = self.attn_combine(energy)      # --> (batch_size, seq_len, 1)

        attention_weights = F.softmax(
            energy.reshape(batch_size, seq_len), dim=-1
        )       # (batch_size, seq_len)
        return attention_weights


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

    model = Seq2SeqNetwork(hidden_size, output_size, max_label_length, with_attention=True, backbone='temp_net')
    output = model(img, labels, label_lengths)
    print('output.shape = ', output.shape)
    # summary(model, img, labels, label_lengths)