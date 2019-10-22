import torch
import torch.nn as nn
import torch.nn.functional as F

import seq_resnet


class Encoder(nn.Module):
    '''
    Input: (N, C=3, H, W)
    conv_feature: (N, C=2048, H=1, W)
    Output: 
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
        print('x.shape = ', x.shape)
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
    Input:
        decoder_input: last output of decoder, or the target label (for teacher forcing)
        hidden: hidden state from last output of decoder, (num_layers*num_directions=4, batch_size, hidden_size=256)
        encoder_output: context vector / encoded feature, (batch_size=N, seq_len=W, input_size=2*hidden_size), 2 for bidirectional
    Output:
        output: (N, seq_len, output_size)
        hidden: (h_n, c_n) for LSTM, h_n for GRU
    '''

    def __init__(self, hidden_size, output_size, max_length):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        
        self.embedding = nn.Embedding(num_embeddings=self.output_size, embedding_dim=self.hidden_size)
        # compute the weights
        # self.attention = nn.Linear(self.hidden_size * 2, max_length)
        # self.attention_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=2, bidirectional=2)
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(self.hidden_size*2, self.output_size)
        self.with_attention = None

    def forward(self, decoder_input, hidden, encoder_outputs):
        # print('decoder_intput.shape = ', decoder_input.shape)
        # print('hidden.shape = ', hidden.shape)

        if self.with_attention:
            raise NotImplementedError

            print('encoder_outputs.shape = ', encoder_outputs.shape)
            embedded = self.embedding(decoder_input)        # (x.size, embedding_dim)
            print('embedded.shape = ', embedded.shape)
            print(embedded.dtype)
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
            output = decoder_input
            attention_weights = None

        # output, hidden = self.gru(output, hidden)
        output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output), dim=1)

        return output, hidden, attention_weights



if __name__ == "__main__":
    # encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    encoder_layer = Encoder()
    decoder = AttentionDecoderRnn(hidden_size=256, output_size=28, max_length=100)
    print(encoder_layer)
    batch_size = 5
    output_size = 28
    hidden_size = 256
    seq_len = 100
    # transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    # print(transformer_encoder)
    from torchsummaryX import summary
    src = torch.rand(5, 3, 32, 128)
    # summary(encoder_layer, src)
    encoder_output, hidden = encoder_layer(src)
    # fake_hidden = torch.randint(low=0, high=28, size=(batch_size, output_size))
    # fake_hidden = torch.rand(hbatch_size, output_size)
    # fake_decoder_input = torch.randint(low=0, high=28, size=(output_size, batch_size, hidden_size))
    fake_decoder_input = torch.rand(batch_size, seq_len, hidden_size)
    # fake_decoder_input = torch.tensor([[0]])
    output, hidden, attention_weights = decoder(fake_decoder_input, hidden, encoder_output)