import torch
import torch.nn.functional as F
import config


class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm=False,
        use_maxpooling=False,
        pool_params=(2, 2),
        conv_params=(3, 1, 1),
    ):
        super().__init__()
        layers = [
            torch.nn.Conv2d(in_channels, out_channels, *conv_params),
            torch.nn.ReLU(),
        ]

        if use_batchnorm:
            layers.append(torch.nn.BatchNorm2d(out_channels))

        if use_maxpooling:
            layers.append(torch.nn.MaxPool2d(*pool_params))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.layers(x)
        return output


class BidirectionalLSTM(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()

        self.rnn = torch.nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.embedding = torch.nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.rnn.flatten_parameters()

        recurrent, _ = self.rnn(x)

        t, b, h = recurrent.size()
        t_rec = recurrent.view(t * b, h)

        output = self.embedding(t_rec)
        output = output.view(t, b, -1)

        return output


class CRNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        conv_layers = [
            ConvBlock(1, 64, use_maxpooling=True),
            ConvBlock(64, 128, use_maxpooling=True),
            ConvBlock(128, 256),
            ConvBlock(256, 256, use_maxpooling=True, pool_params=((1, 2), 2)),
            ConvBlock(256, 512, use_batchnorm=True),
            ConvBlock(
                512,
                512,
                use_batchnorm=True,
                use_maxpooling=True,
                pool_params=((1, 2), 2),
            ),
            ConvBlock(512, 512, conv_params=(2, 1, 0)),
        ]

        self.cnn = torch.nn.Sequential(*conv_layers)

        rnn_layers = [
            BidirectionalLSTM(512, 256, 256),
            BidirectionalLSTM(256, 256, len(config.ALPHABET)),
        ]

        self.rnn = torch.nn.Sequential(*rnn_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_output = self.cnn(x)
        b, c, h, w = conv_output.shape

        conv_output = conv_output.reshape(b, -1, w)
        conv_output = conv_output.permute(2, 0, 1)

        rnn_output = self.rnn(conv_output)
        output = rnn_output.transpose(1, 0)

        for i in range(output.shape[0]):
            output[i] = F.log_softmax(output[i], dim=-1)

        return output
