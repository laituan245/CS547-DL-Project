import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, input_dim=128, n_filters=256, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, n_filters, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=padding)
        self.batchnorm, self.relu = nn.BatchNorm1d(n_filters), nn.ReLU()
    def forward(self, x):
        x = self.relu(self.batchnorm(self.conv1(x)))
        x = self.relu(self.batchnorm(self.conv2(x)))
        return x


class VDCNN(nn.Module):

    def __init__(self, n_classes=14):
        super(VDCNN, self).__init__()
        self.embed = nn.Embedding(num_embeddings=69, embedding_dim=16, padding_idx=0)

        layers = [nn.Conv1d(16, 64, kernel_size=3, padding=1)]
        for _ in range(4):
            layers.append(ConvBlock(input_dim=64, n_filters=64, kernel_size=3, padding=1))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        layers.append(ConvBlock(input_dim=64, n_filters=2*64, kernel_size=3, padding=1))
        for _ in range(4):
            layers.append(ConvBlock(input_dim=2*64, n_filters=2*64, kernel_size=3, padding=1))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        layers.append(ConvBlock(input_dim=2*64, n_filters=4*64, kernel_size=3, padding=1))
        for _ in range(1):
            layers.append(ConvBlock(input_dim=4*64, n_filters=4*64, kernel_size=3, padding=1))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        layers.append(ConvBlock(input_dim=4*64, n_filters=8*64, kernel_size=3, padding=1))
        for _ in range(1):
            layers.append(ConvBlock(input_dim=8*64, n_filters=8*64, kernel_size=3, padding=1))
        layers.append(nn.AdaptiveMaxPool1d(8))
        self.layers = nn.Sequential(*layers)

        self.cls = nn.Sequential(
            nn.Linear(8*8*8*8, 2048), nn.ReLU(), \
            nn.Linear(2048, 2048), nn.ReLU(), \
            nn.Linear(2048, n_classes)
            )
            
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.embed(x).transpose(1,2)
        x = self.layers(x).view(x.size(0),-1)
        return self.cls(x)

    def compute_loss(self, pred_y, true_y):
        return self.ce_loss(pred_y, true_y)
