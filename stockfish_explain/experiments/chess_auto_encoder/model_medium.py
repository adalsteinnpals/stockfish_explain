import torch


class DeepAutoencoder(torch.nn.Module):
    def __init__(self,
                 input_size):
        super().__init__()  
        self.input_size = input_size      
        self.size_0 = 1024
        self.size_1 = 64
        self.size_2 = 32

        self.encoder_0 = torch.nn.Linear(self.input_size, self.size_0)
        self.encoder_1 = torch.nn.Linear(self.size_0, self.size_1)
        self.encoder_2 = torch.nn.Linear(self.size_1, self.size_2)

        self.decoder_1 = torch.nn.Linear(self.size_2, self.size_1)
        self.decoder_2 = torch.nn.Linear(self.size_1, self.size_0)
        self.decoder_3 = torch.nn.Linear(self.size_0, self.input_size)

        self.encoder = torch.nn.Sequential(
            self.encoder_0,
            torch.nn.ReLU(),
            self.encoder_1,
            torch.nn.ReLU(),
            self.encoder_2,
        )
          
        self.decoder = torch.nn.Sequential(
            self.decoder_1,
            torch.nn.ReLU(),
            self.decoder_2,
            torch.nn.ReLU(),
            self.decoder_3,
            torch.nn.Sigmoid()
        )
  
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
