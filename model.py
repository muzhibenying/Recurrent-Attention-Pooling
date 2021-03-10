class Atten_Recurrent_Net(nn.Module):
    def __init__(self, L = 1024, dropout = False, time_step = 2):
        super(Atten_LSTM_Net, self).__init__()
        self.fc1 = nn.Linear(L, L, bias = True)
        self.fc2 = nn.Linear(L, L, bias = True)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)

        self.dropout = dropout
        self.T = time_step

    def forward(self, x):
        device = x.device
        q = torch.randn((1, 512,)).to(device)
        z_time = []
        for i in range(self.T):
            e = torch.mm(x, torch.transpose(q, 1, 0))
            #a = F.softmax(e, dim = 0)
            a = (e * e) / torch.sum(e * e)
            o = torch.mm(torch.transpose(a, 1, 0), x)
            if self.dropout:
                q = torch.sigmoid(self.dropout1(self.fc1(o)) + self.dropout2(self.fc2(q)))
            else:
                q = torch.sigmoid(self.fc1(o) + self.fc2(q))
            z_time.append(o)
        z = sum(z_time)
        return torch.transpose(a, 1, 0), z, x

