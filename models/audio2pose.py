import torch.nn as nn
import torch
from models.util import MyResNet34

class audio2poseLSTM(nn.Module):
    def __init__(self):
        super(audio2poseLSTM,self).__init__()

        self.em_pose = MyResNet34(256, 1)
        self.em_audio = MyResNet34(256, 1)
        self.lstm = nn.LSTM(512,256,num_layers=2,bias=True,batch_first=True)

        self.output = nn.Linear(256,6)


    def forward(self,x):
        pose_em = self.em_pose(x["img"])
        bs = pose_em.shape[0]
        zero_state = torch.zeros((2, bs, 256), requires_grad=True).to(pose_em.device)
        cur_state = (zero_state, zero_state)
        img_em = pose_em
        bs,seqlen,num,dims = x["audio"].shape

        audio = x["audio"].reshape(-1, 1, num, dims)
        audio_em = self.em_audio(audio).reshape(bs, seqlen, 256)

        result = [self.output(img_em).unsqueeze(1)]

        for i in range(seqlen):

            img_em,cur_state = self.lstm(torch.cat((audio_em[:,i:i+1],img_em.unsqueeze(1)),dim=2),cur_state)
            img_em = img_em.reshape(-1, 256)

            result.append(self.output(img_em).unsqueeze(1))
        res = torch.cat(result,dim=1)
        return res
