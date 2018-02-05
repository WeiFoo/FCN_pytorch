import torch.nn.functional as F
import torch.nn as nn

class fcn8vgg(nn.Module):
    def __init__(self, pretrained_vgg, num_classes):
        super(fcn8vgg, self).__init__()
        vgg = pretrained_vgg
        features = list(vgg.features.children())
        classifier = list(vgg.classifier.children())
        self.pool3 = nn.Sequential(*features[:17])
        self.pool4 = nn.Sequential(*features[17:24])
        self.pool5 = nn.Sequential(*features[24:])

        self.pool3_score = nn.Conv2d(256, num_classes, 1)
        self.pool3_score.weight.data.zero_()
        self.pool3_score.bias.data.zero_()

        self.pool4_score = nn.Conv2d(512,num_classes, 1)
        self.pool4_score.weight.data.zero_()
        self.pool4_score.bias.data.zero_()

        fc1 = nn.Conv2d(512, 4096, kernel_size=3, stride=1, padding=1)
        fc2 = nn.Conv2d(4096,4096, kernel_size=1)

        final_predict = nn.Conv2d(4096, num_classes, kernel_size=1)
        final_predict.weight.data.zero_()
        final_predict.bias.data.zero_()

        self.final_predict = nn.Sequential(
            fc1,
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            fc2,
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            final_predict
        )

    def forward(self, x):
        pool3 = self.pool3(x)         # 1/8
        pool4 = self.pool4(pool3)     # 1/16
        pool5 = self.pool5(pool4)
        predicts_score = self.final_predict(pool5) # 1/32

        pool4_score = self.pool4_score(pool4)
        conv7_2x = F.upsample(predicts_score,size=pool4_score.size()[2:4])
        pool4_combine = pool4_score + conv7_2x

        pool3_score = self.pool3_score(pool3)
        pool4_2x = F.upsample(pool4_combine,pool3_score.size()[2:4])
        pool3_combine = pool3_score + pool4_2x

        score = F.upsample(pool3_combine, scale_factor=8)
        return score




