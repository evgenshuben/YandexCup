import torch
from torch import nn
from nemo.collections.asr.models import ASRModel



class AudioConformer(nn.Module):
    def __init__(
            self,
            pretrained_model_name='stt_en_conformer_ctc_small',
            emb_size=512,
            num_classes=10
    ):
        super().__init__()
        self.pretrained_model = ASRModel.from_pretrained(
            model_name=pretrained_model_name,
            map_location=torch.device("cuda")
        )
        self.encoder = self.pretrained_model.encoder
        self.freq_transform = nn.Linear(84, 80)

        self.embedding_size = self.encoder.pre_encode.out.out_features
        self.fc_emb = nn.Linear(self.embedding_size, emb_size) if emb_size != self.embedding_size else nn.Identity()
        self.clf = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        batch_size, freq_size, time_size = x.size()
        x = x.permute(0, 2, 1).reshape(-1, freq_size)
        x = self.freq_transform(x)
        x = x.view(batch_size, time_size, -1).permute(0, 2, 1)

        encoder_output, lengths = self.encoder(
            audio_signal=x,
            length=torch.tensor([x.shape[2]] * x.shape[0], device=x.device)

        )
        features = encoder_output.mean(dim=2)
        emb = self.fc_emb(features)
        cls = self.clf(emb)
        return dict(emb=emb, cls=cls)