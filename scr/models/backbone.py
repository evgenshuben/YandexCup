from torch import nn
import timm

class BackBone(nn.Module):
    def __init__(self, backbone_name, emb_size=512, num_classes=10, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=num_classes
        )

        self.ln = nn.LayerNorm([3, 224, 224])
        self.embedding_size = self.model.get_classifier().in_features
        self.model.reset_classifier(0)
        self.fc_emb = nn.Linear(self.embedding_size, emb_size) if emb_size != self.embedding_size else nn.Identity()
        self.clf = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = self.ln(x)
        features = self.model.forward_features(x)
        pooled_output = features.mean([2, 3])
        pooled_output = pooled_output.view(pooled_output.size(0), -1)
        emb = self.fc_emb(pooled_output)
        cls = self.clf(emb)
        return dict(emb=emb, cls=cls)