






import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer, T5EncoderModel, T5ForConditionalGeneration



class RoBERTaWrapper(nn.Module):
    def __init__(self, model_name="models/roberta-base"):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)

    def forward(self, text_list):  # text_list: List[str]
        encoded = self.tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        input_ids = encoded["input_ids"].to(self.model.device)
        attention_mask = encoded["attention_mask"].to(self.model.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, hidden_states, mask=None):
        attn_weights = self.attn(hidden_states).squeeze(-1)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        attn_scores = torch.softmax(attn_weights, dim=-1)
        return torch.sum(hidden_states * attn_scores.unsqueeze(-1), dim=1)


class SingleRouteEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.encoder(x)


class JointClassifier(nn.Module):
    def __init__(self, input_dim_each=784, hidden_dim=256, fusion_dim=512):
        super().__init__()
        self.route_orig = SingleRouteEncoder(input_dim_each, hidden_dim)
        self.route_dist = SingleRouteEncoder(input_dim_each, hidden_dim)
        self.route_reorder = SingleRouteEncoder(input_dim_each, hidden_dim)


        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, 1)
        )

    def forward(self, orig_vec, orig_metrics, dist_vec, dist_metrics, reorder_vec, reorder_metrics):

        orig_input = torch.cat([orig_vec, orig_metrics], dim=-1)    # [batch_size, 784]
        dist_input = torch.cat([dist_vec, dist_metrics], dim=-1)
        reorder_input = torch.cat([reorder_vec, reorder_metrics], dim=-1)


        orig_feat = self.route_orig(orig_input)     # [batch_size, hidden_dim]
        dist_feat = self.route_dist(dist_input)
        reorder_feat = self.route_reorder(reorder_input)


        fused = torch.cat([orig_feat, dist_feat, reorder_feat], dim=-1)  # [batch_size, hidden_dim*3]
        logits = self.fusion(fused)  # [batch_size, 1]
        return logits.squeeze(-1)





class SingleRouteEncoder5(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super().__init__()
        layers = []

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.ReLU()

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))

        self.mlp = nn.Sequential(*layers)
        self.use_residual = True

    def forward(self, x):
        out = self.activation(self.input_bn(self.input_proj(x)))
        residual = out

        out = self.mlp(out)

        if self.use_residual:
            out = out + residual

        return out

class JointClassifier5(nn.Module):
    def __init__(self, input_dim_each=784, hidden_dim=256, fusion_dim=512, num_layers=3, fusion_layers=3):
        super().__init__()
        self.route_orig = SingleRouteEncoder5(input_dim_each, hidden_dim, num_layers=num_layers)
        self.route_dist = SingleRouteEncoder5(input_dim_each, hidden_dim, num_layers=num_layers)
        self.route_reorder = SingleRouteEncoder5(input_dim_each, hidden_dim, num_layers=num_layers)


        fusion_layers_list = []

        fusion_input_dim = hidden_dim * 3
        fusion_hidden = fusion_dim

        self.fusion_first = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        for _ in range(fusion_layers - 1):
            fusion_layers_list.append(nn.Linear(fusion_hidden, fusion_hidden))
            fusion_layers_list.append(nn.BatchNorm1d(fusion_hidden))
            fusion_layers_list.append(nn.ReLU())
            fusion_layers_list.append(nn.Dropout(0.3))

        self.fusion_mlp = nn.Sequential(*fusion_layers_list)

        self.fusion_out = nn.Linear(fusion_hidden, 1)
        self.use_fusion_residual = True

    def forward(self, orig_vec, orig_metrics, dist_vec, dist_metrics, reorder_vec, reorder_metrics):

        orig_input = torch.cat([orig_vec, orig_metrics], dim=-1)
        dist_input = torch.cat([dist_vec, dist_metrics], dim=-1)
        reorder_input = torch.cat([reorder_vec, reorder_metrics], dim=-1)


        orig_feat = self.route_orig(orig_input)
        dist_feat = self.route_dist(dist_input)
        reorder_feat = self.route_reorder(reorder_input)


        fused = torch.cat([orig_feat, dist_feat, reorder_feat], dim=-1)

        out = self.fusion_first(fused)
        residual = out

        out = self.fusion_mlp(out)

        if self.use_fusion_residual:
            out = out + residual

        logits = self.fusion_out(out)
        return logits.squeeze(-1)
