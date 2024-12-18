ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/layer_pfl'
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
sys.path.append(f'{ROOT_DIR}/code')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#MODELS
class EMNIST(nn.Module):
    def __init__(self, CLASSES):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                    nn.Flatten())


        self.fc1 = nn.Sequential(nn.Linear(256, 100), 
                                nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(100, CLASSES))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class CIFAR(nn.Module):
    def __init__(self, CLASSES):
        super().__init__()
    
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.flatten = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                    nn.Flatten())
        
        self.fc1 = nn.Sequential(nn.Linear(512, 100), 
                        nn.ReLU())
        
        self.fc2= nn.Linear(100, CLASSES)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        out = self.fc2(x)
        return out


class FMNIST(nn.Module):
    def __init__(self, CLASSES):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))

        self.flatten = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                    nn.Flatten())

        
        self.fc1 = nn.Sequential(nn.Linear(256, 100), 
                                nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(100, CLASSES))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class ISIC(nn.Module):
    def __init__(self, CLASSES):
        super().__init__()
    
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
        )

        self.flatten = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                    nn.Flatten())

        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.fc2= nn.Sequential(nn.Linear(256, CLASSES))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        out = self.fc2(x)
        return out


class Attention(nn.Module):
    """ Self-attention """
    def __init__(self, n_embd):
        super().__init__()
        self.key = nn.Linear(n_embd,n_embd, bias=False)
        self.query = nn.Linear(n_embd,n_embd, bias=False)
        self.value = nn.Linear(n_embd,n_embd, bias=False)

    def forward(self, x, mask):
        B,T,C = x.shape
        k = self.key(x) 
        q = self.query(x)
        weights = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        mask = mask.float().masked_fill(mask == 0, -np.inf).unsqueeze(1).repeat(1,T,1) # (B, T, T)
        masked_weights = weights + mask
        masked_weights = F.softmax(masked_weights, dim=-1) # (B, T, T)
        v = self.value(x) 
        out = masked_weights @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    
class Sentiment(nn.Module):
    def __init__(self, CLASSES):
        super().__init__()
        n_embd = 768
        block_size = 119
        #load embedding table and instantiate mapping dict
        embeddings_index_dict = torch.load(f'{ROOT_DIR}/data/Sentiment/token_to_index_and_embeddings.pth')
        embeddings = embeddings_index_dict['embeddings']
        #model
        self.token_embedding_table1 = nn.Embedding.from_pretrained(embeddings.to(device), freeze=False)
        self.position_embedding_table1 = nn.Embedding(block_size, n_embd)
        self.attention1 = Attention(n_embd)
        self.proj1 = nn.Sequential(nn.Linear(n_embd, n_embd))
        self.fc1 =  nn.Sequential(
                        nn.Linear(n_embd, 4 * n_embd),
                        nn.ReLU())
        self.resid1 = nn.Sequential(nn.Linear(4 * n_embd, n_embd)) # resid projection)

        self.fc2= nn.Sequential(nn.Linear(n_embd, CLASSES))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            with torch.no_grad():
                module.weight.data = F.normalize(module.weight.data, p=2, dim=1)

    def forward(self, features):
        token_indices, mask = features
        B, T = token_indices.shape
        model_device = next(self.parameters()).device
        #Token
        tok_emb = self.token_embedding_table1(token_indices)
        #Pos
        pos_emb = torch.arange(T, device=model_device)
        pos_emb = self.position_embedding_table1(pos_emb) # (T,C)
        
        #Att 1
        x_orig = tok_emb + pos_emb # (B,T,C)
        x_att = self.attention1(x_orig, mask)
        x = x_orig + self.proj1(x_att)
        x_ff = self.fc1(x)
        x = x + self.resid1(x_ff)

        #predict
        x = x[:,0] # CLS token
        logits = self.fc2(x) # (B,1,C)
        return logits


class mimic(nn.Module):
    def __init__(self, CLASSES):
        super().__init__()
        n_embd = 768
        block_size = 512
        #load embedding table and instantiate mapping dict
        embeddings_index_dict = torch.load(f'{ROOT_DIR}/data/mimic_iii/token_to_index_and_embeddings.pth')
        embeddings = embeddings_index_dict['embeddings']
        #model
        self.token_embedding_table1 = nn.Embedding.from_pretrained(embeddings.to(device), freeze=False)
        self.position_embedding_table1 = nn.Embedding(block_size, n_embd)
        self.attention1 = Attention(n_embd)
        self.proj1 = nn.Sequential(nn.Linear(n_embd, n_embd))
        self.fc1 =  nn.Sequential(
                        nn.Linear(n_embd, 4 * n_embd),
                        nn.ReLU())
        self.resid1 = nn.Sequential(nn.Linear(4 * n_embd, n_embd)) # resid projection)

        self.fc2= nn.Sequential(nn.Linear(n_embd, CLASSES))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            with torch.no_grad():
                module.weight.data = F.normalize(module.weight.data, p=2, dim=1)

    def forward(self, features):
        token_indices, mask = features
        B, T = token_indices.shape
        model_device = next(self.parameters()).device
        #Token
        tok_emb = self.token_embedding_table1(token_indices)
        
        #Pos
        pos_emb = torch.arange(T, device=model_device)
        pos_emb = self.position_embedding_table1(pos_emb) # (T,C)
        
        #first
        x_orig = tok_emb + pos_emb # (B,T,C)
        x_att = self.attention1(x_orig, mask)
        x = x_orig + self.proj1(x_att)
        x_ff = self.fc1(x)
        x = x + self.resid1(x_ff)
        
        #predict
        x = x[:,0] # CLS token
        logits = self.fc2(x) # (B,1,C)
        return logits

class Heart(torch.nn.Module):
    def __init__(self, CLASSES):
        super().__init__()
        self.fc1 = nn.Sequential(
                nn.Linear(10, 50),
                nn.ReLU(),)
        self.fc2= nn.Sequential(
                nn.Linear(50, 20),
                nn.ReLU())
        self.fc3= nn.Sequential(
                nn.Linear(20, 20),
                nn.ReLU())
        self.fc4 = nn.Sequential(
                nn.Linear(20, CLASSES))

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out

#HYPERNETWORK
class HyperNetwork(nn.Module):
    """Hypernetwork for generating layer-wise aggregation weights across all clients."""
    def __init__(self, embedding_dim: int, hidden_dim: int, model: nn.Module, num_clients: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_clients = num_clients
        
        # Create layer-specific MLPs for each model layer
        self.layer_networks = nn.ModuleDict()
        for name, _ in model.named_parameters():
            layer_name = name.split('.')[0]
            if layer_name not in self.layer_networks:
                self.layer_networks[layer_name] = nn.Sequential(
                    nn.Linear(embedding_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, num_clients),  # Output weights for all clients
                    nn.Softmax(dim=-1)  # Softmax over client dimension
                )
    
    def forward(self, embedding):
        """Generate layer-wise aggregation weights across all clients."""
        weights = {}
        for layer_name, network in self.layer_networks.items():
            # Returns tensor of shape [num_clients]
            weights[layer_name] = network(embedding)
        return weights