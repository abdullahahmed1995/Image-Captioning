import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        # embed feature vectors
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        # define LSTM
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        self.hidden2caption = nn.Linear(hidden_size, vocab_size)
    
  
    def forward(self, features, captions):
        
        captions = captions[:, :-1]
        
        captions = self.word_embeddings(captions)
        
        features = features.unsqueeze(1)
        
        embedded_features = torch.cat((features, captions), 1)
                
        lstm_out, self.hidden = self.lstm(embedded_features)
        
        output_caption = self.hidden2caption(lstm_out)
        
        
        return output_caption
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        ids = []
        states = (torch.randn(1, 1, self.hidden_size).to(inputs.device), 
                  torch.randn(1, 1, self.hidden_size).to(inputs.device))
        for i in range(max_len):
           
            lstm_out, states = self.lstm(inputs, states)
        
            output_caption = self.hidden2caption(lstm_out)
                 
                        
            output_caption = output_caption.squeeze(1)                 
            wordID  = output_caption.argmax(dim=1)              
            ids.append(wordID.item())
            
            inputs = self.word_embeddings(wordID.unsqueeze(0))  
            
        return ids
    
    
  