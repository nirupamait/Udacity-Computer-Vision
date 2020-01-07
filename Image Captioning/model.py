import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        
        # Embedding layer for creating words vectors of a specified size from incoming words.
        #It is between the input and the LSTM layer.
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        #Define lstm that takes embedded word vectors as inputs and output hidden states
        #https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # The linear layer that maps the hidden state output dimension
        # to the number of words we want as output, vocab_size
        self.linear = nn.Linear(hidden_size, vocab_size)  
        
        #initialize weights
        #self.init_weights()
        
            
    def init_hidden(self,batch_size):
       #https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
       #At the start of training, we need to initialize a hidden state;
       #there will be none because the hidden state is formed based on previously seen data.
       # So, this function defines a hidden state with all zeroes
       # The axes semantics are (num_layers, batch_size, hidden_dim)
        return (torch.zeros(1,batch_size,self.hidden_size,device=device),
            torch.zeros(1,batch_size,self.hidden_size,device=device))
    
    def forward(self, features, captions):
        # Discard the <end> word to avoid predicting when <end> is the input of the RNN
        captions = captions[:, :-1]     
        
        # Initialize the hidden state
        batch_size = features.size()[0] # features is of shape (batch_size, embed_size)
        hidden = self.init_hidden(batch_size) 
                
        # Create embedded word vectors for each word in the captions
        embeds = self.word_embeddings(captions) # embeddings new shape : (batch_size, captions length - 1, embed_size)
        
        # Stack the features and captions
        inputs = torch.cat((features.unsqueeze(1), embeds), dim=1) # embeddings new shape : (batch_size, caption length, embed_size)
        
        # Get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hidden state
        lstm_out, hidden = self.lstm(inputs,hidden) # lstm_out shape : (batch_size, caption length, hidden_size)

        # Fully connected layer
        outputs = self.linear(lstm_out) # outputs shape : (batch_size, caption length, vocab_size)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        caption =[]
        batch_size = inputs.size()[0]
        states = self.init_hidden(batch_size)
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs,states)
            outputs = self.linear(lstm_out.squeeze(1))
            _ , predicted = outputs.max(dim=1)
            caption.append(predicted.item())

            inputs = self.word_embeddings(predicted)
            inputs = inputs.unsqueeze(1)
        return caption