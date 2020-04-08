!pip install pytorch-nlp
import torch
from torchnlp import metrics
from torch import nn
from torchtext import datasets
from torchtext import data
from torchtext.vocab import Vectors
from google.colab import drive
from torch import optim
import os
import numpy as np
import matplotlib.pyplot as plt
drive.mount('/content/drive', force_remount=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = '/content/drive/My Drive/MT Project/' #Path to general folder

#General params
batch_size = 32 #Batch size
epochs = 15 #How many epochs we train
learning_rate = 0.01
layers = 6 #Num of layers in the encoder and decoder
hidden_size = 1000 #Hiddensize dimension
dropout = 0.1 #0 equals no dropout
val_every = 3 #Performs validation after this many epochs, 0 is only at last epoch

only_val = False #Indicates if we only evaluate an already trained model

save_model = True
load_model = True
load_path = 0 #0 is auto select latest, else specify

save_path = path+'save/'
val_file = path+'val_files/'




#Preload the GloVe embeddings
vectors_de = Vectors(name='vectors_de.txt', cache=path)
vectors_en = Vectors(name='vectors_en.txt', cache=path)

#Build the vocabs and iterators
src = data.Field(init_token='<bos>', eos_token='<eos>', lower=True, sequential=True)
trg = data.Field(init_token='<bos>', eos_token='<eos>', lower=True, sequential=True)
mt_train = datasets.TranslationDataset(
    path=path + 'dev', exts=('.de', '.en'),
    fields=(src, trg))

mt_dev = datasets.TranslationDataset(
    path=path + 'dev', exts=('.de', '.en'),
    fields=(src, trg))

src.build_vocab(mt_train, min_freq=2, vectors=vectors_de) #198491
trg.build_vocab(mt_train, min_freq=2, vectors=vectors_en) #123156

train_iter = data.BucketIterator(
    dataset=mt_train, batch_size=batch_size,
    sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)))

val_iter = data.BucketIterator(dataset=mt_dev, batch_size=1, train=False)

#Build the two embedding layers
embed_src = torch.nn.Embedding(len(src.vocab), 300)
embed_trg = torch.nn.Embedding(len(trg.vocab), 300)

embed_src.weight.data.copy_(src.vocab.vectors)
embed_trg.weight.data.copy_(trg.vocab.vectors)

#We fix these weights, they are not updated during training
embed_src.weight.requires_grad = False
embed_trg.weight.requires_grad = False

#Auto selects 'functions'
if load_path == 0 and load_model == True:
  models = os.listdir(save_path)
  models.sort()
  load_path = save_path + models[-1]

if val_every == 0:
  val_every = epochs+5





class Encoder(nn.Module):
  def __init__(self, h_size, emb_size, dropout):
    super(Encoder, self).__init__()
    self.hidden = h_size
    self.lstm = nn.LSTM(emb_size, h_size, batch_first=True, bidirectional=False,
                       num_layers=layers, dropout=dropout)
     
  def forward(self, inp, hidden, cell_state):
    #Default encoder structure
    output_enc, (hidden_enc, cell_state_enc) = self.lstm(inp, (hidden, cell_state))
    return output_enc, (hidden_enc, cell_state_enc)
  
  def initHidden(self, size=batch_size):
    #Resets the hidden states and cell states between different sentences
    return (torch.zeros(layers, size, self.hidden, device=device), #Initial hidden states
              torch.zeros(layers, size, self.hidden, device=device)) #(all zeros)
    

class Decoder(nn.Module):
  def __init__(self, h_size, emb_size, output_size, layers, dropout):
    super(Decoder, self).__init__()
    self.hidden_size = h_size
    self.output_dim = output_size
    self.layers = layers
    self.lstm = nn.LSTM(emb_size, self.hidden_size, batch_first=True, 
                        num_layers=layers, dropout=dropout)
    self.linear_1 = nn.Linear(h_size, 10*h_size)
    self.linear_out = nn.Linear(10*h_size, output_size)
    self.softmax = torch.nn.Softmax(dim=1)
  
  def forward(self, enc_states, inp, hidden, cell_state, att_mask, cov):
    #Our decoder with attention and coverage
    #Two linear layers at the end
    output_dec, (hidden_dec, cell_state_dec) = self.lstm(inp, (hidden, cell_state))
    
    attn = hidden_dec[-1,:]
    attn = attn.unsqueeze(1)
    attn = attn.expand_as(enc_states)
    attn = torch.sum(attn*enc_states, dim=-1).squeeze(-1)
    attn_dist = self.softmax(attn - att_mask*10**16)
    cov_loss = torch.sum(torch.min(cov.to('cpu'), attn_dist.to('cpu')))
    cov = cov + attn_dist.to('cpu')
    attn_dist = attn_dist.unsqueeze(-1)
    attn_dist = attn_dist.expand_as(enc_states)
    context = attn_dist*enc_states
    context = torch.sum(context,dim=1)
    context = torch.cat((context, hidden_dec[-1,:]), dim=-1)
    scores = self.linear_1(hidden_dec[-1,:,:])
    scores = self.linear_out(scores).unsqueeze(1)
    return scores, cov, cov_loss, (hidden_dec, cell_state_dec)


def train_loop(lr, epochs, train_iter, val_iter, criterion, enc, dec, save, load, val_file, only_val=False, val_every=val_every):   
  #Only optimize non embedding params
  optim_enc = optim.Adam([param for param in enc.parameters() if param.requires_grad == True], lr = lr, weight_decay=0.0001)
  optim_dec = optim.Adam([param for param in dec.parameters() if param.requires_grad == True], lr = lr, weight_decay=0.0001)
  if load == True:
    #Loader function for presaved weights
    try:
      checkpoint = torch.load(load_path)
      load_epoch = checkpoint['epoch']
      train_loss = checkpoint['train_loss']
      enc.load_state_dict(checkpoint['enc_state_dict'])
      optim_enc.load_state_dict(checkpoint['enc_optimizer_state_dict'])
      dec.load_state_dict(checkpoint['dec_state_dict'])
      optim_dec.load_state_dict(checkpoint['dec_optimizer_state_dict'])
      print('Model loaded succesfully')
      plt.plot(train_loss)
      plt.title('Training loss this far')
      plt.show()
    except:
      #No model loaded error
      raise Exception(('No model found at: '+load_path))    
  else:
      #If we start from scratch
      train_loss = []
      load_epoch = 0
      
  if only_val == False:
    #If we intent to train and not just validate
    for epoch in range(epochs-load_epoch):
      epoch = epoch + load_epoch
      train_iter.init_epoch()
      train_loss_ = 0
      i = 0
      if epoch == 4:
        #Large step size first 4 epochs then we slow it down
        optim_enc = optim.Adam([param for param in enc.parameters() if param.requires_grad == True], lr = 0.1*lr, weight_decay=0.0001)
        optim_dec = optim.Adam([param for param in dec.parameters() if param.requires_grad == True], lr = 0.1*lr, weight_decay=0.0001)
      for data in train_iter:
        optim_enc.zero_grad()
        optim_dec.zero_grad()
        x = data.src.transpose(1,0)
        batch_size = data.batch_size
        y = data.trg.transpose(1,0)
        #Coverage place holder
        cov = torch.zeros(x.size())
        cov=torch.zeros(1).to(device); att_mask = torch.zeros(1).to(device)
        #Invert input sentence
        x = torch.flip(x, [1])
        #Attention mask
        att_mask = torch.where(x == src.vocab.stoi['<pad>'], torch.ones(1), torch.zeros(1))
        att_mask = att_mask.to(device)
        x_emb = embed_src(x).to(device)
        #Placeholder for scores
        scores = torch.zeros((batch_size, y.size()[1], len(trg.vocab))).to(device)
        #Beam place holders
        beam_score = torch.zeros((batch_size, beam_size, y.size()[1]))
        bi_candidates = torch.zeros((batch_size, beam_size**2))
        bs_candidates = torch.zeros((batch_size, beam_size**2))
        beam_hidden = torch.zeros((beam_size, batch_size, num_layers, hidden_size))
        beam_cell = torch.zeros((beam_size, batch_size, num_layers, hidden_size))
        scores_list = torch.zeros((beam_size, batch_size, y.size()[1]))
        best_k = torch.zeros(y.size()[1])
        for j in range(y.size()[1]):
            if j == 0:
              (hidden, cell_state) = enc.initHidden(size=batch_size)
              enc_states, (hidden, cell_state) = enc(x_emb, hidden, cell_state)
              y_hat = torch.zeros((batch_size, y.size()[1], beam_size)).long()
              y_hat[:,j,:] = trg.vocab.stoi['<bos>']
              y_hat_emb = embed_trg(y_hat[:,j,0].to('cpu')).to(device).unsqueeze(1)
              score, (hidden, cell_state) = dec(enc_states, y_hat_emb, hidden, cell_state, att_mask)
              scores_list[:,:,0] = score[:, 0, y[j]] #Score is: batch x 1 x vocab_size
              #Pick the beam_size best candidates
              beam_score[:,:,j], _ = torch.topk(score, beam_size)   
              #Save the hidden states corresponding to them (same here when j = 0)
              beam_hidden = hidden.unsqueeze(0).expand((beam_size, batch_size, num_layers, hidden_size))
              beam_cell = cell_state.unsqueeze(0).expand((beam_size, batch_size, num_layers, hidden_size))
              idx = torch.arange(0,beam_size, 1)
            else:
              for k in range(beam_size):
                #Embed y
                y_hat_emb = embed_trg(y_hat[:,j,k].to('cpu')).to(device).unsqueeze(1)
                #Run decoder step
                score, (hidden, cell_state) = dec(enc_states, y_hat_emb, beam_hidden[idx[k]], beam_cell[idx[k]], att_mask)
                #Get beam size best candidates from this beam (beam k)
                scores_list[k,:,0] = score[:, 0, y[j]] #Score is: batch x 1 x vocab_size
                bs_candidates[:, k*beam_size:(1+k)*beam_size] , bi_candidates[:, k*beam_size:(1+k)*beam_size] = torch.topk(score, beam_size)
                #Save their hidden and cell states
                beam_hidden[k,:,:,:] = hidden
                beam_cell[k,:,:,:] = cell_state
              #Find the beam size best candidates from the k beams
              beam_score[:,:,j], temp_idx = torch.topk(bs_candidates, beam_size) 
              for i1 in range(batch_size): #i1 is batch looper
                for i2 in range(beam_size): #i2 is beam looper
                  y_hat[i1, j+1, i2] = torch.argmax(beam_score[i1, i2])
              idx = temp_idx // beam_size
              best_k[j] = idx[0] #Highest scored candidate in the beam
        y = y.to(device)
        for i1 in batch_size:
          pred = []
          corr = []
          for j in range(y.size()[1]):
            pred.append(trg.vocab.itos[y_hat[i1, j, best_k[j]]])
            corr.append(trg.vocab.itos[y[i1, j]])
            loss = metrics.get_moses_multi_bleu([' '.join(pred[:-1])], [' '.join(correct[1:])])
            loss = loss * 1-scores_list[best_k[j], i1, j] + beam_score[i1, best_k[j], j]
        #We add loss and cov_loss, we multiply by 15 so they have the ratio we want (about 1/3 of loss in first few epochs is from coverage)
        loss = loss + cov_loss*15
        loss.backward()
        optim_enc.step()
        optim_dec.step()
        train_loss_ = train_loss_ + loss.item() / (len(train_iter)*batch_size)
        
        i += 1
        if i % 2000 == 0:
          #This is just so we know the model isent frozen :p
          print(str(i) + ' / ' + str(len(train_iter)))

      train_loss.append(train_loss_)
      
      #Plot the training losses
      plt.close()
      plt.plot(train_loss)
      plt.title('Train loss: %.4f' % train_loss_)
      plt.show()
      
      #Save model after every epoch
      if save == True:
        torch.save({
            'epoch': epoch + 1,
            'enc_state_dict': enc.state_dict(),
            'enc_optimizer_state_dict': optim_enc.state_dict(),
            'dec_state_dict': dec.state_dict(),
            'dec_optimizer_state_dict': optim_dec.state_dict(),
            'train_loss': train_loss}, 
            save_path+str(epoch+1).zfill(3) + '.pt')
        print('Model saved')
      
      #Validate at speficied epochs
      if ((epoch+1) % val_every == 0):
        enc.eval()
        dec.eval()
        val_loop(val_iter, enc, dec, val_file, epoch+1, full_val=True)
        enc.train()
        dec.train()
      #Always validate at last epoch
      elif (epoch+1 == epochs):
        enc.eval()
        dec.eval()
        val_loop(val_iter, enc, dec, val_file, epoch+1, full_val=True)
    print(train_loss)
  
  #If we only wish to validate
  if only_val == True:
    enc.eval()
    dec.eval()
    epoch = load_epoch
    val_loop(val_iter, enc, dec, val_file, epoch+1, full_val=True)

    

    
    
def val_loop(val_iter, enc, dec, val_file_path, epoch, full_val=False):
  val_iter.init_epoch()
  i = 0
  #Place holder for validations stats
  bleu_score = 0
  bleu_error = 0
  for data in val_iter:
    i += 1
    if full_val == False:
      #This can be used to see that the model actually learns without running full val
      if i % 50 == 0:
        break
    j = 0
    #As before, flip x, make att mask etc
    x = data.src.transpose(1,0)
    batch_size = data.batch_size
    y = data.trg.transpose(1,0)
    #Coverage place holder
    cov = torch.zeros(x.size())
    cov=torch.zeros(1).to(device); att_mask = torch.zeros(1).to(device)
    #Invert input sentence
    x = torch.flip(x, [1])
    #Attention mask
    att_mask = torch.where(x == src.vocab.stoi['<pad>'], torch.ones(1), torch.zeros(1))
    att_mask = att_mask.to(device)
    x_emb = embed_src(x).to(device)
    #Placeholder for scores
    scores = torch.zeros((batch_size, y.size()[1], len(trg.vocab))).to(device)
    #Beam place holders
    beam_score = torch.zeros((batch_size, beam_size, y.size()[1]))
    bi_candidates = torch.zeros((batch_size, beam_size**2))
    bs_candidates = torch.zeros((batch_size, beam_size**2))
    beam_hidden = torch.zeros((beam_size, batch_size, num_layers, hidden_size))
    beam_cell = torch.zeros((beam_size, batch_size, num_layers, hidden_size))
    scores_list = torch.zeros((beam_size, batch_size, y.size()[1]))
    best_k = torch.zeros(y.size()[1])
    
    #Write the validations to a file
    val_file = open(val_file_path + str(epoch).zfill(3)+'.txt', 'a')
    correct_file = open(val_file_path + str(epoch).zfill(3)+'_correct.txt','a')
    cont = True
    pred = []; correct = []
    
    while cont:
        
        if j == 0:
          (hidden, cell_state) = enc.initHidden(size=batch_size)
          enc_states, (hidden, cell_state) = enc(x_emb, hidden, cell_state)
          y_hat = torch.zeros((batch_size, y.size()[1], beam_size)).long()
          y_hat[:,j,:] = trg.vocab.stoi['<bos>']
          y_hat_emb = embed_trg(y_hat[:,j,0].to('cpu')).to(device).unsqueeze(1)
          score, (hidden, cell_state) = dec(enc_states, y_hat_emb, hidden, cell_state, att_mask)
          scores_list[:,:,0] = score[:, 0, y[j]] #Score is: batch x 1 x vocab_size
          #Pick the beam_size best candidates
          beam_score[:,:,j], _ = torch.topk(score, beam_size)   
          #Save the hidden states corresponding to them (same here when j = 0)
          beam_hidden = hidden.unsqueeze(0).expand((beam_size, batch_size, num_layers, hidden_size))
          beam_cell = cell_state.unsqueeze(0).expand((beam_size, batch_size, num_layers, hidden_size))
          idx = torch.arange(0,beam_size, 1)
        else:
          for k in range(beam_size):
            #Embed y
            y_hat_emb = embed_trg(y_hat[:,j,k].to('cpu')).to(device).unsqueeze(1)
            #Run decoder step
            score, (hidden, cell_state) = dec(enc_states, y_hat_emb, beam_hidden[idx[k]], beam_cell[idx[k]], att_mask)
            #Get beam size best candidates from this beam (beam k)
            scores_list[k,:,0] = score[:, 0, y[j]] #Score is: batch x 1 x vocab_size
            bs_candidates[:, k*beam_size:(1+k)*beam_size] , bi_candidates[:, k*beam_size:(1+k)*beam_size] = torch.topk(score, beam_size)
            #Save their hidden and cell states
            beam_hidden[k,:,:,:] = hidden
            beam_cell[k,:,:,:] = cell_state
          #Find the beam size best candidates from the k beams
          beam_score[:,:,j], temp_idx = torch.topk(bs_candidates, beam_size) 
          for i1 in range(batch_size): #i1 is batch looper
            for i2 in range(beam_size): #i2 is beam looper
              y_hat[i1, j+1, i2] = torch.argmax(beam_score[i1, i2])
          idx = temp_idx // beam_size
          best_k[j] = idx[0] #Highest scored candidate in the beam
    y = y.to(device)
    for i1 in batch_size:
      pred = []
      corr = []
      for j in range(y.size()[1]):
        pred.append(trg.vocab.itos[y_hat[i1, j, best_k[j]]])
        corr.append(trg.vocab.itos[y[i1, j]])

      if (idx == trg.vocab.stoi['<eos>']) or (i > 100):
        cont = False
        try:
          bleu_score += metrics.get_moses_multi_bleu([' '.join(pred)], [' '.join(correct[1:])]) / len(val_iter)
        except:
          bleu_error += 1
        pred.append('\n')
        correct.append('\n')
        val_file.write(' '.join(pred))
        correct_file.write(' '.join(correct))
      j += 1
    if full_val == False:
      print(correct)
      print(pred)
  
  print('Average BLEU score for this validation pass: %.4f' % (bleu_score))
  print('%.0f sentences where not scored duo to error' %(bleu_error))
  
  val_file.close()
  correct_file.close()  

#Runs the model
criterion = torch.nn.CrossEntropyLoss(ignore_index=trg.vocab.stoi['<pad>']) 
enc = Encoder(hidden_size, 300, dropout).to(device)
dec = Decoder(hidden_size, 300, len(trg.vocab), layers, dropout).to(device)
train_loop(learning_rate, epochs, train_iter, val_iter, criterion, enc, dec, save_model, load_model, val_file, only_val)
