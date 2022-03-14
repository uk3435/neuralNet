import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.utils.class_weight import compute_class_weight
import torchvision.datasets as datasets
from torchvision import transforms


device = torch.device('cpu')

df = pd.read_excel('veri1.xlsx')



train_text, temp_text, train_text2, temp_text2,train_text3, temp_text3,train_text4, temp_text4,train_text5, temp_text5,train_labels, temp_labels=train_test_split(df['FU_1'],df['FU_2'],df['FU_3'],df['FU_4'],df['FU_5'],df['Sonuc'],random_state=2018, test_size=0.3, stratify=df['Sonuc']) 


val_text,test_text,val_text2,test_text2,val_text3,test_text3,val_text4,test_text4,val_text5,test_text5,val_labels, test_labels=train_test_split(temp_text, temp_text2,temp_text3, temp_text4,temp_text5,temp_labels, random_state=2018, test_size=0.5, stratify=temp_labels)


train_tensor = torch.tensor(train_text.tolist())
train_tensor2=torch.tensor(train_text2.tolist())
train_tensor3=torch.tensor(train_text3.tolist())
train_tensor4=torch.tensor(train_text4.tolist())
train_tensor5=torch.tensor(train_text5.tolist())


train_labels_tensor = torch.tensor(train_labels.tolist())
val_tensor = torch.tensor(val_text.tolist())
val_tensor2 = torch.tensor(val_text2.tolist())
val_tensor3 = torch.tensor(val_text3.tolist())
val_tensor4 = torch.tensor(val_text4.tolist())
val_tensor5= torch.tensor(val_text5.tolist())
val_labels_tensor = torch.tensor(val_labels.tolist())

test_tensor = torch.tensor(test_text.tolist())
test_tensor2=torch.tensor(test_text2.tolist())
test_tensor3=torch.tensor(test_text3.tolist())
test_tensor4=torch.tensor(test_text4.tolist())
test_tensor5=torch.tensor(test_text5.tolist())



test_labels_tensor = torch.tensor(test_labels.tolist())


class NN(nn.Module):

    def __init__(self,input_size,num_classes):
        super(NN,self).__init__()
        self.fc1=nn.Linear(input_size,5)
        self.fc2=nn.Linear(5,num_classes)
        #self.dropout = nn.Dropout(0.3)

        self.softmax = nn.Softmax(dim=1)

    def forward(self,x,x2,x3,x4,x5):

        x=torch.cat((x, x2,x3,x4,x5),dim=1)        
        # print("x")
        # print(x)
        # print(x.shape)

        x=F.relu(self.fc1(x))
        #x = self.dropout(x)
        x=self.fc2(x)
        x = self.softmax(x)
        return x

    
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#torch.cat((x, x, x), 0)
input_size=5
num_classes=2
learning_rate=0.001            
batch_size=32
num_epochs=1


train_dataset=TensorDataset(train_tensor,train_tensor2,train_tensor3,train_tensor4,train_tensor5,train_labels_tensor)

train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

val_dataset=TensorDataset(val_tensor,val_tensor2,val_tensor3,val_tensor4,val_tensor5,val_labels_tensor)

val_loader=DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=True)

test_dataset=TensorDataset(test_tensor,test_tensor2,test_tensor3,test_tensor4,test_tensor5,test_labels_tensor)
test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)



model= NN(input_size=input_size,num_classes=num_classes).to(device)

#criterion=nn.CrossEntropyLoss()

optimizer=optim.Adam(model.parameters(),lr=learning_rate)

from sklearn.utils.class_weight import compute_class_weight


class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels),y=train_labels)


weights= torch.tensor(class_weights,dtype=torch.float)
weights =weights.to(device)


# loss function
criterion  = nn.CrossEntropyLoss(weights) 


def train():
    model.train()

    total_loss, total_accuracy = 0, 0


    total_preds=[]


    for batch_idx,(data,data2,data3,data4,data5,targets) in enumerate(train_loader):
        

        data=data.to(device=device)
       
        data2=data2.to(device=device)
        data3=data3.to(device=device)
        data4=data4.to(device=device)
        data5=data5.to(device=device)
        targets=targets.to(device=device)
        print("targets")
        print(targets)

        

        data=data.reshape(data.shape[0],-1)
        data2=data2.reshape(data2.shape[0],-1)
        data3=data3.reshape(data3.shape[0],-1)
        data4=data4.reshape(data4.shape[0],-1)
        data5=data5.reshape(data5.shape[0],-1)
      

        scores=model(data,data2,data3,data4,data5)
        loss=criterion(scores,targets)
        print("loss")
        print(loss)

        optimizer.zero_grad()
        loss.backward()
        total_loss = total_loss + loss.item()     
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        scores=scores.detach().cpu().numpy()
        print("train preds")
        print(scores)
        total_preds.append(scores)

    avg_loss = total_loss / len(train_loader)
    total_preds  = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds



def evaluate():
  
    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for batch_idx,(data,data2,data3,data4,data5,targets) in enumerate(val_loader):
        data=data.to(device=device)
        data2=data2.to(device=device)
        data3=data3.to(device=device)
        data4=data4.to(device=device)
        data5=data5.to(device=device)
        targets=targets.to(device=device)
        print("targets")
        print(targets)

        

        data=data.reshape(data.shape[0],-1)
        data2=data2.reshape(data2.shape[0],-1)
        data3=data3.reshape(data3.shape[0],-1)
        data4=data4.reshape(data4.shape[0],-1)
        data5=data5.reshape(data5.shape[0],-1)
        
        with torch.no_grad():

            scores=model(data,data2,data3,data4,data5)
            loss=criterion(scores,targets)
            total_loss = total_loss + loss.item()

            preds = scores.detach().cpu().numpy()
            print("val preds")
            print(preds)

            total_preds.append(preds)

    avg_loss = total_loss / len(val_loader) 

  # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)
   
    return avg_loss, total_preds


best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]

for epoch in range(num_epochs):
     
    print('\n Epoch {:} / {:}'.format(epoch + 1, num_epochs))
    
    #train model
    train_loss,_= train()
    
    #evaluate model
    valid_loss,_ = evaluate()
    
    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights_nn01.pt')
    
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')


#load weights of best model
path = 'saved_weights_nn01.pt'
model.load_state_dict(torch.load(path))

# # get predictions for test data
with torch.no_grad():

    for batch_idx,(data,data2,data3,data4,data5,targets) in enumerate(test_loader):
        data=data.to(device=device)
        data2=data2.to(device=device)
        data3=data3.to(device=device)
        data4=data4.to(device=device)
        data5=data5.to(device=device)
        targets=targets.to(device=device)
        print("targets")
        print(targets)

        

        data=data.reshape(data.shape[0],-1)
        data2=data2.reshape(data2.shape[0],-1)
        data3=data3.reshape(data3.shape[0],-1)
        data4=data4.reshape(data4.shape[0],-1)
        data5=data5.reshape(data5.shape[0],-1)

  
    preds = model(data,data2,data3,data4,data5)
    preds = preds.detach().cpu().numpy()



# model's performance
preds = np.argmax(preds, axis = 1)
print(classification_report(targets, preds))


# confusion matrix
print(pd.crosstab(targets, preds))
