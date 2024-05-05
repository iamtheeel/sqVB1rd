import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from timeit import default_timer as timer
from OpCounter import timeStrFromS

class Trainer:
    def __init__(self,model, training_data, testing_data, optimizer, criterion, epochs, learning_rate,batch_size):
        self.model = model
        self.training_loader = DataLoader(training_data,batch_size=batch_size,shuffle=True)
        self.testing_loader = DataLoader(testing_data,batch_size=batch_size,shuffle=False)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.criterion = criterion
        self.model.to(self.device)
        self.set_training_config()

    def set_training_config(self):
        if self.optimizer == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError("Only SGD is supported for now")

        if self.criterion == "MSE":
            self.criterion = nn.MSELoss()
        elif self.criterion == "MAE":
            self.criterion = nn.L1Loss()
        elif self.criterion == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Only MSE is supported for now")

    def get_training_config(self):
        return {
            "model": self.model,
            "training_loader": self.training_loader,
            "testing_loader": self.testing_loader,
            "optimizer": self.optimizer,
            "criterion": self.criterion,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate
        }

    
    def train(self):
        self.model.train()
        lossArr = []
        modelName = self.model.__class__.__name__
        
        for epoch in range(self.epochs):
            train_loss, train_acc, total = 0, 0, 0
            correct,total = 0,0
            runStartTime = timer()
            for batch, (image, label) in enumerate(self.training_loader):
                image = image.to(self.device)
                label = label.to(self.device)

                self.optimizer.zero_grad()
                y_pred = self.model(image)
                loss = self.criterion(y_pred, label)
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(y_pred, 1)
                total += label.size(0)
                correct += predicted.eq(label).sum().item()
                train_loss += loss.item()
                train_acc = 100 * correct / total

                #Timing
                runEndTime = timer()
                runTime = runEndTime - runStartTime
                #runTime = timeStrFromS(runTime)

                thisTrainLoss = train_loss/(batch +1)
                lossArr.append(thisTrainLoss)

                #if batch%50==0:
                #print(f"Model: {modelName}, Epoch, batch: {epoch}, {batch} | Train Loss: {thisTrainLoss:.3f} | Train Acc: {train_acc:.2f}")
                #print(f"Epoch: {epoch} | Batch: {batch} | lr: {thisLR} | Train Loss: {thisTrainLoss:.3f} | Train Acc: {train_acc:.2f} | Elapsed Time: {runTimeStr}")
                print(f"Epoch: {epoch} | Batch: {batch} | Train Loss: {thisTrainLoss:.3f} | Train Acc: {train_acc:.2f} | Elapsed Time: {runTime}")
                    #print(f"Epoch: {epoch} | Train Loss: {train_loss / (batch + 1):.3f} | Train Acc: {train_acc:.2f}") 

        plt.plot(range((epoch+1)*(batch+1)), lossArr)    
        plt.title("Training loss v Batch")
        plt.savefig("../output/trainingLoss.png")
        #plt.show()
        return thisTrainLoss # Return the final training loss
    
    def test(self, classes):
        self.model.eval()
        test_loss, test_acc, total = 0, 0, 0
        correct,total = 0,0
        y_preds =[] # for confusion matrix
        y_targs = []
        with torch.inference_mode():
            for image, label in self.testing_loader:
                image = image.to(self.device)
                label = label.to(self.device)
                test_pred = self.model(image)
                loss = self.criterion(test_pred, label) 
                _, predicted = torch.max(test_pred, 1)
                total += label.size(0)
                correct += predicted.eq(label).sum().item()
                test_loss += loss.item()
                test_acc = 100 * correct / total

                y_preds.append(test_pred) #for confusion matrix
                y_targs.append(label) #for confusion matrix 
            finalTestLoss = test_loss/len(self.testing_loader)
            print(f"Test Loss: {finalTestLoss:.3f} | Test Acc: {test_acc:.2f}")
        
        from torchmetrics import ConfusionMatrix
        from mlxtend.plotting import plot_confusion_matrix

        y_pred_tensor = torch.cat(y_preds)
        y_targ_tensor = torch.cat(y_targs)
        confMat = ConfusionMatrix(num_classes=len(classes), task='multiclass')
        confMat_values = confMat(preds=y_pred_tensor, target=y_targ_tensor)

        plot_confusion_matrix(conf_mat=confMat_values.numpy(), class_names=classes)
        plt.title("Confusion Matrix")
        plt.savefig("../output/confMatrix.png")
        #plt.show()

        return finalTestLoss, test_acc