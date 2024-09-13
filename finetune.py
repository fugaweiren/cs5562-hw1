import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import os
from resnet_attack_todo import  ResnetPGDAttacker
from tqdm import tqdm
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

class ResnetFinetune(ResnetPGDAttacker):
    '''
    Inherit self.model & self.dataloader from ResnetPGDAttacker
    '''
    def init_metrics(self):
        '''
        Initialise train & test metrics for further analysis
        '''
        self.train_accs = []
        self.train_losses = []

        self.test_accs = []
        self.test_losses = []
        self.use_adv_valid =False

    def add_optimizer(self, alpha):
        '''
        Unfreeze all parameters for full finetuning & adding optimizer for training

        :param alpha: Learning rate. Default set as 1e-5
        '''
        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
    
    def add_trainloader(self, dataloader: DataLoader):
        '''
        Adding trainloader for custom adverserial dataset
        '''
        self.trainloader = dataloader
    
    def finetune_train(self, eps, alpha, steps, test_batch_num):
        '''
        Main Training & testing loop for finetuning on adverserial samples. 

        :param eps: Epsilon value in Gradient Descent. Set to 0.1.
        :param alpha: Learning Rate value in Gradient Descent
        :param steps: Number of epochs for training
        :param test_batch_num: batch_num for clean samples from Imagenet-HuggingFace dataset (Only for validation)
        :return: Update train accuracy/loss on adversarial images and test accuracy/loss on clean/adversarial images.
        '''

        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        for epoch in range(steps):
            
            correct = 0
            total = 0
            train_loss = 0

            # Training starts 
            for i, inputs in enumerate(tqdm(self.trainloader)):
                inputs = [v.to(self.device) for v in inputs]
                b_correct, b_total, b_loss = self.finetune(*inputs)
                correct +=b_correct
                total +=b_total
                train_loss +=b_loss
            
            num_batch = i+1 
            print(f"Epoch {epoch} training loss: {train_loss/num_batch:>7f}")
            print(f"Epoch {epoch} training acc : {correct/total:>7f}")
            self.train_losses.append(train_loss/num_batch)
            self.train_accs.append(correct/total)
            

            # Testing
            if self.use_adv_valid:
                # Compute test metrics on adversarial dataset(s)
                test_correct , test_total, test_loss = self.compute_accuracy_loss_no_batch_num()
            else:
                # Compute test metrics on clean ImageNet dataset(s)
                test_correct , test_total, test_loss = self.compute_accuracy_loss(test_batch_num)
            
            print(f"Epoch {epoch} test loss: {test_loss:>7f}")
            print(f"Epoch {epoch} test acc : {test_correct/test_total:>7f}")
            self.test_losses.append(test_loss)
            self.test_accs.append(test_correct/test_total)

    def compute_accuracy_loss_no_batch_num(self):
        '''
        Compute model test accuracy & loss from custom adverserial dataset (using as self.dataloader)
        :return: Update model test accuracy & average loss over batch
        '''
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            total_loss = 0
            for i, inputs in enumerate(tqdm(self.dataloader)):
                inputs = [v.to(self.device) for v in inputs]
                outputs = self.model(inputs[0]).softmax(1)
                predictions = outputs.argmax(dim=1)

                loss = self.loss_fn(outputs, inputs[1])
                
                correct += torch.sum(predictions == inputs[1]).item()
                total += len(inputs[1])
                total_loss+=loss
            num_batch = i+1 
        return correct , total, total_loss/num_batch
    
    def compute_accuracy_loss(self, batch_num):
        '''
        Compute model accuracy for specified number of data batches from clean ImageNet datatset (using as self.dataloader)
        :param batch_num: Number of batches on which we compute model accuracy lose
        :return: Update model test accuracy & average loss over batch
        '''
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            total_loss = 0
            for i, inputs in enumerate(tqdm(self.dataloader, total=batch_num)):
                inputs = {k: v.to(self.device) for (k, v) in inputs.items()}
                outputs = self.model(inputs['image']).softmax(1)
                predictions = outputs.argmax(dim=1)
                loss = self.loss_fn(outputs, inputs['label'])
                correct += (predictions == inputs['label']).sum().item()
                total += predictions.size(0)
                total_loss+=loss
            num_batch = i+1 
        return correct , total, total_loss/num_batch


    def finetune(self, adv_images, labels):
        '''
        Training loop for 1 batch
        :param adv_images: Batch of adverserial input images that we finetune self.model on, size (BATCH_SIZE, 3, 224, 224)
        :param labels:  Batch of input labels, size (BATCH_SIZE)
        :return: Update model train accuracy & train loss over 1 batch
        '''
        self.model.train()
        adv_images.requires_grad =False
        self.optimizer.zero_grad()
        outputs = self.model(adv_images).softmax(1)
        predictions = outputs.argmax(dim=1)
        loss = self.loss_fn(outputs, labels)
        loss.backward()
        self.optimizer.step()
        
        correct = torch.sum(predictions == labels).detach().item()
        total = len(labels)
        loss_value = loss.detach().item()
        # print(f"batch training loss: {loss_value:>7f}")
        # print(f"batch training acc : {correct/total:>7f}")
        return correct, total, loss_value
    
class AdversialDataset(Dataset):
    def __init__(self, path = "results", load=True, data_file="eps_01"):
        '''
        Custom Dataset Object for loading adversarial data from different eps. 
        :param path: Directory that where torch.save() saves the adversarial images and labels.
        :param load: Flag to construct dataset using adversarial images from all eps=0.03,0.1, ..., 0.9 if True.
            Construct dataset only using f"{path}/{datafile}" (1-eps dataset) if False
        :param data_file: Dict filename that contains labels and adversarial samples
        '''
        self.labels = None
        self.adv_images = None

        if load:
            # Load every datafile in directory
            self.load(path)
        else:
            # Load specified datafile 
            self.load_1_eps(path,data_file)
        self.labels = self.labels.type(torch.LongTensor) 

    def load_1_eps(self, path, data_file):
        '''
        Load specified datafile  
        :param path: Directory that where torch.save() saves the adversarial images and labels.
        :param data_file: Dict filename that contains labels and adversarial samples
        :return:  Unpickled adversarial images and labels given specified datafile
        '''

        eps_dict = torch.load(f"{path}/{data_file}")
        self.labels = eps_dict["labels"]
        self.adv_images = eps_dict["adv_images"]

    def load(self, path):
        '''
        Load specified datafile  
        :param path: Directory that where torch.save() saves the adversarial images and labels.
        :return: Unpickled adversarial images and labels given directory
        '''

        for i, eps in enumerate(os.listdir(path)):
            eps_dict = torch.load(f"{path}/{eps}")
            if i ==0:
                # Initialise self.label & self.adv_images with the first dataset
                self.labels = eps_dict["labels"]
                self.adv_images = eps_dict["adv_images"]
            else:
                # Collate all datasets into AdversialDataset object
                self.labels = torch.cat([self.labels, eps_dict["labels"]])
                self.adv_images = torch.cat([self.adv_images, eps_dict["adv_images"]])

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.adv_images[idx], self.labels[idx]

def train_val_dataset(dataset: AdversialDataset, val_split=0.25):
    '''
    Train-test-split on AdversialDataset() using sklearn
    :param dataset: AdversialDataset() which holds all the advesarial samples and labels
    :param val_split: Percentage of total samples used for testset. Set as 25%.
    :return: train dataset, test dataset
    '''
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)