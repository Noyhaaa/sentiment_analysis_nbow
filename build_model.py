import collections
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchtext
import torch.nn as nn
import torch.optim as optim
import tqdm

from pre_processing import PreProcessing

class NBoW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, ids):
        # ids = [batch size, seq len]
        embedded = self.embedding(ids)
        # embedded = [batch size, seq len, embedding dim]
        pooled = embedded.mean(dim=1)
        # pooled = [batch size, embedding dim]
        prediction = self.fc(pooled)
        # prediction = [batch size, output dim]
        return prediction

class TrainModel(PreProcessing):
    def __init__(self):
        super().__init__()
        # Retrive dataloader 
        self.train_data_loader, self.valid_data_loader, self.test_data_loader = self.create_dataloader()

        self.vocab_size = len(self.vocab)
        self.embedding_dim = 300
        self.output_dim = len(self.train_data.unique("label"))
        self.model = NBoW(self.vocab_size, self.embedding_dim, self.output_dim , self.pad_index)

        # Initialize parameters 
        self.vectors = torchtext.vocab.GloVe()
        self.pretrained_embedding = self.vectors.get_vecs_by_tokens(self.vocab.get_itos())
        # Initialize embedding layer weight with embedding pretrained vector 
        self.model.embedding.weight.data = self.pretrained_embedding
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

    def model_numbers_parameters(self, model):
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"The model has {count_parameters(model):,} trainable parameters")

    def train(self, data_loader, model, criterion, optimizer, device):
        model.train()
        epoch_losses = []
        epoch_accs = []
        for batch in tqdm.tqdm(data_loader, desc="training..."):
            ids = batch["ids"].to(device)
            label = batch["label"].to(device)
            prediction = model(ids)
            loss = criterion(prediction, label)
            accuracy = self.get_accuracy(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
        return np.mean(epoch_losses), np.mean(epoch_accs)

    def evaluate(self, data_loader, model, criterion, device):
        model.eval()
        epoch_losses = []
        epoch_accs = []
        with torch.no_grad():
            for batch in tqdm.tqdm(data_loader, desc="evaluating..."):
                ids = batch["ids"].to(device)
                label = batch["label"].to(device)
                prediction = model(ids)
                loss = criterion(prediction, label)
                accuracy = self.get_accuracy(prediction, label)
                epoch_losses.append(loss.item())
                epoch_accs.append(accuracy.item())
        return np.mean(epoch_losses), np.mean(epoch_accs)

    def get_accuracy(self, prediction, label):
        batch_size, _ = prediction.shape
        predicted_classes = prediction.argmax(dim=-1)
        correct_predictions = predicted_classes.eq(label).sum()
        accuracy = correct_predictions / batch_size
        return accuracy

    def train_model(self, train_data_loader, valid_data_loader, model, criterion, optimizer, device):
        n_epochs = 10
        best_valid_loss = float("inf")

        metrics = collections.defaultdict(list)

        for epoch in range(n_epochs):
            train_loss, train_acc = self.train(
                train_data_loader, model, criterion, optimizer, device
            )
            valid_loss, valid_acc = self.evaluate(valid_data_loader, model, criterion, device)
            metrics["train_losses"].append(train_loss)
            metrics["train_accs"].append(train_acc)
            metrics["valid_losses"].append(valid_loss)
            metrics["valid_accs"].append(valid_acc)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), "nbow.pt")
            print(f"epoch: {epoch}")
            print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
            print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(metrics["train_losses"], label="train loss")
        ax.plot(metrics["valid_losses"], label="valid loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_xticks(range(n_epochs))
        ax.legend()
        ax.grid()
        #plt.show()

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(metrics["train_accs"], label="train accuracy")
        ax.plot(metrics["valid_accs"], label="valid accuracy")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_xticks(range(n_epochs))
        ax.legend()
        ax.grid()
        plt.show()