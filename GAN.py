import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


# Mount Google Drive to access the dataset
from google.colab import drive

drive.mount('/content/drive')

# Load and preprocess the data
data = pd.read_csv('/content/drive/MyDrive/ICCIT2023/paper.csv')
data.dropna(inplace=True)

# Encode categories
label_encoder = LabelEncoder()
data['Sub-task A'] = label_encoder.fit_transform(data['Sub-task A'])

# Define a custom Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return self.texts.shape[0]

    def __getitem__(self, idx):
        text = self.texts[idx].toarray().squeeze()
        label = self.labels[idx] if self.labels is not None else -1
        return torch.tensor(text, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Prepare the input features and labels
X = data['Text']
y = data['Sub-task A']

# Vectorize the 'Text' column
tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
text_features = tfidf_vectorizer.fit_transform(X)

# Split the data into training and testing sets (80% train, 20% test)
train_texts, test_texts, train_labels, test_labels = train_test_split(text_features, y, test_size=0.2, random_state=42, stratify=y)

# Further split the training data into labeled and unlabeled
n_labeled = int(0.8 * train_texts.shape[0])  # 80% of the training data is labeled
labeled_texts = train_texts[:n_labeled]
labeled_labels = train_labels[:n_labeled].values
unlabeled_texts = train_texts[n_labeled:]

# Create PyTorch Datasets and DataLoaders
labeled_dataset = TextDataset(labeled_texts, labeled_labels)
unlabeled_dataset = TextDataset(unlabeled_texts)
test_dataset = TextDataset(test_texts, test_labels.values)
labeled_loader = DataLoader(labeled_dataset, batch_size=32, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the discriminator model architecture
class Discriminator(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes + 1)  # num_classes + 1 for the 'fake' class
        )

    def forward(self, x):
        return self.model(x)

# Define the generator model
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, z):
        return self.model(z)

# Hyperparameters
input_dim = text_features.shape[1]
latent_dim = 100  # Latent vector size for the generator
num_classes = len(label_encoder.classes_)

# Initialize the models, optimizers, and loss function
discriminator = Discriminator(input_dim, num_classes)
generator = Generator(latent_dim, input_dim)
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=0.001)
optimizer_generator = optim.Adam(generator.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification

# GAN Training Function
def train_gan(generator, discriminator, optimizer_g, optimizer_d, labeled_loader, unlabeled_loader, num_epochs=10):
    history = {'d_loss': [], 'g_loss': [], 'accuracy': []}
    
    for epoch in range(num_epochs):
        total_d_loss, total_g_loss = 0, 0
        correct, total = 0, 0
        
        print(f"Starting GAN Epoch {epoch + 1}/{num_epochs}")
        
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        
        while True:
            try:
                x_labeled, y_labeled = next(labeled_iter)
            except StopIteration:
                break
            
            optimizer_d.zero_grad()
            real_outputs = discriminator(x_labeled)
            d_loss_real = criterion(real_outputs, y_labeled)
            
            batch_size = x_labeled.size(0)
            z = torch.randn(batch_size, latent_dim)
            fake_data = generator(z)
            fake_outputs = discriminator(fake_data.detach())
            fake_labels = torch.full((batch_size,), num_classes, dtype=torch.long)
            d_loss_fake = criterion(fake_outputs, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()
            
            optimizer_g.zero_grad()
            fake_data = generator(z)
            fake_outputs = discriminator(fake_data)
            target_labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)
            g_loss = criterion(fake_outputs, target_labels)
            g_loss.backward()
            optimizer_g.step()
            
            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            
            _, predicted = torch.max(real_outputs.data, 1)
            total += y_labeled.size(0)
            correct += (predicted == y_labeled).sum().item()
        
        avg_d_loss = total_d_loss / len(labeled_loader)
        avg_g_loss = total_g_loss / len(labeled_loader)
        accuracy = 100 * correct / total
        history['d_loss'].append(avg_d_loss)
        history['g_loss'].append(avg_g_loss)
        history['accuracy'].append(accuracy)
        
        print(f"Epoch {epoch + 1} Completed. Discriminator Loss: {avg_d_loss:.4f}, Generator Loss: {avg_g_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return history

# Evaluation Function with ROC
def evaluate_model_with_roc(discriminator, test_loader):
    discriminator.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for x, y in test_loader:
            outputs = discriminator(x)
            outputs = outputs[:, :num_classes]
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return all_labels, all_preds, all_probs

# Train the GAN and capture history
print("Training GAN model...")
history = train_gan(generator, discriminator, optimizer_generator, optimizer_discriminator, labeled_loader, unlabeled_loader)

# Evaluate the Discriminator and get probabilities
print("Evaluating Discriminator...")
gan_labels, gan_preds, gan_probs = evaluate_model_with_roc(discriminator, test_loader)

# Print evaluation results for GAN (Discriminator)
print("GAN Classification Report (Discriminator):")
print(classification_report(gan_labels, gan_preds))

# Plot confusion matrix
def plot_confusion_matrix(labels, preds, title):
    conf_matrix = confusion_matrix(labels, preds)
    conf_matrix_df = pd.DataFrame(conf_matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

plot_confusion_matrix(gan_labels, gan_preds, title="Confusion Matrix")

# Plot training history
def plot_history(history):
    epochs = range(1, len(history['d_loss']) + 1)
    
    plt.figure(figsize=(12, 6))
    
    # Plot Discriminator Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['d_loss'], 'b-', label='Discriminator Loss')
    plt.plot(epochs, history['g_loss'], 'r-', label='Generator Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['accuracy'], 'g-', label='Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)

# Plot ROC curve
def plot_roc_curve(labels, probs, num_classes):
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(labels, np.array(probs)[:, i], pos_label=i)
        roc
