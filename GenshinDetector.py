import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

data_dir = 'dataset'
data_dir_cropped = 'dataset_cropped'
data_dir_expanded = 'dataset_expanded'

train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_dataset_cropped = datasets.ImageFolder(root=data_dir_cropped, transform=transform)
train_dataset_expanded = datasets.ImageFolder(root=data_dir_expanded, transform=transform)

combined_dataset = ConcatDataset([train_dataset_cropped, train_dataset_expanded])
# combined_dataset = ConcatDataset([train_dataset])

train_loader = DataLoader(combined_dataset, batch_size=5, shuffle=True, num_workers=4)

# https://www.geeksforgeeks.org/auto-encoders/
# Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        encoded_dim = 128 * 8 * 8
        self.decoder_input = nn.Linear(encoded_dim, encoded_dim)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded_input = self.decoder_input(encoded)
        decoded = self.decoder(decoded_input)
        return decoded

    def encode(self, x):
        return self.encoder(x)

model = ConvAutoencoder().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, loader, criterion, optimizer, device, num_epochs=5):
    model.train()
    epoch_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(loader)
        epoch_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.2f}')

    return epoch_losses

def extract_features_in_chunks(model, loader, device, chunk_size=1000):
    model.eval()
    features = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            encoded = model.encode(images)
            features.append(encoded.cpu().numpy())
            current_size = sum(f.shape[0] for f in features)
            if current_size >= chunk_size:
                yield np.concatenate(features, axis=0)
                features = []
    if features:
        yield np.concatenate(features, axis=0)

def predict_image(image_path, model, transform, kmeans, device):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image {image_path} does not exist.")
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        encoded = model.encode(image)
    feature = encoded.cpu().numpy()
    cluster = kmeans.predict(feature)
    return cluster[0]

if __name__ == "__main__":
    num_epochs = 25
    epoch_losses = train(model, train_loader, criterion, optimizer, device, num_epochs)

    plt.figure(figsize=(20,10))
    plt.plot(range(1, num_epochs+1), epoch_losses)
    plt.title('training loss')
    plt.xlabel('epoch')
    plt.ylabel('mean-squared err Loss')
    plt.show()

    torch.save(model.state_dict(), 'autoencoder_model.pth')
    print("Training complete and model saved.")

    num_clusters = 5
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=11, batch_size=50)
    for feat_chunk in extract_features_in_chunks(model, train_loader, device, chunk_size=500):
        kmeans.partial_fit(feat_chunk)

    image_paths = [
        'Hu_Tao_Test.jpg',
        'Kokomi_Test.jpg',
        'Ayaka_Test.jpg',
        'Albedo_Test.jpg',
        'Ryu_Test.jpg',
        'Terry_Bogard_Test.jpg',
        'X_Test.jpg'
    ]

    for image_path in image_paths:
        if os.path.exists(image_path):
            cluster = predict_image(image_path, model, transform, kmeans, device)
            print(f'The predicted cluster for {image_path} is: {cluster}')
        else:
            print(f"Image {image_path} does not exist.")
