import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from customized_database import EmoDB
from CNN import CNNNetwork

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

AUDIO_DIR = "./EmoDb_berlin_database/audio"
ANNOTATIONS_FILE = "./EmoDb_berlin_database/metadata/EmoDB.csv"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


def create_data_loader(train_data, batch_size):
    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    return train_data_loader


def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # calculate loss
        prediction = model(inputs)
        loss = loss_fn(prediction, targets)

        # backpropagate loss and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"Loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
        print("-----------")
    print("Training is done")


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    device = "cpu"

    emodb = EmoDB(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram,
                  SAMPLE_RATE, NUM_SAMPLES, device)

    train_data_loader = create_data_loader(emodb, BATCH_SIZE)

    cnn = CNNNetwork().to(device)

    # instantiate loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_data_loader, loss_fn, optimiser, device,
          EPOCHS)

    torch.save(cnn.state_dict(), "cnn.pth")
    print("Model trained and stored at cnn.pth")
