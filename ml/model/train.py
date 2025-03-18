import torch
from torchvision import transforms
from tqdm import tqdm
from dataset import ChessPiecesDataset
from chess_piece_nn import ChessPieceNN


data_dir = "data"

transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(48)])
dataset = ChessPiecesDataset(data_dir, transform)

train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=16, shuffle=True, num_workers=0
)

model = ChessPieceNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = torch.nn.CrossEntropyLoss()

torch.manual_seed(7454)

if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
else:
    device = torch.device("cpu")

model.train()
for epoch in range(10):
    pbar = tqdm(train_loader)
    total_correct = 0
    total_samples = 0
    for batch_idx, batch in enumerate(pbar):
        images, targets = batch
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(output, 1)
        total_correct += (predicted == targets).sum().item()
        total_samples += targets.size(0)

        accuracy = total_correct / total_samples
        pbar.set_description(
            f"Train loss: {loss.item():.2f}, Accuracy: {accuracy:.4f}"
        )

torch.save({"model_state_dict": model.state_dict()}, "model.pt")
