import argparse
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
import torchmetrics
from torchvision.transforms import v2
from torchvision.datasets import Imagenette
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import torchvision.models as models

torch.manual_seed(23)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# print(torch.cuda.get_arch_list())


class BaseCNNModel(L.LightningModule):
    def __init__(self, num_cls=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 32x80x80

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 64x40x40

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 128x20x20

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 256x10x10

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 512x5x5

            nn.AdaptiveAvgPool2d((1, 1)),  # -> 512x1x1
            # adaptive pooling to get fixed size output before flattening
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_cls)
        )

        # evaluate the model using accuracy metric no propagation of gradients
        self.accuracy = torchmetrics.Accuracy(
            task='multiclass', num_classes=num_cls)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch):
        x, y = batch  # since each batch is a tuple of (images, labels)
        logits = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class AllCNNModel(L.LightningModule):
    def __init__(self, num_cls=10):
        super().__init__()

        self.classifier = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2),  # 96x32x32
            nn.BatchNorm2d(96),
            nn.ReLU(),

            # 1x1 conv for channel mixing (padding=0)
            nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),

            # Second conv block
            nn.Conv2d(96, 192, kernel_size=5, stride=1,
                      padding=2),  # 192x32x32
            nn.BatchNorm2d(192),
            nn.ReLU(),

            # 1x1 conv again
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            # Downsampling
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # → 192x16x16

            # Final conv layers
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            # Output layer (num_classes = 10)
            nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1)),  # → [B, 10, 1, 1]
            nn.Flatten()  # → [B, 10]
        )

        self.accuracy = torchmetrics.Accuracy(
            task='multiclass', num_classes=num_cls, top_k=1)

    def forward(self, x):
        x = self.classifier(x)
        return x

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
        responsible for updating the model parameters during training base on cross entropy loss
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class PreTrainedModel(L.LightningModule):
    def __init__(self, num_cls=10):
        super().__init__()
        # Placeholder for pre-trained model initialization
        # This would typically load a pre-trained model like ResNet, VGG, etc.
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_cls)
        )

        self.accuracy = torchmetrics.Accuracy(
            task='multiclass', num_classes=num_cls, top_k=1)

    def forward(self, x):
        # effnet already flattens the output, no need to view()
        x = self.backbone(x)
        return x

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
        responsible for updating the model parameters during training base on cross entropy loss
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# --------------------
# Step 2: Define the data, transform the data
# --------------------

train_transform = v2.Compose([
    v2.CenterCrop(64),
    v2.Resize((64, 64)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = v2.Compose([
    v2.CenterCrop(64),
    v2.Resize((64, 64)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = Imagenette(
    root='data/imagenette/train/', transform=train_transform, split='train', download=False, size='160px')

# use 10% of data training set for validation
train_set_size = int(len(train_dataset) * 0.9)
val_set_size = len(train_dataset) - train_set_size
train_set, val_set = torch.utils.data.random_split(
    train_dataset, [train_set_size, val_set_size])

val_set.dataset.transform = test_transform

# use batch loader to load data in batches
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=128, shuffle=False)

# test dataset
test_dataset = Imagenette(
    root='data/imagenette/test/', transform=test_transform, download=False, split='val', size='160px')

# init model ------------------------------
print("Usage: python main.py <model_choice>, 1 - basic, 2 - all CNN, 3 Pre trained")
parser = argparse.ArgumentParser(description="Sample script with argument")
# positional argument
parser.add_argument(
    "model_choice", help="1 - basic, 2 - all CNN, 3 Pre trained")

args = parser.parse_args()
print(f"Model choice: {args.model_choice}")

if args.model_choice == "1":
    model = BaseCNNModel(num_cls=10)
elif args.model_choice == "2":
    model = AllCNNModel(num_cls=10)
elif args.model_choice == "3":
    model = PreTrainedModel(num_cls=10)
    # -----------------------------------------
    # early stopping
early_stopping = EarlyStopping(
    monitor='val_loss', patience=10, mode='min', verbose=True)

# configure checkpoints
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss', mode='min', save_top_k=1, dirpath='checkpoints/', filename=f'best_model_{args.model_choice}')

# fit the model
trainer = L.Trainer(
    max_epochs=50,
    accelerator='auto',
    devices=1,
    callbacks=[early_stopping, checkpoint_callback],

)
trainer.fit(model, train_loader, val_loader)
# test on test dataset
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=128, shuffle=False)
trainer.test(model, dataloaders=test_loader)
