# ============================================================
# ACRIMA + RIM-ONE DL
# ConvNeXt-Tiny + Multi-Level Feature Fusion (MLFF)
# Data Augmentation + Grad-CAM + Pretrained BLIP VLM
#
# NO VLM FINE-TUNING
# NO DEMO / SINGLE IMAGE PREDICTION IN THIS CELL
# ============================================================


# ============================================================
# 1. INSTALL REQUIRED PACKAGES
# ============================================================

!pip install -q transformers accelerate


# ============================================================
# 2. IMPORT LIBRARIES
# ============================================================

import os
import random
import zipfile
import shutil
import copy
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ConvNeXt_Tiny_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration
)

print("PyTorch version:", torch.__version__)


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# 4. GOOGLE DRIVE
# ============================================================

from google.colab import drive

drive.mount('/content/drive')


# ============================================================
# 5. PATHS
# ============================================================

# ------------------------------------------------------------
# ACRIMA
# ------------------------------------------------------------

ACRIMA_DIR = "/content/drive/MyDrive/Database/Database/Images"


# ------------------------------------------------------------
# RIM-ONE ZIP
#
# Upload your RIM-ONE_DL_images.zip to Google Drive.
# Change this path only if you put the ZIP somewhere else.
# ------------------------------------------------------------



RIM_EXTRACT_DIR = "/content/drive/MyDrive/RIM-ONE_DL_images"


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

RESULTS_DIR = "/content/drive/MyDrive/ConvNeXt_MLFF_Combined_Results"

CHECKPOINT_DIR = os.path.join(
    RESULTS_DIR,
    "checkpoints"
)

GRADCAM_DIR = os.path.join(
    RESULTS_DIR,
    "gradcam"
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)


CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_convnext_tiny_mlff_combined.pth"
)


print("\nResults directory:")
print(RESULTS_DIR)



# ============================================================
# 7. FIND ACTUAL RIM-ONE ROOT
# ============================================================

# Your ZIP contains:
#
# RIM-ONE_DL_images/
#     partitioned_by_hospital/
#
# Therefore this should exist:

RIM_ROOT = os.path.join(
    RIM_EXTRACT_DIR,
    "partitioned_by_hospital"
)

if not os.path.exists(RIM_ROOT):

    raise FileNotFoundError(
        f"RIM-ONE root not found:\n{RIM_ROOT}"
    )

print("\nRIM-ONE root:")
print(RIM_ROOT)


# ============================================================
# 8. CREATE ACRIMA DATAFRAME
# ============================================================

def collect_acrima_images(root_dir):

    records = []

    if not os.path.exists(root_dir):
        raise FileNotFoundError(
            f"ACRIMA directory not found:\n{root_dir}"
        )

    for filename in os.listdir(root_dir):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        ):
            continue

        image_path = os.path.join(
            root_dir,
            filename
        )

        # ACRIMA labeling rule:
        # "_g_" -> glaucoma
        # otherwise -> normal

        if "_g_" in filename.lower():
            label = 1
            class_name = "glaucoma"
        else:
            label = 0
            class_name = "normal"

        records.append({
            "image_path": image_path,
            "label": label,
            "class_name": class_name,
            "dataset": "ACRIMA"
        })

    return pd.DataFrame(records)


acrima_df = collect_acrima_images(
    ACRIMA_DIR
)


print("\nACRIMA:")
print(acrima_df["class_name"].value_counts())
print("Total:", len(acrima_df))


# ============================================================
# 9. CREATE RIM-ONE DATAFRAME
# ============================================================

def collect_rimone_images(root_dir):

    records = []

    for split_name in [
        "training_set",
        "test_set"
    ]:

        split_dir = os.path.join(
            root_dir,
            split_name
        )

        if not os.path.exists(split_dir):
            continue

        for class_name, label in [
            ("normal", 0),
            ("glaucoma", 1)
        ]:

            class_dir = os.path.join(
                split_dir,
                class_name
            )

            if not os.path.exists(class_dir):
                continue

            for filename in os.listdir(class_dir):

                if not filename.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
                ):
                    continue

                image_path = os.path.join(
                    class_dir,
                    filename
                )

                records.append({
                    "image_path": image_path,
                    "label": label,
                    "class_name": class_name,
                    "dataset": "RIM-ONE"
                })

    return pd.DataFrame(records)


rim_df = collect_rimone_images(
    RIM_ROOT
)


print("\nRIM-ONE:")
print(rim_df["class_name"].value_counts())
print("Total:", len(rim_df))


# ============================================================
# 10. COMBINE ACRIMA + RIM-ONE
# ============================================================

combined_df = pd.concat(
    [
        acrima_df,
        rim_df
    ],
    ignore_index=True
)


# Shuffle once
combined_df = combined_df.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)


print("\n======================================")
print("COMBINED DATASET")
print("======================================")

print("\nTotal images:", len(combined_df))

print("\nClass distribution:")
print(
    combined_df["class_name"].value_counts()
)

print("\nDataset distribution:")
print(
    combined_df["dataset"].value_counts()
)

print("\nDataset + class:")
print(
    pd.crosstab(
        combined_df["dataset"],
        combined_df["class_name"]
    )
)


# ============================================================
# 11. REMOVE EXACT DUPLICATE FILE PATHS
# ============================================================

before = len(combined_df)

combined_df = combined_df.drop_duplicates(
    subset=["image_path"]
).reset_index(drop=True)

after = len(combined_df)

print(
    f"\nRemoved duplicate paths: {before - after}"
)


# ============================================================
# 12. CHECK THAT IMAGES CAN BE OPENED
# ============================================================

print("\nChecking image files...")

bad_images = []

for path in combined_df["image_path"]:

    try:

        with Image.open(path) as img:
            img.verify()

    except Exception:

        bad_images.append(path)


if len(bad_images) > 0:

    print(
        f"Found {len(bad_images)} invalid images."
    )

    combined_df = combined_df[
        ~combined_df["image_path"].isin(bad_images)
    ].reset_index(drop=True)

else:

    print("All images are readable.")


# ============================================================
# 13. STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# 70% TRAIN
# 15% VALIDATION
# 15% TEST

train_df, temp_df = train_test_split(
    combined_df,
    test_size=0.30,
    stratify=combined_df["label"],
    random_state=SEED
)


val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=SEED
)


train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


print("\n======================================")
print("DATASET SPLIT")
print("======================================")

print(
    f"Train      : {len(train_df)}"
)

print(
    f"Validation : {len(val_df)}"
)

print(
    f"Test       : {len(test_df)}"
)


print("\nTRAIN distribution:")
print(
    train_df["class_name"].value_counts()
)


print("\nVALIDATION distribution:")
print(
    val_df["class_name"].value_counts()
)


print("\nTEST distribution:")
print(
    test_df["class_name"].value_counts()
)


print("\nTrain dataset source:")
print(
    train_df["dataset"].value_counts()
)

print("\nValidation dataset source:")
print(
    val_df["dataset"].value_counts()
)

print("\nTest dataset source:")
print(
    test_df["dataset"].value_counts()
)


# ============================================================
# 14. SAVE SPLIT INFORMATION
# ============================================================

train_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "train_split.csv"
    ),
    index=False
)

val_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "validation_split.csv"
    ),
    index=False
)

test_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "test_split.csv"
    ),
    index=False
)


# ============================================================
# 15. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", DEVICE)


# ============================================================
# 16. DATA AUGMENTATION
# ============================================================

# IMPORTANT:
#
# Augmentation is used ONLY for TRAINING.
#
# Validation and TEST do NOT use augmentation.

train_transform = transforms.Compose([

    transforms.Resize((256, 256)),

    transforms.RandomResizedCrop(
        224,
        scale=(0.80, 1.0),
        ratio=(0.95, 1.05)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=15
    ),

    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20,
        saturation=0.10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Validation/Test = NO augmentation

eval_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 17. DATASET CLASS
# ============================================================

class FundusDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None
    ):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.transform = transform


    def __len__(self):

        return len(self.dataframe)


    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        image_path = row["image_path"]

        label = int(row["label"])

        image = Image.open(
            image_path
        ).convert("RGB")


        if self.transform is not None:

            image = self.transform(image)


        return (
            image,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# ============================================================
# 18. CREATE DATASETS
# ============================================================

train_dataset = FundusDataset(
    train_df,
    train_transform
)

val_dataset = FundusDataset(
    val_df,
    eval_transform
)

test_dataset = FundusDataset(
    test_df,
    eval_transform
)


# ============================================================
# 19. DATALOADERS
# ============================================================

BATCH_SIZE = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


print("\nDataLoaders created.")


# ============================================================
# 20. CONVNEXT-TINY + MLFF
# ============================================================

class ConvNeXtTinyMLFF(nn.Module):

    def __init__(self):

        super().__init__()


        # ----------------------------------------------------
        # PRETRAINED CONVNEXT-TINY
        # ----------------------------------------------------

        weights = ConvNeXt_Tiny_Weights.DEFAULT

        backbone = models.convnext_tiny(
            weights=weights
        )

        self.backbone = backbone


        # ----------------------------------------------------
        # MULTI-LEVEL FEATURES
        #
        # ConvNeXt Tiny:
        #
        # Level 1 -> 96
        # Level 2 -> 192
        # Level 3 -> 384
        # Level 4 -> 768
        # ----------------------------------------------------

        self.proj1 = nn.Sequential(
            nn.Conv2d(
                96,
                128,
                kernel_size=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.proj2 = nn.Sequential(
            nn.Conv2d(
                192,
                128,
                kernel_size=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.proj3 = nn.Sequential(
            nn.Conv2d(
                384,
                128,
                kernel_size=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.proj4 = nn.Sequential(
            nn.Conv2d(
                768,
                128,
                kernel_size=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )


        # ----------------------------------------------------
        # MLFF
        # ----------------------------------------------------

        self.fusion = nn.Sequential(

            nn.Conv2d(
                512,
                256,
                kernel_size=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True)
        )


        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.50),

            nn.Linear(
                128,
                128
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.40),

            nn.Linear(
                128,
                2
            )
        )


    def forward(
        self,
        x,
        return_features=False
    ):

        # ----------------------------------------------------
        # EXTRACT CONVNEXT MULTI-LEVEL FEATURES
        # ----------------------------------------------------

        features = []

        for i, layer in enumerate(
            self.backbone.features
        ):

            x = layer(x)

            if i in [1, 3, 5, 7]:

                features.append(x)


        f1 = self.proj1(features[0])
        f2 = self.proj2(features[1])
        f3 = self.proj3(features[2])
        f4 = self.proj4(features[3])


        # ----------------------------------------------------
        # UPSAMPLE ALL FEATURES TO f1 SIZE
        # ----------------------------------------------------

        target_size = f1.shape[-2:]


        f2 = nn.functional.interpolate(
            f2,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )

        f3 = nn.functional.interpolate(
            f3,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )

        f4 = nn.functional.interpolate(
            f4,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )


        # ----------------------------------------------------
        # MULTI-LEVEL FEATURE FUSION
        # ----------------------------------------------------

        fused = torch.cat(
            [
                f1,
                f2,
                f3,
                f4
            ],
            dim=1
        )


        fused = self.fusion(
            fused
        )


        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        pooled = self.pool(
            fused
        )

        logits = self.classifier(
            pooled
        )


        if return_features:

            return logits, fused

        return logits


# ============================================================
# 21. CREATE MODEL
# ============================================================

model = ConvNeXtTinyMLFF()

model = model.to(DEVICE)


print("\nModel created:")
print(model)


# ============================================================
# 22. PARTIAL FREEZING
# ============================================================

# Freeze only the earliest ConvNeXt stage.
#
# More layers remain trainable so the model can
# adapt to ACRIMA + RIM-ONE.

for i, layer in enumerate(
    model.backbone.features
):

    if i <= 1:

        for param in layer.parameters():

            param.requires_grad = False


# Count trainable parameters

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"\nTrainable parameters: {trainable_params:,}"
)

print(
    f"Total parameters: {total_params:,}"
)


# ============================================================
# 23. CLASS WEIGHTS
# ============================================================

class_counts = (
    train_df["label"]
    .value_counts()
    .sort_index()
)

num_normal = class_counts.get(
    0,
    1
)

num_glaucoma = class_counts.get(
    1,
    1
)

total_train = (
    num_normal +
    num_glaucoma
)

weight_normal = (
    total_train /
    (2 * num_normal)
)

weight_glaucoma = (
    total_train /
    (2 * num_glaucoma)
)

class_weights = torch.tensor(
    [
        weight_normal,
        weight_glaucoma
    ],
    dtype=torch.float32
).to(DEVICE)


print("\nClass weights:")
print(class_weights)


# ============================================================
# 24. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05
)


# ============================================================
# 25. OPTIMIZER
# ============================================================

optimizer = optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=3e-5,
    weight_decay=1e-3
)


# ============================================================
# 26. LEARNING RATE SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# 27. TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    all_preds = []
    all_labels = []


    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )


        optimizer.zero_grad()


        logits = model(
            images
        )


        loss = criterion(
            logits,
            labels
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        preds = torch.argmax(
            logits,
            dim=1
        )


        all_preds.extend(
            preds.detach()
            .cpu()
            .numpy()
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )


    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )


    epoch_acc = accuracy_score(
        all_labels,
        all_preds
    )


    return epoch_loss, epoch_acc


# ============================================================
# 28. VALIDATION FUNCTION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    all_labels = []
    all_preds = []
    all_probs = []


    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )


        logits = model(
            images
        )


        loss = criterion(
            logits,
            labels
        )


        running_loss += (
            loss.item()
            * images.size(0)
        )


        probs = torch.softmax(
            logits,
            dim=1
        )[:, 1]


        preds = torch.argmax(
            logits,
            dim=1
        )


        all_labels.extend(
            labels.cpu().numpy()
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_probs.extend(
            probs.cpu().numpy()
        )


    loss_value = (
        running_loss /
        len(loader.dataset)
    )


    accuracy = accuracy_score(
        all_labels,
        all_preds
    )


    precision = precision_score(
        all_labels,
        all_preds,
        zero_division=0
    )


    recall = recall_score(
        all_labels,
        all_preds,
        zero_division=0
    )


    f1 = f1_score(
        all_labels,
        all_preds,
        zero_division=0
    )


    try:

        auc = roc_auc_score(
            all_labels,
            all_probs
        )

    except:

        auc = 0.0


    return {
        "loss": loss_value,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "labels": np.array(all_labels),
        "preds": np.array(all_preds),
        "probs": np.array(all_probs)
    }


# ============================================================
# 29. TRAINING LOOP
# ============================================================

EPOCHS = 30

EARLY_STOPPING_PATIENCE = 7

best_val_loss = float("inf")

best_state = None

patience_counter = 0


history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "val_auc": []
}


print("\n======================================")
print("STARTING TRAINING")
print("======================================")


for epoch in range(
    1,
    EPOCHS + 1
):


    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        DEVICE
    )


    val_metrics = evaluate(
        model,
        val_loader,
        criterion,
        DEVICE
    )


    val_loss = val_metrics["loss"]

    val_acc = val_metrics["accuracy"]

    val_auc = val_metrics["auc"]


    scheduler.step(
        val_loss
    )


    history["train_loss"].append(
        train_loss
    )

    history["train_acc"].append(
        train_acc
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_acc"].append(
        val_acc
    )

    history["val_auc"].append(
        val_auc
    )


    current_lr = optimizer.param_groups[0]["lr"]


    print(
        f"\nEpoch [{epoch:02d}/{EPOCHS}]"
    )

    print(
        f"Train Loss : {train_loss:.4f}"
    )

    print(
        f"Train Acc  : {train_acc:.4f}"
    )

    print(
        f"Val Loss   : {val_loss:.4f}"
    )

    print(
        f"Val Acc    : {val_acc:.4f}"
    )

    print(
        f"Val AUC    : {val_auc:.4f}"
    )

    print(
        f"LR         : {current_lr:.7f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL USING VALIDATION LOSS
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_state = copy.deepcopy(
            model.state_dict()
        )


        torch.save(
            {
                "model_state_dict":
                    best_state,

                "best_val_loss":
                    best_val_loss,

                "epoch":
                    epoch,

                "seed":
                    SEED
            },
            CHECKPOINT_PATH
        )


        patience_counter = 0


        print(
            "✓ Best model saved."
        )


    else:

        patience_counter += 1


        print(
            f"No improvement "
            f"({patience_counter}/"
            f"{EARLY_STOPPING_PATIENCE})"
        )


    if patience_counter >= EARLY_STOPPING_PATIENCE:

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# 30. LOAD BEST CHECKPOINT
# ============================================================

print("\n======================================")
print("LOADING BEST CHECKPOINT")
print("======================================")


if not os.path.exists(
    CHECKPOINT_PATH
):

    raise FileNotFoundError(
        "Best checkpoint was not created."
    )


# IMPORTANT:
#
# Earlier you got a PyTorch 2.6+
# weights_only error.
#
# weights_only=False fixes that for a
# checkpoint created by this trusted code.

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


print(
    "Best checkpoint loaded successfully."
)

print(
    "Best validation loss:",
    checkpoint["best_val_loss"]
)

print(
    "Best epoch:",
    checkpoint["epoch"]
)


# ============================================================
# 31. FINAL VALIDATION RESULTS
# ============================================================

val_metrics = evaluate(
    model,
    val_loader,
    criterion,
    DEVICE
)


print("\n======================================")
print("VALIDATION RESULTS")
print("======================================")


print(
    f"Accuracy  : "
    f"{val_metrics['accuracy']:.4f}"
)

print(
    f"Precision : "
    f"{val_metrics['precision']:.4f}"
)

print(
    f"Recall    : "
    f"{val_metrics['recall']:.4f}"
)

print(
    f"F1 Score  : "
    f"{val_metrics['f1']:.4f}"
)

print(
    f"AUC       : "
    f"{val_metrics['auc']:.4f}"
)


# ============================================================
# 32. FINAL TEST RESULTS
# ============================================================

test_metrics = evaluate(
    model,
    test_loader,
    criterion,
    DEVICE
)


print("\n======================================")
print("FINAL TEST RESULTS")
print("======================================")


print(
    f"Accuracy  : "
    f"{test_metrics['accuracy']:.4f}"
)

print(
    f"Precision : "
    f"{test_metrics['precision']:.4f}"
)

print(
    f"Recall    : "
    f"{test_metrics['recall']:.4f}"
)

print(
    f"F1 Score  : "
    f"{test_metrics['f1']:.4f}"
)

print(
    f"AUC       : "
    f"{test_metrics['auc']:.4f}"
)


# ============================================================
# 33. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_metrics["labels"],
    test_metrics["preds"]
)


print("\nConfusion Matrix:")
print(cm)


print("\nClassification Report:")

print(
    classification_report(
        test_metrics["labels"],
        test_metrics["preds"],
        target_names=[
            "Normal",
            "Glaucoma"
        ],
        zero_division=0
    )
)


# ============================================================
# 34. SAVE TEST RESULTS
# ============================================================

test_results_df = test_df.copy()

test_results_df["predicted_label"] = (
    test_metrics["preds"]
)

test_results_df["glaucoma_probability"] = (
    test_metrics["probs"]
)


test_results_df["prediction"] = (
    test_results_df["predicted_label"]
    .map({
        0: "normal",
        1: "glaucoma"
    })
)


test_results_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "combined_test_predictions.csv"
    ),
    index=False
)


# ============================================================
# 35. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "training_history.csv"
    ),
    index=False
)


# ============================================================
# 36. PLOT TRAINING CURVES
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history["train_loss"],
    label="Train Loss"
)

plt.plot(
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Training and Validation Loss"
)

plt.legend()

plt.grid(True)

plt.show()


plt.figure(figsize=(8, 5))

plt.plot(
    history["train_acc"],
    label="Train Accuracy"
)

plt.plot(
    history["val_acc"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "Training and Validation Accuracy"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 37. GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model

        self.target_layer = target_layer

        self.activations = None

        self.gradients = None


        self.forward_handle = (
            target_layer.register_forward_hook(
                self.save_activation
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self.save_gradient
            )
        )


    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output


    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]


    def generate(
        self,
        image_tensor,
        target_class=None
    ):

        self.model.eval()

        image_tensor = image_tensor.to(
            DEVICE
        )

        self.model.zero_grad()


        logits = self.model(
            image_tensor
        )


        if target_class is None:

            target_class = torch.argmax(
                logits,
                dim=1
            ).item()


        score = logits[
            0,
            target_class
        ]


        score.backward()


        activations = (
            self.activations
        )

        gradients = (
            self.gradients
        )


        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )


        cam = (
            weights *
            activations
        ).sum(
            dim=1
        )


        cam = torch.relu(
            cam
        )


        cam = cam.squeeze().detach().cpu().numpy()


        cam -= cam.min()


        if cam.max() > 0:

            cam /= cam.max()


        return cam, target_class


    def close(self):

        self.forward_handle.remove()

        self.backward_handle.remove()


# ============================================================
# 38. GRAD-CAM TARGET LAYER
# ============================================================

# Last ConvNeXt feature stage

target_layer = model.backbone.features[7]

gradcam = GradCAM(
    model,
    target_layer
)


print(
    "\nGrad-CAM initialized successfully."
)


# ============================================================
# 39. GENERATE SAMPLE GRAD-CAM
# ============================================================

def denormalize_image(
    tensor
):

    image = tensor.cpu().numpy()

    image = np.transpose(
        image,
        (1, 2, 0)
    )

    mean = np.array([
        0.485,
        0.456,
        0.406
    ])

    std = np.array([
        0.229,
        0.224,
        0.225
    ])

    image = (
        image * std
    ) + mean

    image = np.clip(
        image,
        0,
        1
    )

    return image


sample_image, sample_label = (
    test_dataset[0]
)


cam, predicted_class = gradcam.generate(
    sample_image.unsqueeze(0),
    target_class=None
)


original_image = denormalize_image(
    sample_image
)


plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)

plt.imshow(
    original_image
)

plt.title(
    f"Original\nTrue: "
    f"{'Glaucoma' if sample_label == 1 else 'Normal'}"
)

plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    original_image
)

plt.imshow(
    cam,
    cmap="jet",
    alpha=0.45
)

plt.title(
    f"Grad-CAM\nPredicted: "
    f"{'Glaucoma' if predicted_class == 1 else 'Normal'}"
)

plt.axis("off")

plt.tight_layout()

plt.show()


# ============================================================
# CORRECTED VLM PART
# Pretrained BLIP + Grad-CAM Highlighted Image
# NO FINE-TUNING
# ============================================================

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD PRETRAINED BLIP
# ============================================================

VLM_MODEL_NAME = "Salesforce/blip-image-captioning-base"

print("Loading pretrained VLM...")

processor = BlipProcessor.from_pretrained(
    VLM_MODEL_NAME
)

vlm_model = BlipForConditionalGeneration.from_pretrained(
    VLM_MODEL_NAME
)

vlm_model = vlm_model.to(DEVICE)

vlm_model.eval()

print("VLM loaded successfully.")
print("Fine-tuning: NO")


# ============================================================
# 2. CREATE GRAD-CAM OVERLAY IMAGE
# ============================================================

def create_gradcam_overlay(
    image_tensor,
    cam
):

    # Convert normalized tensor back to image
    image = image_tensor.detach().cpu().numpy()

    image = np.transpose(
        image,
        (1, 2, 0)
    )

    mean = np.array([
        0.485,
        0.456,
        0.406
    ])

    std = np.array([
        0.229,
        0.224,
        0.225
    ])

    image = (
        image * std
    ) + mean

    image = np.clip(
        image,
        0,
        1
    )

    # Convert CAM to 0-255
    cam_uint8 = np.uint8(
        255 * cam
    )

    # Resize CAM
    cam_img = Image.fromarray(
        cam_uint8
    ).resize(
        (image.shape[1], image.shape[0])
    )

    cam_resized = np.array(
        cam_img
    ) / 255.0

    # Create jet heatmap
    heatmap = plt.get_cmap("jet")(
        cam_resized
    )[:, :, :3]

    # Overlay
    overlay = (
        0.55 * image +
        0.45 * heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    overlay_uint8 = np.uint8(
        overlay * 255
    )

    return Image.fromarray(
        overlay_uint8
    )


# ============================================================
# 3. VLM EXPLANATION FUNCTION
# ============================================================

def generate_vlm_explanation(
    image,
    predicted_class,
    probability,
    cam=None
):

    # --------------------------------------------------------
    # If Grad-CAM is available, give VLM the highlighted image
    # --------------------------------------------------------

    if cam is not None:

        image_for_vlm = create_gradcam_overlay(
            image,
            cam
        )

    else:

        # Convert tensor to PIL image
        image_np = image.detach().cpu().numpy()

        image_np = np.transpose(
            image_np,
            (1, 2, 0)
        )

        mean = np.array([
            0.485,
            0.456,
            0.406
        ])

        std = np.array([
            0.229,
            0.224,
            0.225
        ])

        image_np = (
            image_np * std
        ) + mean

        image_np = np.clip(
            image_np,
            0,
            1
        )

        image_for_vlm = Image.fromarray(
            np.uint8(image_np * 255)
        )


    # --------------------------------------------------------
    # Prediction text
    # --------------------------------------------------------

    if predicted_class == 1:

        prediction_text = (
            "The computer vision model predicted "
            "GLAUCOMA."
        )

    else:

        prediction_text = (
            "The computer vision model predicted "
            "NORMAL."
        )


    # --------------------------------------------------------
    # VLM prompt
    # --------------------------------------------------------
    #
    # We tell the VLM:
    # - image is retinal fundus
    # - Grad-CAM highlighted the model-attended region
    # - describe visible evidence
    # - don't invent measurements
    #
    # --------------------------------------------------------

    prompt = f"""
A retinal fundus photograph is shown with a heatmap
highlighting the region used by an image classification model.

{prediction_text}

Describe the visible retinal features in the highlighted
region that may be relevant to the classification.

Focus on:
- optic disc appearance
- optic cup appearance if clearly visible
- retinal structure
- visible suspicious or non-suspicious characteristics

Do not invent numerical measurements.
Do not claim a medical diagnosis from the image alone.
Describe only visually observable characteristics.
"""


    # --------------------------------------------------------
    # Process image + prompt
    # --------------------------------------------------------

    inputs = processor(
        images=image_for_vlm,
        text=prompt,
        return_tensors="pt"
    )


    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }


    # --------------------------------------------------------
    # Generate explanation
    # --------------------------------------------------------

    with torch.no_grad():

        output = vlm_model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=5,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3
        )


    explanation = processor.decode(
        output[0],
        skip_special_tokens=True
    )


    # --------------------------------------------------------
    # Remove prompt from output if BLIP repeats it
    # --------------------------------------------------------

    explanation = explanation.strip()


    return explanation


# ============================================================
# 4. TEST VLM WITH YOUR EXISTING TEST IMAGE
# ============================================================

sample_image, sample_label = test_dataset[0]

sample_path = test_df.iloc[0]["image_path"]


# ------------------------------------------------------------
# Get model prediction
# ------------------------------------------------------------

model.eval()

with torch.no_grad():

    input_tensor = sample_image.unsqueeze(
        0
    ).to(DEVICE)

    logits = model(
        input_tensor
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    glaucoma_probability = probabilities[
        0, 1
    ].item()


# ============================================================
# 5. GENERATE GRAD-CAM
# ============================================================

cam, predicted_class = gradcam.generate(
    sample_image.unsqueeze(0),
    target_class=predicted_class
)


# ============================================================
# 6. GENERATE VLM EXPLANATION
# ============================================================

vlm_explanation = generate_vlm_explanation(
    image=sample_image,
    predicted_class=predicted_class,
    probability=glaucoma_probability,
    cam=cam
)


# ============================================================
# 7. DISPLAY FINAL RESULT
# ============================================================

print("\n")
print("=" * 60)
print("MODEL + GRAD-CAM + VLM RESULT")
print("=" * 60)

print(
    "\nTrue Class:",
    "GLAUCOMA" if sample_label == 1 else "NORMAL"
)

print(
    "Predicted Class:",
    "GLAUCOMA" if predicted_class == 1 else "NORMAL"
)

print(
    f"Glaucoma Probability: "
    f"{glaucoma_probability:.4f}"
)

print("\nVLM Visual Explanation:")
print(vlm_explanation)


# ============================================================
# 8. DISPLAY IMAGE + GRAD-CAM
# ============================================================

overlay_image = create_gradcam_overlay(
    sample_image,
    cam
)


plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)

plt.imshow(
    denormalize_image(sample_image)
)

plt.title(
    "Original Fundus Image"
)

plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    overlay_image
)

plt.title(
    "Grad-CAM Highlighted Image"
)

plt.axis("off")


plt.tight_layout()

plt.show()




# ============================================================
# 43. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("EXPERIMENT COMPLETED")
print("=" * 60)

print(
    f"ACRIMA images     : {len(acrima_df)}"
)

print(
    f"RIM-ONE images    : {len(rim_df)}"
)

print(
    f"Combined images   : {len(combined_df)}"
)

print(
    f"Training images   : {len(train_df)}"
)

print(
    f"Validation images : {len(val_df)}"
)

print(
    f"Testing images    : {len(test_df)}"
)

print("\nModel:")
print(
    "ConvNeXt-Tiny + Multi-Level Feature Fusion"
)

print("\nAugmentation:")
print(
    "YES - Training only"
)

print(
    "NO - Validation/Test"
)

print("\nExplainability:")
print(
    "Grad-CAM"
)

print("\nVLM:")
print(
    "BLIP image-captioning-base"
)

print(
    "Fine-tuning: NO"
)

print("\nBest checkpoint:")
print(
    CHECKPOINT_PATH
)

print("\nResults:")
print(
    RESULTS_DIR
)

print("\nDone.")
     
