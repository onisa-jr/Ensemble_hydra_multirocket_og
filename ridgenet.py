import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_prepare_data(dataset="Car", resample=1, batch_size=64):
    """
    Load Hydra + MultiRocket features and prepare PyTorch dataloaders
    Combines train+test, shuffles by seed (resample), then splits back
    """
    # Load pretrained transformers
    model = joblib.load(f"baseline_models_resamples/resample_{resample}/{dataset}/clf.pkl")
    hydra = model["hydra"]
    multirocket = model["multirocket"]
    scaler_hydra = model["scaler_hydra"]
    scaler_std = model["scaler_std"]

    # Load raw time series data
    train_df = pd.read_csv(f"datasets/{dataset}/{dataset}_TRAIN.tsv", sep="\t", header=None)
    test_df = pd.read_csv(f"datasets/{dataset}/{dataset}_TEST.tsv", sep="\t", header=None)

    # Store original split sizes
    n_train_original = len(train_df)
    n_test_original = len(test_df)

    # Extract labels and features
    y_train, X_train = train_df.iloc[:, 0].values, train_df.iloc[:, 1:].values.reshape(train_df.shape[0], 1, -1)
    y_test, X_test = test_df.iloc[:, 0].values, test_df.iloc[:, 1:].values.reshape(test_df.shape[0], 1, -1)

    # Transform features using saved transformers
    Xt_hydra_train = scaler_hydra.transform(hydra.transform(X_train))
    Xt_multi_train = scaler_std.transform(multirocket.transform(X_train))
    Xt_hydra_test = scaler_hydra.transform(hydra.transform(X_test))
    Xt_multi_test = scaler_std.transform(multirocket.transform(X_test))

    # Concatenate features (Hydra + MultiRocket)
    X_train_features = np.concatenate([Xt_hydra_train, Xt_multi_train], axis=1)
    X_test_features = np.concatenate([Xt_hydra_test, Xt_multi_test], axis=1)

    # Combine train and test
    X_combined = np.concatenate([X_train_features, X_test_features], axis=0)
    y_combined = np.concatenate([y_train, y_test], axis=0)

    # Shuffle combined data with seed=resample
    np.random.seed(resample)
    indices = np.random.permutation(len(X_combined))
    X_combined = X_combined[indices]
    y_combined = y_combined[indices]

    # Split back to original sizes
    X_train_features = X_combined[:n_train_original]
    X_test_features = X_combined[n_train_original:]
    y_train = y_combined[:n_train_original]
    y_test = y_combined[n_train_original:]

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_features, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_enc, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_features, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_enc, dtype=torch.long)

    # Create datasets and dataloaders
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    test_ds = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Get data info
    input_dim = X_train_features.shape[1]
    n_classes = len(np.unique(y_train))

    print(f"Dataset: {dataset}")
    print(f"Input dimension: {input_dim:,}")
    print(f"Number of classes: {n_classes}")
    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples: {len(test_ds)}")
    print(f"Shuffle seed: {resample}")

    return train_loader, test_loader, input_dim, n_classes, le


class HybridRidgeNet(nn.Module):
    """
    Advanced classifier inspired by RidgeClassifier strengths:
    - Ridge regression's L2 regularization (fast, stable)
    - Non-linear refinement (what Ridge misses)
    - Multi-scale features (what Ridge misses)
    - Probability calibration (what Ridge misses)
    """

    def __init__(self, input_dim, n_classes,
                 alpha=1.0,  # Ridge regularization strength
                 hidden_dims=[1024, 512, 256],
                 dropout=0.3):
        super().__init__()

        self.alpha = alpha
        self.n_classes = n_classes

        # Stage 1: Ridge-inspired linear compression with strong L2
        self.ridge_projection = LinearWithL2(
            input_dim,
            hidden_dims[0],
            alpha=alpha * 2.0
        )
        self.proj_bn = nn.BatchNorm1d(hidden_dims[0])

        # Stage 2: Multi-scale feature extraction
        self.multiscale = MultiScaleBlock(hidden_dims[0], hidden_dims[1], alpha=alpha)

        # Stage 3: Non-linear refinement blocks
        self.refinement_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 2):
            self.refinement_blocks.append(
                RidgeRefinementBlock(
                    hidden_dims[i + 1],
                    hidden_dims[i + 2],
                    alpha=alpha,
                    dropout=dropout
                )
            )

        # Stage 4: Class probability calibration
        self.calibration = ProbabilityCalibration(hidden_dims[-1], n_classes, alpha=alpha)

        # Learnable class weights
        self.class_weights = nn.Parameter(torch.ones(n_classes))

        self.apply(self._ridge_init)

    def _ridge_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.ridge_projection(x)
        x = self.proj_bn(x)
        x = F.relu(x)
        x = self.multiscale(x)
        for block in self.refinement_blocks:
            x = block(x)
        x = self.calibration(x)
        x = x * self.class_weights.unsqueeze(0)
        return x

    def get_l2_loss(self):
        l2_loss = 0.0
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_loss += torch.sum(param ** 2)
        return self.alpha * l2_loss


class LinearWithL2(nn.Module):
    def __init__(self, in_features, out_features, alpha=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.alpha = alpha

    def forward(self, x):
        return self.linear(x)


class MultiScaleBlock(nn.Module):
    def __init__(self, in_dim, out_dim, alpha=1.0):
        super().__init__()

        # Ensure dimensions sum exactly to out_dim
        dim1 = out_dim // 3
        dim2 = out_dim // 3
        dim3 = out_dim - dim1 - dim2  # Remainder goes here

        self.path1 = LinearWithL2(in_dim, dim1, alpha=alpha)
        self.path2 = nn.Sequential(
            LinearWithL2(in_dim, in_dim // 2, alpha=alpha),
            nn.ReLU(),
            LinearWithL2(in_dim // 2, dim2, alpha=alpha)
        )
        self.path3 = nn.Sequential(
            LinearWithL2(in_dim, in_dim // 2, alpha=alpha),
            nn.ReLU(),
            LinearWithL2(in_dim // 2, in_dim // 4, alpha=alpha),
            nn.ReLU(),
            LinearWithL2(in_dim // 4, dim3, alpha=alpha)
        )

        self.fusion = nn.Sequential(
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.fusion(out)
        return out


class RidgeRefinementBlock(nn.Module):
    def __init__(self, in_dim, out_dim, alpha=1.0, dropout=0.3):
        super().__init__()

        self.main = nn.Sequential(
            LinearWithL2(in_dim, out_dim, alpha=alpha),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            LinearWithL2(out_dim, out_dim, alpha=alpha * 0.5),
            nn.BatchNorm1d(out_dim)
        )

        self.skip = LinearWithL2(in_dim, out_dim, alpha=alpha) if in_dim != out_dim else nn.Identity()

        self.gate = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.Sigmoid()
        )

        self.activation = nn.ReLU()

    def forward(self, x):
        identity = self.skip(x)
        out = self.main(x)
        gate = self.gate(out)
        out = gate * out + (1 - gate) * identity
        out = self.activation(out)
        return out


class ProbabilityCalibration(nn.Module):
    def __init__(self, in_dim, n_classes, alpha=1.0):
        super().__init__()

        self.pre_logits = nn.Sequential(
            LinearWithL2(in_dim, in_dim // 2, alpha=alpha),
            nn.BatchNorm1d(in_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = LinearWithL2(in_dim // 2, n_classes, alpha=alpha * 0.5)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        x = self.pre_logits(x)
        logits = self.classifier(x)
        calibrated_logits = logits / self.temperature
        return calibrated_logits


class RidgeLoss(nn.Module):
    def __init__(self, alpha=1.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, pred, target, model):
        ce = self.ce_loss(pred, target)
        l2_penalty = model.get_l2_loss()
        total_loss = ce + l2_penalty
        return total_loss, ce, l2_penalty


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss, total_ce, total_l2 = 0, 0, 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss, ce_loss, l2_loss = criterion(outputs, y_batch, model)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_l2 += l2_loss.item()

        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    return {
        'loss': total_loss / len(train_loader),
        'ce_loss': total_ce / len(train_loader),
        'l2_loss': total_l2 / len(train_loader),
        'accuracy': 100 * correct / total
    }


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    return 100 * correct / total


def train_fld(DATASET="Haptics",
              RESAMPLE=1,
              BATCH_SIZE=64,
              EPOCHS=150,
              LEARNING_RATE=0.001,
              ALPHA=0.5):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load data
    train_loader, test_loader, input_dim, n_classes, label_encoder = load_and_prepare_data(
        dataset=DATASET,
        resample=RESAMPLE,
        batch_size=BATCH_SIZE
    )

    # Create model
    model = HybridRidgeNet(
        input_dim=input_dim,
        n_classes=n_classes,
        alpha=ALPHA,
        hidden_dims=[1024, 512, 256],
        dropout=0.3
    ).to(device)

    # Training setup
    criterion = RidgeLoss(alpha=ALPHA, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Print model info
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)

    # Training loop
    best_test_acc = 0.0
    for epoch in range(EPOCHS):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # torch.save(model.state_dict(), f'best_model_{DATASET}.pth')

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}]")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Test Acc: {test_acc:.2f}% | Best: {best_test_acc:.2f}%")
            print(f"  CE: {train_metrics['ce_loss']:.4f} | "
                  f"L2: {train_metrics['l2_loss']:.4f}")
            print("-" * 60)

    print(f"\n{'=' * 60}")
    print(f"Training Complete!")
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"{'=' * 60}")
    return best_test_acc
