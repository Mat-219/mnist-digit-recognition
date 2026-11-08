"""Trains all models from src/models.py, saves results table and plots to results/.

Usage (from the repo root):
    python -m src.train
    EPOCHS=3 python -m src.train      # quick run
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.data import load_mnist
from src.models import EXPERIMENTS

tf.random.set_seed(0)
np.random.seed(0)

EPOCHS = int(os.environ.get("EPOCHS", 10))
BATCH = 100
OUT = "results"
os.makedirs(OUT, exist_ok=True)

(x_train, y_train), (x_test, y_test) = load_mnist()
print("train:", x_train.shape, "test:", x_test.shape)

results, histories, models = {}, {}, {}
for name, (build, make_opt) in EXPERIMENTS.items():
    print(f"\n=== {name} ===")
    model = build()
    model.compile(optimizer=make_opt(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    t0 = time.time()
    h = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH,
                  validation_data=(x_test, y_test), verbose=2)
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    results[name] = {
        "params": int(model.count_params()),
        "test_acc": round(float(acc), 4),
        "max_test_acc": round(float(max(h.history["val_accuracy"])), 4),
        "time_s": round(time.time() - t0, 1),
    }
    histories[name], models[name] = h.history, model

# --- results table ---
print(f"\n{'model':<26}{'params':>9}{'test_acc':>10}{'max_acc':>9}{'time[s]':>9}")
for n, r in results.items():
    print(f"{n:<26}{r['params']:>9}{r['test_acc']:>10}{r['max_test_acc']:>9}{r['time_s']:>9}")
with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=2)

ep = range(1, EPOCHS + 1)

# --- plot 1: test accuracy / loss comparison ---
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for n, h in histories.items():
    ax[0].plot(ep, h["val_accuracy"], marker="o", ms=3, label=n)
    ax[1].plot(ep, h["val_loss"], marker="o", ms=3, label=n)
ax[0].set_title("Test accuracy"); ax[0].set_ylim(0.9, 1.0)
ax[1].set_title("Test loss (cross-entropy)")
for a in ax:
    a.set_xlabel("epoch"); a.grid(alpha=0.3); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/1_comparison.png", dpi=140); plt.close()

# --- plot 2: overfitting without / with dropout ---
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
for a, n in zip(ax, ["2_dense5_relu", "3_dense5_relu_dropout"]):
    a.plot(ep, histories[n]["loss"], label="train loss")
    a.plot(ep, histories[n]["val_loss"], label="test loss")
    a.set_title(n); a.set_xlabel("epoch"); a.grid(alpha=0.3); a.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/2_overfitting.png", dpi=140); plt.close()

# --- plot 3: misclassified digits of the best model ---
best = max(results, key=lambda n: results[n]["test_acc"])
pred = np.argmax(models[best].predict(x_test, verbose=0), axis=1)
wrong = np.where(pred != y_test)[0]
fig = plt.figure(figsize=(10, 10))
for k, idx in enumerate(wrong[:100]):
    a = fig.add_subplot(10, 10, k + 1)
    a.imshow(x_test[idx, :, :, 0], cmap="gray_r")
    a.set_title(f"{y_test[idx]}→{pred[idx]}", fontsize=8, color="red")
    a.axis("off")
fig.suptitle(f"{best}: {len(wrong)} errors / 10000 (true→predicted)")
plt.tight_layout(); plt.savefig(f"{OUT}/3_misclassified.png", dpi=120); plt.close()

# --- plot 4: confusion matrix (errors only) ---
cm = tf.math.confusion_matrix(y_test, pred).numpy()
np.fill_diagonal(cm, 0)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Reds"); plt.colorbar()
for i in range(10):
    for j in range(10):
        if cm[i, j]:
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
plt.xticks(range(10)); plt.yticks(range(10))
plt.xlabel("predicted"); plt.ylabel("true"); plt.title(f"Errors only – {best}")
plt.tight_layout(); plt.savefig(f"{OUT}/4_confusion_errors.png", dpi=140); plt.close()

models[best].save(f"{OUT}/best_model.keras")
print(f"\nBest: {best}. Plots and results.json saved in {OUT}/")