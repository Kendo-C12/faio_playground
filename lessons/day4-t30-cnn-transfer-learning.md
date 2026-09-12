# T30 · CNNs and transfer learning — Day 4

**Anchor task(s):**
- [`faio-2025/day2/Lost_in_the_Museum.md`](../faio-2025/day2/Lost_in_the_Museum.md)
- [`faio-2025/qualification/task6_Simple_Objects.md`](../faio-2025/qualification/task6_Simple_Objects.md)

Rating 1 · **exam-probability rank 2** · ~50 min

Two anchors on purpose: one task where a pretrained network is the only route, and one where reaching for a network loses to twenty lines of OpenCV.

## [concept-first]

**Convolution.** Slide a small learnable kernel (3×3, say) over the image; at each position output the weighted sum of the patch. One kernel produces one *feature map*. Three properties follow, and they are the whole reason CNNs beat dense layers on pixels: weights are **shared** across positions (so a 3×3 kernel is 9 parameters regardless of image size), the response is **translation-equivariant** (an edge detected at the top-left is detected the same way at the bottom-right), and each output sees only a local patch.

**Pooling / stride.** `MaxPool2d(2)` halves height and width by taking the max of each 2×2 block. Stacking conv + pool grows the **receptive field**: layer 1 sees 3×3 pixels, layer 5 sees a large region. So early layers learn edges and colour blobs, middle layers textures and motifs, late layers object-level structure. That hierarchy is what you are borrowing when you use someone else's weights.

**A backbone is the network minus its classifier head.** ResNet-50 ends with a global average pool producing a 2048-dim vector, then a 1000-way linear layer for ImageNet classes. Drop the linear layer and the **penultimate** 2048-dim vector is a general-purpose visual description of the image — that vector is the embedding of T22. You get it for free, with no labels and no training.

**Freeze vs fine-tune.**

- *Frozen backbone, inference only.* Forward pass, take the penultimate vector. Zero training, minutes of compute, no overfitting risk. The correct first move always.
- *Linear probe.* Freeze the backbone, train only a new head on your labels. Cheap, needs few labels.
- *Fine-tune the last block.* Unfreeze the final stage at a low learning rate (1e-4 or below) while earlier layers stay frozen. Useful when your domain is far from ImageNet.
- *Full fine-tune / train from scratch.* Needs many labels, many GPU-hours, and careful regularisation. In a timed round this is almost never the right call.

**Augmentation is a hypothesis about the test set.** Augmenting with a transform teaches the network to produce the same answer with and without it — so choose transforms that *mimic the corruption you expect at test time*, not whatever the tutorial listed. Flipping an image horizontally is wrong if left-right matters; a motion blur is right if your test images are handheld photos.

**The hardware constraint.** The IOAI 2026 individual contest runs each task on one ~16 GB GPU, with **no internet** and 5 GB of storage. Consequences to internalise now: weights must be downloaded *before* the round and kept under the storage budget; `timm.create_model(..., pretrained=True)` will fail offline, so load from a local checkpoint; 16 GB means a ResNet-50 or a ViT-B at batch 32 is comfortable and a from-scratch training run is not. Plan for **inference or light fine-tuning**, never training from scratch.

## [problem-first]

Open [`Lost_in_the_Museum.md`](../faio-2025/day2/Lost_in_the_Museum.md). 20,000 images: 10,000 studio-quality paintings, 1,000 visitor photos "shadowed, zoomed-in, poorly lit" with "motion blur, glare, cropping, low resolution", and 9,000 distractors. You are not told which is which, and you submit only embeddings.

Derive the requirements:

1. Matching is by cosine similarity between a photo and its painting, so the backbone must map two *very* different-looking images of the same artwork close together. This is a **domain gap** problem, not a classification problem.
2. There are no labels at all and no train/test split you control, so training a classifier is not even available. A frozen pretrained backbone is the baseline, and the work goes into *which* backbone and *how* you pre-process.
3. The corruption list is given explicitly. So build augmentation that reproduces it — random resized crop, Gaussian/motion blur, downscale-then-upscale, colour jitter and brightness — and use it either for test-time augmentation (average the embeddings of several augmented views) or to fine-tune with a self-supervised objective that pulls augmented views of one image together.
4. Do **not** reach for an ImageNet classifier's logits. ImageNet classes describe objects; here the "object" is always "a painting". The penultimate feature vector is what carries the identity.
5. 9,000 distractors mean false neighbours are abundant, so a generic feature that keys on "oil painting texture" retrieves garbage. You need features keyed on *composition and content*.

Now read [`task6_Simple_Objects.md`](../faio-2025/qualification/task6_Simple_Objects.md) as the counter-example. Its Tips name OpenCV, and its Input section guarantees "each figure is a single connected component when considering edge pixels". A CNN could be trained to regress the count; it would be slower, need the training images, and would still be worse than `findContours`, because the guarantee makes the answer *exact* rather than learned. The judgement call is this: **when the statement hands you a deterministic guarantee, the right tool is the algorithm that uses it, not a network.** Apply that to 2026 — a synthetic, fully specified, countable image task is classical CV; a messy real-photo task is a pretrained backbone.

## [code-first]

```python
import torch, torch.nn as nn, numpy as np
from torchvision import models, transforms
from PIL import Image

dev = "cuda" if torch.cuda.is_available() else "cpu"

# Pre-download weights BEFORE the round: the contest GPU has no internet.
# Offline: models.resnet50(); m.load_state_dict(torch.load("resnet50.pth"))
m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
m.fc = nn.Identity()                 # drop the 1000-way head -> penultimate 2048-dim vector
m.eval().to(dev)                     # eval() matters: BatchNorm must use running stats
for p in m.parameters():
    p.requires_grad = False          # frozen: inference only, nothing to overfit

# Clean transform: ImageNet normalisation is part of the pretrained contract.
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
base = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                           transforms.ToTensor(), NORM])

# Augmentation chosen to MIMIC the stated corruption in Lost_in_the_Museum:
# motion blur, glare, cropping, low resolution.
aug = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),      # visitor crops the frame
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.08),              # glare and poor lighting
    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),             # motion blur
    transforms.Resize(112), transforms.Resize(224),           # low resolution round-trip
    transforms.ToTensor(), NORM,
])

@torch.no_grad()
def embed(path, tta=4):
    """Average the clean view with augmented views, then L2-normalise (T22)."""
    img = Image.open(path).convert("RGB")
    batch = torch.stack([base(img)] + [aug(img) for _ in range(tta)]).to(dev)
    v = m(batch).mean(0).cpu().numpy()
    return v / (np.linalg.norm(v) + 1e-12)

# Task 6 counter-example: no network, no training, and it is exact.
# import cv2; len(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
```

## [drill]

1. Why does a 3×3 conv kernel have the same parameter count on a 64×64 and a 2048×2048 image?
2. What exactly do you take from ResNet-50 to get an embedding, and how many dimensions is it?
3. You forget `m.eval()`. Name the layer type that misbehaves and what it does to your embeddings.
4. The contest GPU has no internet. Which one line of the snippet fails, and what replaces it?
5. `Lost_in_the_Museum` lists motion blur and cropping. Name the two matching augmentations and say why horizontal flip is a worse choice here.
6. Give one concrete signal, readable from a statement, that a CNN is the wrong tool.

<details><summary>Answers</summary>

1. Weights are shared across all spatial positions, so the kernel is 9 parameters (plus bias) per input/output channel pair regardless of resolution.
2. The penultimate activation — the global-average-pooled vector after the last conv stage, with `fc` replaced by `nn.Identity()`. 2048 dimensions.
3. BatchNorm. In train mode it normalises by *batch* statistics, so the same image embeds differently depending on which images share its batch — retrieval becomes noise. (Dropout also stays active.)
4. `models.resnet50(weights=...IMAGENET1K_V2)` tries to download. Replace with an unweighted constructor plus `load_state_dict(torch.load(local_path))`, with the checkpoint pre-downloaded inside the 5 GB budget.
5. `GaussianBlur` for motion blur and `RandomResizedCrop` for cropping. A visitor does not mirror the painting, so horizontal flip teaches invariance to something the test set never contains — it spends capacity and can make mirrored-composition paintings collide.
6. A deterministic guarantee in the statement — e.g. task 6's "each figure is a single connected component when considering edge pixels" — which makes a classical algorithm exact where a network is only approximate.

</details>

**Rep:** using `Lost_in_the_Museum`'s own corruption list, write the `aug` pipeline from memory, then embed one image twice (clean and augmented) and print the cosine similarity between the two vectors. A frozen backbone worth using should stay above roughly 0.8.

## Traps & 60-second recall

- Take the **penultimate** vector, not the logits. Logits describe ImageNet classes, not your images.
- `m.eval()` plus `torch.no_grad()`, always — BatchNorm in train mode makes embeddings batch-dependent.
- Keep the pretrained normalisation constants; changing them invalidates the weights.
- Frozen inference first, linear probe second, last-block fine-tune third, from scratch never.
- Pre-download weights: the IOAI 2026 individual contest GPU has no internet, ~16 GB VRAM and 5 GB storage.
- Choose augmentations that reproduce the stated test-time corruption; anything else is wasted capacity.
- L2-normalise before any cosine comparison (T22).
- If the statement gives an exact structural guarantee, classical CV beats any network — that is task 6.
