# 04 · Image models

## The decision table

| Situation | Approach | Takes |
|---|---|---|
| **Count, measure, or locate simple shapes** | OpenCV contours — **no ML at all** | 15 min |
| Classify images, small dataset | pretrained CNN features + `LogisticRegression` | 30 min |
| Find similar or matching images | pretrained CNN embeddings + cosine | 30 min |
| Classify, larger dataset, GPU available | fine-tune the last layers of a pretrained CNN | 1 h+ |
| Images already converted to numeric features | treat as a table — see [02](./02_tabular_models.md) | 10 min |
| Train a CNN from scratch | **don't** | — |

## 1. Classical OpenCV — try this before any model

If the task is counting objects, measuring them, or finding shapes with clean edges, image processing gives an *exact* answer where a model gives an estimate.

```python
import cv2, numpy as np

img  = cv2.imread("blueprint.png")          # note: BGR order, not RGB
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Background is the most common grey level — do not assume it is white
bg = int(np.bincount(gray.ravel()).argmax())
mask = (np.abs(gray.astype(np.int16) - bg) > 20).astype(np.uint8) * 255

mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

# RETR_EXTERNAL = outermost outlines only.
# RETR_LIST also returns the inner edge of a hollow shape and DOUBLES your count.
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
count = sum(1 for c in contours if cv2.contourArea(c) >= 20)
```

Telling shapes apart, once you have contours:

```python
for c in contours:
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    verts = len(cv2.approxPolyDP(c, 0.04 * peri, True))
    circularity = 4 * np.pi * area / (peri ** 2 + 1e-9)   # 1.0 = perfect circle
    x, y, w, h = cv2.boundingRect(c)
    aspect = w / h

    if circularity > 0.8:  shape = "circle"
    elif verts == 3:       shape = "triangle"
    elif verts == 4:       shape = "rectangle"
    else:                  shape = "other"
```

Docs: [contours tutorial](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html) · [thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html) · [morphology](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)

## 2. Pretrained CNN as a feature extractor — the ML default

Do not train a network. Use one someone else trained, take the numbers from its second-to-last layer, and put an ordinary classifier on top.

```python
import torch, numpy as np
from torchvision import models, transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression

device = "cuda" if torch.cuda.is_available() else "cpu"
net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
net.fc = torch.nn.Identity()        # chop off the classifier, keep the features
net.eval().to(device)

prep = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@torch.no_grad()
def embed(paths, batch=32):
    out = []
    for i in range(0, len(paths), batch):
        imgs = [prep(Image.open(p).convert("RGB")) for p in paths[i:i+batch]]
        out.append(net(torch.stack(imgs).to(device)).cpu().numpy())
    return np.vstack(out)

Xtr, Xte = embed(train_paths), embed(test_paths)
clf = LogisticRegression(max_iter=2000).fit(Xtr, y_train)
```

**Why this and not fine-tuning:** no training loop, no learning rate, no epochs, no overfitting to debug. Ten minutes of code and it is usually within a few points of a fine-tuned model on small data.

`resnet18` is small and fast. For better features use `resnet50` or a `timm` model — same code, one line changed.

Docs: [torchvision models](https://pytorch.org/vision/stable/models.html) · [timm](https://huggingface.co/docs/timm/quickstart)

## 3. Image similarity and retrieval

```python
emb = embed(all_paths)
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)   # L2 normalise

sims = emb @ emb[query_idx]        # normalised dot product == cosine similarity
sims[query_idx] = -np.inf          # never match the query with itself
top3 = np.argsort(-sims)[:3]
```

**Use when:** matching a photo to a catalogue, deduplication, "find the same object".
**Always L2-normalise** before comparing, or vector length drowns out direction.

If the query images look different from the gallery — blurry phone photos against clean studio shots — apply augmentation that mimics that damage when building the gallery embeddings.

## 4. Handcrafted image features — the tabular route

When images are simple, turn each into a row of numbers and use gradient boosting.

```python
def image_features(path):
    g = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([g], [0], None, [16], [0, 256]).ravel() / g.size
    m = cv2.moments(g)
    hu = cv2.HuMoments(m).ravel()                       # rotation/scale invariant
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)    # compress the range
    return np.concatenate([[g.mean(), g.std(), g.min(), g.max()], hist, hu])
```

**Use when:** shapes or textures matter more than semantic content, or no GPU is available.
**Hu moments** stay the same when the image is moved, scaled or rotated, which makes them strong for symbols and letters.

Docs: [image moments](https://docs.opencv.org/4.x/d8/d23/classcv_1_1Moments.html)

## 5. Light fine-tuning — only if you have time and a GPU

```python
for p in net.parameters():        # freeze everything
    p.requires_grad = False
net.fc = torch.nn.Linear(512, n_classes)   # train only this new layer
```

**Use when:** several thousand labelled images, a GPU, and an hour to spare.
**Skip when:** anything above is true but not all of them. The feature-extraction route gets most of the benefit at a fraction of the risk.

Article: [PyTorch transfer learning tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

## Augmentation, if you do train

```python
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])
```

Choose augmentations that imitate the damage you expect at test time. Do not flip digits or letters — that changes their identity.

## Speed on many images

```python
gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)    # skip colour if unused
```

Time ten images, multiply by the total, and decide before launching. At 0.4 s per image, 8000 images take 53 minutes.
