from pathlib import Path
from typing import Callable
import torch
import torch.optim as optim
from pydantic import BaseModel
from torch.utils.data import DataLoader
from mltrainer import ReportTypes, Trainer, TrainerSettings

from vectormesh import RegexVectorizer
from vectormesh.components import (
    MeanAggregator,
    AttentionAggregator,
    Concatenate2D,
    FixedPadding,
    NeuralNet,
    Parallel,
    Projection,
    Serial,
    Skip,
)
from vectormesh.components.metrics import F1Score
from vectormesh.data.cache import VectorCache
from vectormesh.data.vectorizers import (
    build_legal_reference_pattern,
    harmonize_legal_reference,
)

artefacts = Path("../artefacts")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

PREFIX = "hypothese6"

trainpath_d = next(artefacts.glob("*legal_dutch*train/"))
validpath_d = next(artefacts.glob("*legal_dutch*valid/"))
traincache_d = VectorCache.load(path=trainpath_d)
validcache_d = VectorCache.load(path=validpath_d)

trainpath_b = next(artefacts.glob("*legal_bert*train/"))
validpath_b = next(artefacts.glob("*legal_bert*valid/"))
traincache_b = VectorCache.load(path=trainpath_b)
validcache_b = VectorCache.load(path=validpath_b)

print("Loaded:", trainpath_d.name, validpath_d.name, trainpath_b.name, validpath_b.name)

def detect_embedding_col(cache: VectorCache) -> str:
    sample = next(iter(cache.dataset))
    candidates = []
    for k, v in sample.items():
        if hasattr(v, "shape") and len(v.shape) == 2 and v.shape[-1] == 768:
            candidates.append(k)
    if not candidates:
        raise RuntimeError(f"No (chunks,768) embedding column found. keys={list(sample.keys())}")
    if len(candidates) > 1:
        print("Warning: multiple embedding cols found:", candidates, "-> using", candidates[0])
    return candidates[0]


EMB_COL_D = detect_embedding_col(traincache_d)
EMB_COL_B = detect_embedding_col(traincache_b)
print("Detected embedding cols:", EMB_COL_D, "(legal_dutch)", "|", EMB_COL_B, "(legal_bert)")


def make_settings(run_name: str, trainloader, validloader, *, prefix: str = PREFIX):
    run_id = f"{prefix}_{run_name}"
    return TrainerSettings(
        epochs=50,
        metrics=[
            F1Score(average="micro"),
            F1Score(average="macro"),
        ],
        logdir=Path("demo") / run_id,
        train_steps=len(trainloader),
        valid_steps=len(validloader),
        reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.TOML],
    )


class OneHot(BaseModel):
    num_classes: int
    label_col: str
    target_col: str

    def __call__(self, observation):
        raw = observation[self.label_col]
        label = int(raw.view(-1)[0].item()) if torch.is_tensor(raw) else int(raw)
        vec = torch.zeros(self.num_classes, dtype=torch.float32)
        vec[label] = 1.0
        return {self.target_col: vec}


onehot = OneHot(num_classes=32, label_col="labels", target_col="onehot")

class Combined2BertDataset(torch.utils.data.Dataset):
    """
    Combineert:
    - embedding uit cache A (legal_dutch)
    - embedding uit cache B (legal_bert)
    - plus overige velden (text/labels/regex/onehot etc) uit A
    """
    def __init__(self, ds_a, ds_b, emb_col_a: str, emb_col_b: str,
                 out_col_a: str = "emb_dutch", out_col_b: str = "emb_bert"):
        self.ds_a = ds_a
        self.ds_b = ds_b
        self.emb_col_a = emb_col_a
        self.emb_col_b = emb_col_b
        self.out_col_a = out_col_a
        self.out_col_b = out_col_b
        if len(ds_a) != len(ds_b):
            raise ValueError(f"Datasets differ in length: {len(ds_a)} vs {len(ds_b)}")

    def __len__(self):
        return len(self.ds_a)

    def __getitem__(self, idx):
        a = self.ds_a[idx]
        b = self.ds_b[idx]

        # sanity check
        if "labels" in a and "labels" in b:
            la = a["labels"]
            lb = b["labels"]
            la = int(la.view(-1)[0].item()) if torch.is_tensor(la) else int(la)
            lb = int(lb.view(-1)[0].item()) if torch.is_tensor(lb) else int(lb)
            if la != lb:
                raise ValueError(f"Label mismatch at idx={idx}: {la} vs {lb}")

        out = dict(a)
        out[self.out_col_a] = a[self.emb_col_a]  # (chunks,768)
        out[self.out_col_b] = b[self.emb_col_b]  # (chunks,768)
        return out


regexvectorizer = RegexVectorizer(
    col_name="regex",
    pattern_builder=build_legal_reference_pattern,
    harmonizer=harmonize_legal_reference,
    min_doc_frequency=15,
    max_features=200,
    device="cpu",
    training_texts=traincache_d["text"],
)

extended_traincache_d = VectorCache.create(
    cache_dir=Path("tmp/artefacts"),
    vectorizer=regexvectorizer,
    dataset=traincache_d.dataset,
    dataset_tag=trainpath_d.name,
)
extended_validcache_d = VectorCache.create(
    cache_dir=Path("tmp/artefacts"),
    vectorizer=regexvectorizer,
    dataset=validcache_d.dataset,
    dataset_tag=validpath_d.name,
)


train_d_oh = extended_traincache_d.map(onehot)
valid_d_oh = extended_validcache_d.map(onehot)

train_h = Combined2BertDataset(
    train_d_oh, traincache_b.dataset,
    EMB_COL_D, EMB_COL_B,
    out_col_a="emb_dutch", out_col_b="emb_bert",
)
valid_h = Combined2BertDataset(
    valid_d_oh, validcache_b.dataset,
    EMB_COL_D, EMB_COL_B,
    out_col_a="emb_dutch", out_col_b="emb_bert",
)

class Collate2BertRegex(BaseModel):
    vec_dutch: str
    vec_bert: str
    vec_regex: str
    target_col: str
    padder: Callable

    def __call__(self, batch):
        emb1 = [item[self.vec_dutch] for item in batch]
        emb2 = [item[self.vec_bert] for item in batch]
        reg  = [item[self.vec_regex] for item in batch]

        X1 = self.padder(emb1)              # (B,30,768)
        X2 = self.padder(emb2)              # (B,30,768)
        X3 = torch.stack(reg).float()       # (B,REGEX_DIM)
        y  = torch.stack([item[self.target_col] for item in batch]).float()
        return (X1, X2, X3), y


padder = FixedPadding(max_chunks=30)
collate_fn = Collate2BertRegex(
    vec_dutch="emb_dutch",
    vec_bert="emb_bert",
    vec_regex="regex",
    target_col="onehot",
    padder=padder,
)

trainloader = DataLoader(train_h, batch_size=32, shuffle=True, collate_fn=collate_fn)
validloader = DataLoader(valid_h, batch_size=32, shuffle=False, collate_fn=collate_fn)


settings = make_settings("run2_mean_skip_2bert_plus_regex", trainloader, validloader)
loss_fn = torch.nn.BCEWithLogitsLoss()

(_, _, X3_tmp), _ = next(iter(trainloader))
REGEX_DIM = int(X3_tmp.shape[-1])
print("Detected REGEX_DIM:", REGEX_DIM)


parallel = Parallel([
    Serial([MeanAggregator(), NeuralNet(hidden_size=768, out_size=32)]),
    Serial([MeanAggregator(), NeuralNet(hidden_size=768, out_size=32)]),
    Serial([NeuralNet(hidden_size=REGEX_DIM, out_size=32)]),
])


pipeline = Serial([
    parallel,
    Concatenate2D(),
    Projection(hidden_size=96, out_size=32),
    Skip(transform=NeuralNet(hidden_size=32, out_size=32), in_size=32),
]).to(device)

print("pipeline params on:", next(pipeline.parameters()).device)


(X1, X2, X3), y = next(iter(trainloader))
X = (X1.to(device), X2.to(device), X3.to(device))
with torch.no_grad():
    yhat = pipeline(X)
print("sanity yhat:", yhat.shape)

trainer = Trainer(
    model=pipeline,
    settings=settings,
    loss_fn=loss_fn,
    optimizer=optim.Adam,
    traindataloader=trainloader,
    validdataloader=validloader,
    scheduler=optim.lr_scheduler.ReduceLROnPlateau,
    device=device,
)

trainer.loop()
print("================== EINDE hypothese6 attn+skip+2bert+regex ==================")
