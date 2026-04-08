from pathlib import Path
from typing import Callable
import shutil
import torch
import torch.optim as optim
from pydantic import BaseModel
from torch.utils.data import DataLoader
from mltrainer import ReportTypes, Trainer, TrainerSettings
from vectormesh import RegexVectorizer
from vectormesh.components import (
    Concatenate2D,
    FixedPadding,
    MeanAggregator,
    MoE,
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



import torch

print(f"PyTorch versie: {torch.__version__}")
print(f"CUDA beschikbaar: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\n--- Diagnose ---")
    print(f"Aantal GPU's gevonden: {torch.cuda.device_count()}")
    try:
        # Dit dwingt een foutmelding af die ons precies vertelt waarom het niet werkt
        torch.cuda.init()
    except Exception as e:
        print(f"CUDA Init Foutmelding: {e}")

artefacts = Path("../artefacts")
trainpath = next(artefacts.glob("*bert*train/"))
validpath = next(artefacts.glob("*bert*valid/"))
traincache = VectorCache.load(path=trainpath)
validcache = VectorCache.load(path=validpath)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def make_settings(run_name: str, trainloader, validloader, *, prefix: str = "baseline"):
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


# ================== 1E NOTEBOOK BASELINE ==================

# range van 1024 of alles
# train = traincache.select(range(1024))
# valid = validcache.select(range(1024))
train = traincache.dataset
valid = validcache.dataset

traintag = trainpath.name
validtag = validpath.name

column_name = "legal_dutch"  # the vector we want to use

padder = FixedPadding(max_chunks=30)
batch = next(train.iter(batch_size=16))

emb_list = batch[column_name]
print([e.shape for e in emb_list])

# emb_padded = padder(emb_list).to(device)
# print("padded:", emb_padded.shape)

pipeline = Serial([MeanAggregator(), NeuralNet(hidden_size=768, out_size=32)]).to(device)
print("pipeline params on:", next(pipeline.parameters()).device)
# print("out:", pipeline(emb_padded).shape)


class OneHot(BaseModel):
    """
    Turns a sparse integer label into a one-hot encoded vector.
    """

    num_classes: int
    label_col: str
    target_col: str

    def __call__(self, observation):
        vec = torch.zeros(self.num_classes, dtype=torch.float32)
        vec[observation[self.label_col]] = 1.0
        return {self.target_col: vec}
# Waarom one-hot + BCEWithLogits:
# - je traint alsof het multi-label is (ook al is het in praktijk mogelijk single-label)
# - het werkt, maar is iets anders dan CrossEntropyLoss
onehot = OneHot(num_classes=32, label_col="labels", target_col="onehot")

# Dataset wordt uitgebreid met key "onehot"
train_oh = train.map(onehot)
valid_oh = valid.map(onehot)

# Collate: maakt batches (X, y) met padding + stacking
class Collate(BaseModel):
    """
    processes a batch of Dataset items into padded tensors
    """
    embedding_col: str
    target_col: str
    padder: Callable

    def __call__(self, batch):
        embeddings = [item[self.embedding_col] for item in batch]
        X = self.padder(embeddings)
        y = torch.stack([item[self.target_col] for item in batch]).float()
        return X, y

# DataLoader geeft normaal een lijst dicts.
# Dit collate_fn maakt daar iets van waar je model mee kan werken:
# X: padded embeddings → (B, 30, 768)
# y: targets → (B, 32)
collate_fn = Collate(
    embedding_col="legal_dutch",
    target_col="onehot",
    padder=padder,
)

trainloader = DataLoader(train_oh, batch_size=32, shuffle=True, collate_fn=collate_fn) # shuffle=True => alleen voor training
validloader = DataLoader(valid_oh, batch_size=32, shuffle=False, collate_fn=collate_fn)
settings1 = make_settings("run1_baseline", trainloader, validloader, prefix="baseline")

# Sanity check: één batch naar device
X, y = next(iter(trainloader))
X = X.to(device)
y = y.to(device)
print(X.device, y.device)

print("X,y shapes:", X.shape, y.shape)

loss_fn = torch.nn.BCEWithLogitsLoss()

trainer = Trainer(
    model=pipeline,
    settings=settings1,
    loss_fn=loss_fn,
    optimizer=optim.Adam,
    traindataloader=trainloader,
    validdataloader=validloader,
    scheduler=optim.lr_scheduler.ReduceLROnPlateau,
    device=device,
)
print("MODEL PARAM DEVICE:", next(trainer.model.parameters()).device)

trainer.loop()
print("================== EINDE 1E NOTEBOOK ==================")



# ================== 2E NOTEBOOK ==================

# Initialize & fit with training_texts
# RegexVectorizer: maakt een extra feature-kolom "regex"
# Doel: uit de ruwe tekst (traincache["text"]) regel-gebaseerde patronen halen (zoals wetsverwijzingen / juridische referenties).
# pattern_builder bouwt regex-patronen (bv. “art. 6:162 BW” varianten).
# harmonizer normaliseert varianten naar 1 canonical vorm.
# min_doc_frequency=15: een regex-feature wordt alleen meegenomen als die in minstens 15 documenten voorkomt.
# max_features=200: cap op aantal regex-features.
# col_name="regex": de vector die hieruit komt gaat in datasetkolom regex (meestal een 1D vector per sample, bv (123,)).
# device="cpu" is logisch: regex-building is geen GPU-werk.

regexvectorizer = RegexVectorizer(
    col_name="regex",
    pattern_builder=build_legal_reference_pattern,
    harmonizer=harmonize_legal_reference,
    min_doc_frequency=15,
    max_features=200,
    device="cpu",
    training_texts=traincache["text"],  # we fit it on all 15k texts!
)

# VectorCache.create: voegt die "regex" kolom toe aan je dataset
extended_traincache = VectorCache.create(
    cache_dir=Path("tmp/artefacts"),
    vectorizer=regexvectorizer,  # use our new regex vectorizer
    dataset=traincache.dataset,  # use the existing dataset
    dataset_tag=traintag,  # this will check for existing metadata.json
)

extended_validcache = VectorCache.create(
    cache_dir=Path("tmp/artefacts"),
    vectorizer=regexvectorizer,  # use our new regex vectorizer
    dataset=validcache.dataset,  # use the existing dataset
    dataset_tag=validtag,  # this will check for existing metadata.json
)

# CollateParallel: maakt batches met twee inputs (X1, X2)
class CollateParallel(BaseModel):
    """
    processes a batch of Dataset items into padded tensors
    """

    vec1_col: str
    vec2_col: str
    target_col: str
    padder: Callable

    # X1 = padded BERT chunk embeddings.
    # X2 = regex feature vectors gestackt (geen padding nodig).
    # y = one-hot labels.
    # DataLoader levert nu: ((X1, X2), y).
    # Belangrijk: het model moet dus ook een tuple als input accepteren
    def __call__(self, batch):
        embeddings1 = [
            item[self.vec1_col] for item in batch
        ]  # 2D tensors (chunks, dim) =>  legal_dutch: (chunks, 768)
        embeddings2 = [item[self.vec2_col] for item in batch]  # 1D tensors (dim,) =>  regex: (dims,)
        X1 = self.padder(
            embeddings1
        )  # pad the 2D tensor, now it is a 3D (batch, chunks, dim) -> (B, max_chunks, 768)
        X2 = torch.stack(embeddings2).float()  # the regex doesnt need padding  -> (B, regex_dims)
        y = torch.stack([item[self.target_col] for item in batch]).float()
        return (X1, X2), y

collate_fn = CollateParallel(
    vec1_col="legal_dutch",
    vec2_col="regex",
    target_col="onehot",
    padder=FixedPadding(max_chunks=30),
)

# range van 1024 of alles pakken
# train = extended_traincache.select(range(1024))
# valid = extended_validcache.select(range(1024))
train = extended_traincache.dataset
valid = extended_validcache.dataset
train_oh = train.map(onehot)
valid_oh = valid.map(onehot)

trainloader = DataLoader(train_oh, batch_size=32, shuffle=True, collate_fn=collate_fn)
validloader = DataLoader(valid_oh, batch_size=32, shuffle=False, collate_fn=collate_fn)
settings2 = make_settings("run2_parallel", trainloader, validloader, prefix="baseline")
settings3 = make_settings("run3_skip", trainloader, validloader, prefix="baseline")

(X1, X2), y = next(iter(trainloader))
X1 = X1.to(device)
X2 = X2.to(device)
y  = y.to(device)
print(X1.device, X2.device, y.device)

# en als je daarna een X wil hebben:
X = (X1, X2)


# Parallel2 + pipeline2: twee “paden” die later samenkomen
# Dit zijn twee branches:
# - Branch A (voor X1 = BERT chunks)
# - Branch B (voor X2 = regex vector)
# Dus: beide inputs worden naar een zelfde latent space van 32 geprojecteerd.

parallel2 = Parallel(
    [
        # (batch, chunks, dims) -> (batch, dims) -> (batch, 32)
        Serial([MeanAggregator(), NeuralNet(hidden_size=768, out_size=32)]),
        # (batch, dims) -> (batch, 32)
        Serial([NeuralNet(hidden_size=123, out_size=32)]),
    ]
)

# parallel pipeline
pipeline2 = Serial(
    [
        parallel2,  # (X1, X2) -> (batch, 32), (batch, 32)
        Concatenate2D(),  # (batch, 32), (batch, 32) -> (batch, 64) => Concat de twee 32-vectors → 64.
        NeuralNet(hidden_size=64, out_size=32),  # (batch, 64) -> (batch, 32) => laatste  doet 64 → 32 (jouw classes) en levert logits voor BCEWithLogitsLoss.
    ]
)
pipeline2 = pipeline2.to(device)
print(next(pipeline2.parameters()).device)

yhat = pipeline2(X)

trainer2 = Trainer(
    model=pipeline2,
    settings=settings2,
    loss_fn=loss_fn,
    optimizer=optim.Adam,
    traindataloader=trainloader, # geeft tuples terug
    validdataloader=validloader,
    scheduler=optim.lr_scheduler.ReduceLROnPlateau,
    device=device,
)
print("MODEL PARAM DEVICE:", next(trainer2.model.parameters()).device)

trainer2.loop()
print("================== EINDE 1E LOOP 2E NOTEBOOK ==================")

# ================== VERVOLG 2E NOTEBOOK ==================

# parallel3 is hetzelfde idee als parallel2
parallel3 = Parallel(
    [
        # (batch, chunks, dims) -> (batch, dims) -> (batch, 32)
        Serial([MeanAggregator(), NeuralNet(hidden_size=768, out_size=32)]),
        # (batch, dims) -> (batch, 32)
        Serial([NeuralNet(hidden_size=123, out_size=32)]),
    ]
)

pipeline3 = Serial(
    [
        parallel3,  # (X1, X2) -> (batch, 32), (batch, 32)
        Concatenate2D(),  # (batch, 32), (batch, 32) -> (batch, 64) => voegt de twee paden samen
        Projection(hidden_size=64, out_size=32),  # (batch, 64) -> (batch, 32) => dit is de “fusie laag” => Vaak is Projection een “lichte” lineaire projectie (soms met dropout/activering afhankelijk van implementatie), bedoeld als feature mixer.
        Skip(
            transform=NeuralNet(hidden_size=32, out_size=32),
            in_size=32,
        ), #Dit is de echte “hypothese 3” stap
    #     Dit is een residual/skip connection idee:
    #     - Je hebt al een tensor na Projection: noem die h met shape (B,32).
    #     - Skip doet typisch iets in de trant van: out = h + f(h) waar f hier jouw transform is (een NN 32→32).
    # Waarom is dit nuttig?
    # - Het voorkomt dat alle informatie “door” een extra niet-lineaire transformatie móét.
    # - Het model kan leren: “laat dit signaal grotendeels intact, en pas alleen een correctie toe”.
    # - Dat is exact het soort signaalbehoud waar jouw hypothese op doelt (minder vervorming van informatieve features → stabieler/generaliseerbaarder).

    ]
)

pipeline3 = pipeline3.to(device)
print(next(pipeline3.parameters()).device)

trainer3 = Trainer(
    model=pipeline3,
    settings=settings3,
    loss_fn=loss_fn,
    optimizer=optim.Adam,
    traindataloader=trainloader,
    validdataloader=validloader,
    scheduler=optim.lr_scheduler.ReduceLROnPlateau,
    device=device,
)
print("MODEL PARAM DEVICE:", next(trainer3.model.parameters()).device)

trainer3.loop()
print("================== EINDE 2E LOOP 2E NOTEBOOK ==================")

# ================== 3E NOTEBOOK ==================

train_oh = traincache.dataset.map(onehot)
valid_oh = validcache.dataset.map(onehot)

collate_fn_moe = Collate(
    embedding_col="legal_dutch",
    target_col="onehot",
    padder=FixedPadding(max_chunks=30),
)

trainloader_moe = DataLoader(train_oh, batch_size=32, shuffle=True, collate_fn=collate_fn_moe)
validloader_moe = DataLoader(valid_oh, batch_size=32, shuffle=False, collate_fn=collate_fn_moe)

moe = MoE(
    experts=[
        NeuralNet(hidden_size=768, out_size=32),
        NeuralNet(hidden_size=768, out_size=32),
        NeuralNet(hidden_size=768, out_size=32),
        NeuralNet(hidden_size=768, out_size=32),
    ],
    hidden_size=768,
    out_size=32,
    top_k=2,
)

pipeline4 = Serial([MeanAggregator(), moe]).to(device)
settings4 = make_settings("run4_moe", trainloader_moe, validloader_moe, prefix="baseline")
loss_fn_moe = torch.nn.BCEWithLogitsLoss()

X, y = next(iter(trainloader_moe))
print("MOE batch type:", type(X), "shape:", X.shape, "y:", y.shape)

trainer4 = Trainer(
    model=pipeline4,
    settings=settings4,
    loss_fn=loss_fn_moe,
    optimizer=optim.Adam,
    traindataloader=trainloader_moe,
    validdataloader=validloader_moe,
    scheduler=optim.lr_scheduler.ReduceLROnPlateau,
    device=device,
)

trainer4.loop()
print("================== EINDE 3E NOTEBOOK ==================")
# shutil.rmtree("tmp/", ignore_errors=True)
# shutil.rmtree("logs/", ignore_errors=True)
# shutil.rmtree("demo/", ignore_errors=True)

