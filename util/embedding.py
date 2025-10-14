import math, random
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Poincaré ball utilities
# ---------------------------
EPS = 1e-5
MAX_NORM = 1 - 1e-3  # keep inside unit ball

def guess_parents(codes: List[str]) -> List[Tuple[str, str]]:
    S = set(codes)
    def nearest_existing_parent(code: str):
        # Try removing trailing chars; also try stripping decimals to block-level
        cand = code
        while len(cand) > 1:
            cand = cand[:-1].rstrip(".")
            if cand in S:
                return cand
        return None

    edges = []
    for c in codes:
        p = nearest_existing_parent(c)
        if p is not None:
            edges.append((p, c))
    return edges

def mobius_proj(x, max_norm=MAX_NORM):
    # project to inside the unit ball
    norm = x.norm(dim=-1, keepdim=True).clamp_min(EPS)
    factor = torch.where(norm >= max_norm, max_norm / norm, torch.ones_like(norm))
    return x * factor

def poincare_dist(u, v, eps=EPS):
    # d(u,v) = arcosh(1 + 2||u-v||^2 / ((1-||u||^2)(1-||v||^2)))
    uu = torch.clamp(1 - (u*u).sum(dim=-1), min=eps)
    vv = torch.clamp(1 - (v*v).sum(dim=-1), min=eps)
    uv = (u - v).pow(2).sum(dim=-1)
    x = 1 + 2 * uv / (uu * vv)
    return torch.acosh(torch.clamp(x, min=1+eps))

# ---------------------------
# Dataset & sampler
# ---------------------------
class HierPairs(torch.utils.data.IterableDataset):
    """Stream positive (parent,child) pairs with on-the-fly negatives."""
    def __init__(self, id_edges: List[Tuple[int,int]], num_nodes: int, neg_k=10):
        super().__init__()
        self.pos = id_edges
        self.num_nodes = num_nodes
        self.neg_k = neg_k
        # adjacency for quick "avoid trivial negatives" if desired
        self.adj = {i:set() for i in range(num_nodes)}
        for p,c in id_edges:
            self.adj[p].add(c)
            self.adj[c].add(p)
    def __iter__(self):
        while True:
            p, c = random.choice(self.pos)
            negs = []
            while len(negs) < self.neg_k:
                j = random.randrange(self.num_nodes)
                if j != p and j != c and (j not in self.adj[p]):
                    negs.append(j)
            yield p, c, torch.tensor(negs, dtype=torch.long)

# ---------------------------
# Model: embedding table on the ball
# ---------------------------
class PoincareEmbedding(nn.Module):
    def __init__(self, num_nodes: int, dim: int = 16, init_scale=1e-3):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, dim)
        nn.init.uniform_(self.emb.weight, a=-init_scale, b=init_scale)
        with torch.no_grad():
            self.emb.weight.copy_(mobius_proj(self.emb.weight))
    def forward(self, idx):
        return mobius_proj(self.emb(idx))

# ---------------------------
# Loss: log-softmax over (one positive, many negatives)
# score(i,j) = -d(i,j) so closer is higher
# ---------------------------
def reconstruction_loss(model: PoincareEmbedding, p_idx, c_idx, negs_idx):
    p = model(p_idx)         # [B, d]
    c = model(c_idx)         # [B, d]
    negs = model(negs_idx)   # [B, K, d]
    # distances
    d_pos = poincare_dist(p, c)                  # [B]
    d_neg = poincare_dist(p.unsqueeze(1), negs)  # [B, K]
    # logits: higher = better ⇒ use -distance
    logits = torch.cat([(-d_pos).unsqueeze(1), -d_neg], dim=1)  # [B, 1+K]
    targets = torch.zeros(p.size(0), dtype=torch.long, device=logits.device)  # positive at index 0
    return F.cross_entropy(logits, targets)

# ---------------------------
# Training loop
# ---------------------------
def train_poincare(
    code2id: Dict[str,int],
    edges: List[Tuple[str,str]],
    dim=16,
    neg_k=10,
    steps=20000,
    batch_size=512,
    lr=5e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    # map edges to ids (parent, child)
    id_edges = [(code2id[p], code2id[c]) for p,c in edges if p in code2id and c in code2id]
    num_nodes = len(code2id)
    ds = HierPairs(id_edges, num_nodes, neg_k=neg_k)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    model = PoincareEmbedding(num_nodes, dim=dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    it = iter(loader)
    avg = 0.0
    for step in range(1, steps+1):
        p_idx, c_idx, neg_idx = next(it)
        p_idx = p_idx.to(device)
        c_idx = c_idx.to(device)
        neg_idx = neg_idx.to(device)

        loss = reconstruction_loss(model, p_idx, c_idx, neg_idx)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # re-project after optimizer step
        with torch.no_grad():
            model.emb.weight.copy_(mobius_proj(model.emb.weight))

        avg = 0.98*avg + 0.02*loss.item()
        if step % 1000 == 0:
            print(f"step {step:6d}  loss ~ {avg:.4f}")

    # return code → embedding dict (torch tensor)
    with torch.no_grad():
        emb = mobius_proj(model.emb.weight.data).cpu()
    id2code = {i:c for c,i in code2id.items()}
    return {id2code[i]: emb[i] for i in range(num_nodes)}

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    codes_all = ["I", "I5", "I50", "I50.9", "E", "E11", "E11.9", "E11.65"]  # replace with yours
    edges = guess_parents(codes_all)
    # Build code vocabulary
    codes = sorted(list({c for e in edges for c in e}))
    code2id = {c:i for i,c in enumerate(codes)}
    # Train
    code2vec = train_poincare(code2id, edges,
                              dim=32, neg_k=10, steps=5000, batch_size=256, lr=1e-2)
    # Access an embedding
    print("I50.9 →", code2vec["I50.9"][:5])  # first 5 dims
