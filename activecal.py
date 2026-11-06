"""activecal: offline active-calibration simulator (direction B) on top of ubpkd (UBP code, commit 2e513ae).

Question: for a new user, which images should they view during calibration so that a decoder pretrained on other
subjects adapts best, counted in recorded presentations?

Protocol (Oct 20 brief, direction B):
  * Source decoder f0: UBP EEGProject trained on every subject except the target (LOSO, ubpkd's loso path),
    uniform-11 blur targets. f0 never sees the 200 held-out concepts (split_indices('dev') removes them everywhere).
  * Target = simulated new user. Its 200 held-out concepts are the evaluation set (never selectable); every other
    training image is the acquisition pool.
  * Evaluation: 200-way retrieval on the held-out concepts, averaged over 10 draws (one image per concept per draw):
    2,000 queries instead of 200, ~3x less noise. Reported with its standard error.
  * acquire='rep': one presentation reveals one repetition (4 per image); acquire='image': all 4 at once (4 presentations).
    Training input = mean of the revealed repetitions.
  * All policies start from the same 25 random images (per seed); each round adds min(gap to the next budget,
    max(25, presentations // 2)) presentations (~2 rounds per doubling of the budget).
  * Adapters: 'lowrank' (W0 + B@A on the first layer), 'full' (first layer, L2 pull to f0), 'all' (every layer, L2 pull
    to f0), 'scratch' (fresh model on target data only: does the prior help at all?). Optional rehearsal on source
    samples. Contrastive negatives come from the whole pool's image targets (features only, no EEG).
  * A policy only sees image features, already-revealed target responses and source-subject recordings. Repetition
    consistency is logged post hoc (noise check).

Groups (run_group): 'diag' (fundamental checks + full-pool ceilings), 'gate' (random policy x adapter settings),
'main' (all policies with the frozen adapter). Every row goes to one CSV; finished runs are skipped by tag.
"""
import os, csv, copy, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ubpkd as U

VERSION = 'activecal-0.3'
TARGET_SPEC = 'uniform:11'
SEED_SET = 25
BUDGETS = [25, 50, 100, 200, 400, 800, 1600, 3200]
POLICIES = ['random', 'category', 'kcenter', 'typiclust', 'error_region', 'expected_gain', 'intersubject', 'repeat_new']
BASE_CFG = dict(adapter='lowrank', rank=8, lr=1e-3, wd=1e-4, steps=200, batch=64, anchor=10.0, n_neg=1024,
                rehearse=0.0, acquire='rep', scratch_steps=1500, scratch_lr=3e-4, select=False)
GATE_CFGS = [dict(adapter='lowrank', rank=8, lr=1e-3, acquire='rep'),
             dict(adapter='lowrank', rank=8, lr=1e-3, acquire='image'),
             dict(adapter='lowrank', rank=32, lr=1e-3, acquire='image'),
             dict(adapter='full', anchor=10.0, lr=1e-4, acquire='image'),
             dict(adapter='all', anchor=1.0, lr=1e-4, acquire='image'),
             dict(adapter='all', anchor=1.0, lr=1e-4, acquire='image', rehearse=0.5),
             dict(adapter='scratch', acquire='image')]
_BEST = dict(adapter='all', anchor=1.0, lr=1e-4, acquire='image', rehearse=0.5)
GATE_CFGS += [dict(_BEST, steps=n) for n in (25, 50, 100)] + [dict(_BEST, select=True)]   # 0.3: step-count check
SELECT_GRID = [0, 25, 50, 100, 200]       # select=True: steps chosen on 20% of the acquired images, 0 = keep f0
ENV = dict(results_csv=None, frozen=dict(BASE_CFG))      # frozen = the adapter used by group 'main'


def make_cfg(**kw):
    c = dict(BASE_CFG)
    c.update(kw)
    return c


def cfg_name(c):
    if c['adapter'] == 'scratch':
        return f"scratch-s{c['scratch_steps']}-lr{c['scratch_lr']:g}-{c['acquire']}"
    core = c['adapter'] + (f"-r{c['rank']}" if c['adapter'] == 'lowrank' else f"-a{c['anchor']:g}")
    return f"{core}-lr{c['lr']:g}-s{c['steps']}-reh{c['rehearse']:g}-{c['acquire']}" + ('-sel' if c.get('select') else '')


def run_tag(group, target, montage, policy, cfg, seed):
    return f"{group}/{montage}-{policy}|{cfg_name(cfg)}|t{target:02d}|seed{seed}|{U.ENV['source']}"


# ----------------------------------------------------------------------------------------------------------------
# Source model (LOSO) and target data
# ----------------------------------------------------------------------------------------------------------------
def ensure_source(target, montage):
    cfg = U.make_cfg('al_source', montage, TARGET_SPEC, lr=1e-5, early_stop='val', single_trial=False, save=True,
                     loso_target=target, label=f'{montage}-proj-{TARGET_SPEC}-source')
    ck = os.path.join(U.ENV['ckpt_dir'], U.run_tag(cfg).replace('/', '_').replace('|', '__') + '.pt')
    if not os.path.exists(ck):
        print(f'  training source model {U.run_tag(cfg)}', flush=True)
        U.append_result(U.run(cfg))
    m = U.build_model('EEGProjectLayer', len(U.MONTAGES[montage]), U.CLIP_MAP['RN50'][1]).to(U.ENV['device'])
    m.load_state_dict(torch.load(ck, weights_only=False)['state_dict'])
    return m.eval()


def subject_reps(sub, montage):
    """Un-averaged training trials (n_img, 4, C, T): subject_montage's selection + whitening, without the averaging."""
    path = os.path.join(U.ENV['cache_dir'], U.ENV['source'], f'sub-{sub:02d}', f'{montage}_reps.npz')
    if not os.path.exists(path):
        idx = [U.CHANNELS.index(c) for c in U.MONTAGES[montage]]
        tr = U._load_raw(sub, 'train', U.ENV['source'])
        x = np.asarray(tr['eeg'])[:, :, idx, :U.T_LEN].astype(np.float16)
        if U.ENV['source'] == 'clean_mvnn':
            te = U._load_raw(sub, 'test', U.ENV['source'])
            x, _ = U.mvnn_montage(x, np.asarray(tr['session']),
                                  np.asarray(te['eeg'])[:, :, idx, :U.T_LEN].astype(np.float16), np.asarray(te['session']))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, reps=x, paths=np.asarray(tr['img'])[:, 0].astype(str))
    d = np.load(path)
    return d['reps'], d['paths']


class Prep:
    """Everything one (target, montage) needs, shared by all policies, seeds and adapters."""

    def __init__(self, target, montage, n_rehearse=20000, n_draws=10):
        dev = U.ENV['device']
        self.target, self.montage, self.f0 = target, montage, ensure_source(target, montage)
        d = U.subject_montage(target, montage)
        reps, rpaths = subject_reps(target, montage)
        assert (rpaths == d['train_paths']).all(), 'repetition and averaged caches disagree on image order'
        labels, paths = d['train_labels'], d['train_paths']
        pool, _ = U.split_indices(labels, 'dev')
        held = np.setdiff1d(np.unique(labels), np.unique(labels[pool]))
        assert len(held) == 200 and not np.isin(labels[pool], held).any(), 'held-out concepts leak into the pool'
        self.P, self.concept, self.reps, self.n = paths[pool], labels[pool], reps[pool], len(pool)
        self.full_x = self.reps.astype(np.float32).mean(1)
        ev = [np.array([np.nonzero(labels == c)[0][j] for c in held]) for j in range(n_draws)]
        self.ev_x = [d['train'][ix] for ix in ev]
        self.ev_g = [U.feats_for(paths[ix], TARGET_SPEC, 'RN50') for ix in ev]
        self.ev_gc = [U.feats_for(paths[ix], 'none', 'RN50') for ix in ev]
        self.ev_st = reps[ev[0]].reshape(-1, *reps.shape[2:])
        self.ev_idx, self.y = ev, np.arange(len(held))
        self.tgt = U.feats_for(self.P, TARGET_SPEC, 'RN50').to(dev)
        self.clip = F.normalize(U.feats_for(self.P, 'none', 'RN50'), dim=-1).to(dev)
        self._sources(n_rehearse)
        self._typicality()
        x = torch.from_numpy(self.reps.reshape(self.n, self.reps.shape[1], -1).astype(np.float32))
        x = F.normalize(x - x.mean(-1, keepdim=True), dim=-1)
        r = x.shape[1]
        self.consistency = (((x @ x.transpose(1, 2)).sum((1, 2)) - r) / (r * (r - 1))).numpy()   # post hoc only
        self.f0_eval = evaluate(self.f0, self)

    def _sources(self, n_rehearse):
        """Inter-subject dispersion of f0 embeddings, mean source EEG (surrogate input), rehearsal samples."""
        pos = {p: i for i, p in enumerate(self.P)}
        srcs = [s for s in U.ENV['pool_subjects'] if s != self.target]
        acc_u, acc_x, rx, rt, self.src_stats = 0, 0, [], [], {}
        rng = np.random.RandomState(0)
        for s in srcs:
            d = U.subject_montage(s, self.montage)
            rows = np.array([pos.get(p, -1) for p in d['train_paths']])
            x = np.zeros((self.n,) + d['train'].shape[1:], np.float32)
            x[rows[rows >= 0]] = d['train'][rows >= 0].astype(np.float32)
            acc_u = acc_u + F.normalize(U.embed(self.f0, x), dim=-1)
            acc_x = acc_x + x
            pick = rng.choice(self.n, n_rehearse // len(srcs), replace=False)
            rx.append(x[pick].astype(np.float16)); rt.append(pick)
            self.src_stats[s] = (float(x.mean()), float(x.std()))
        self.dispersion = (1 - (acc_u / len(srcs)).norm(dim=-1)).cpu().numpy()
        self.x_surrogate = (acc_x / len(srcs)).astype(np.float16)
        self.reh_x, self.reh_t = np.concatenate(rx), np.concatenate(rt)

    def _typicality(self, k=20, chunk=2048):
        dist = torch.cat([(1 - (self.clip[a:a + chunk] @ self.clip.T).topk(k + 1, dim=-1).values[:, 1:]).mean(-1)
                          for a in range(0, self.n, chunk)])
        self.typ = (1 / dist.clamp_min(1e-6)).cpu().numpy()
        x = self.clip - self.clip.mean(0)
        self.clip_pca = (x @ torch.linalg.svd(x, full_matrices=False)[2][:64].T).cpu().numpy()


@torch.no_grad()
def evaluate(m, prep):
    t = [U.retrieval(m, x, g, prep.y) for x, g in zip(prep.ev_x, prep.ev_g)]
    a = np.array([r[0] for r in t])
    return dict(top1=float(a.mean()), top1_se=float(a.std(ddof=1) / math.sqrt(len(a))),
                top5=float(np.mean([r[1] for r in t])),
                clean_top1=float(np.mean([U.retrieval(m, x, g, prep.y)[0] for x, g in zip(prep.ev_x, prep.ev_gc)])),
                st_top1=U.retrieval(m, prep.ev_st, prep.ev_g[0], np.repeat(prep.y, prep.reps.shape[1]))[0])


# ----------------------------------------------------------------------------------------------------------------
# Adapters
# ----------------------------------------------------------------------------------------------------------------
class LowRankLinear(nn.Module):
    def __init__(self, lin, rank):
        super().__init__()
        self.register_buffer('W0', lin.weight.detach().clone())
        self.register_buffer('b0', None if lin.bias is None else lin.bias.detach().clone())
        self.A = nn.Parameter(torch.zeros(rank, lin.in_features, device=lin.weight.device))
        self.B = nn.Parameter(torch.randn(lin.out_features, rank, device=lin.weight.device) * 0.01)

    def forward(self, x):
        return F.linear(x, self.W0 + self.B @ self.A, self.b0)


def _swap_first_linear(m, new):
    lin = U.first_linear(m)
    for mod in m.modules():
        for name, child in mod.named_children():
            if child is lin:
                return setattr(mod, name, new)
    raise RuntimeError('first linear layer not found')


def select_steps(prep, X, T, seed, cfg, frac=0.2):
    """Hold out 20% of the acquired images, train on the rest, return the grid step count with the best held-out
    cosine to the image targets (0 = f0). Uses only data the user has already recorded."""
    n = len(X)
    perm = np.random.RandomState(seed).permutation(n)
    va, tr = perm[:max(1, int(round(frac * n)))], perm[max(1, int(round(frac * n))):]
    Tv = F.normalize(torch.as_tensor(T[va]).float().to(U.ENV['device']), dim=-1)
    score = lambda m: float((F.normalize(U.embed(m, X[va]), dim=-1) * Tv).sum(-1).mean())
    best, best_s = SELECT_GRID[0], score(prep.f0)
    for k in SELECT_GRID[1:]:
        s = score(adapt(prep, X[tr], T[tr], seed, dict(cfg, select=False), steps=k))
        if s > best_s:
            best, best_s = k, s
    return best


def adapt(prep, X, T, seed, cfg, steps=None):
    """Calibrate on acquired pairs (X inputs, T image targets), from f0 or from scratch."""
    dev = U.ENV['device']
    if cfg.get('select') and steps is None and len(X) >= 5:
        T = T.cpu().numpy() if torch.is_tensor(T) else np.asarray(T)
        X = np.asarray(X)
        steps = select_steps(prep, X, T, seed, cfg)
    torch.manual_seed(seed)
    X, T = torch.as_tensor(X).to(dev).float(), torch.as_tensor(T).to(dev).float()
    anchor, n_reh = None, 0
    if cfg['adapter'] == 'scratch':
        m = U.build_model('EEGProjectLayer', len(U.MONTAGES[prep.montage]), U.CLIP_MAP['RN50'][1]).to(dev)
        params, lr, steps = list(m.parameters()), cfg['scratch_lr'], cfg['scratch_steps'] if steps is None else steps
    else:
        m = copy.deepcopy(prep.f0).requires_grad_(False)
        lr, steps, n_reh = cfg['lr'], cfg['steps'] if steps is None else steps, int(round(cfg['rehearse'] * cfg['batch']))
        if cfg['adapter'] == 'lowrank':
            low = LowRankLinear(U.first_linear(m), cfg['rank'])
            _swap_first_linear(m, low)
            params = [low.A, low.B]
        else:
            params = list((U.first_linear(m) if cfg['adapter'] == 'full' else m).parameters())
            for p in params:
                p.requires_grad_(True)
            anchor = [(p, p.detach().clone()) for p in params]
    if steps == 0:
        return m.eval()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg['wd'])
    n, bs = len(X), min(cfg['batch'], len(X))
    m.train()
    for _ in range(steps):
        idx = torch.randperm(n, device=dev)[:bs]
        x, t = X[idx], T[idx]
        if n_reh:
            j = np.random.randint(0, len(prep.reh_x), n_reh)
            x = torch.cat([x, torch.as_tensor(prep.reh_x[j]).to(dev).float()])
            t = torch.cat([t, prep.tgt[prep.reh_t[j]]])
        gal = torch.cat([t, prep.tgt[torch.randint(prep.n, (cfg['n_neg'],), device=dev)]])
        loss = F.cross_entropy(m.softplus(m.logit_scale) * m(x) @ gal.T, torch.arange(len(x), device=dev))
        if anchor and cfg['anchor'] > 0:
            loss = loss + cfg['anchor'] * sum((p - p0).pow(2).sum() for p, p0 in anchor) / sum(p0.pow(2).sum() for _, p0 in anchor)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return m.eval()


# ----------------------------------------------------------------------------------------------------------------
# Policies: each returns k pool indices to present next
# ----------------------------------------------------------------------------------------------------------------
class State:
    def __init__(self, prep, seed):
        self.prep, self.seed, self.model = prep, seed, prep.f0
        self.rng = np.random.RandomState(seed)
        self.used = np.zeros(prep.n, np.int64)
        self.order = np.argsort(np.random.RandomState(1000 + seed).rand(prep.n, prep.reps.shape[1]), axis=1)

    def acquired(self):
        idx = np.nonzero(self.used)[0]
        r = self.prep.reps[idx]
        return idx, np.stack([r[j, self.order[i, :self.used[i]]].astype(np.float32).mean(0) for j, i in enumerate(idx)])

    def new(self):
        return np.nonzero(self.used == 0)[0]

    def loss(self, idx, x):
        z = F.normalize(U.embed(self.model, x), dim=-1)
        return (1 - (z * F.normalize(self.prep.tgt[idx], dim=-1)).sum(-1)).cpu().numpy()


def _top(scores, cand, k):
    return cand[np.argsort(-scores)[:k]]


def p_random(st, k):
    return st.rng.choice(st.new(), k, replace=False)


def p_category(st, k):
    cand = st.new()
    counts = np.bincount(st.prep.concept[st.used > 0], minlength=st.prep.concept.max() + 1)
    order = cand[np.argsort(counts[st.prep.concept[cand]] + st.rng.rand(len(cand)))]
    _, first = np.unique(st.prep.concept[order], return_index=True)          # one image per least-covered concept
    picks = order[np.sort(first)][:k]
    return picks if len(picks) == k else np.concatenate([picks, np.setdiff1d(order, picks)[:k - len(picks)]])


def p_kcenter(st, k):
    clip, cand = st.prep.clip, torch.as_tensor(st.new(), device=st.prep.clip.device)
    d = (1 - clip[cand] @ clip[np.nonzero(st.used)[0]].T).min(-1).values
    out = []
    for _ in range(k):
        j = int(d.argmax())
        out.append(int(cand[j]))
        d = torch.minimum(d, 1 - clip[cand] @ clip[cand[j]])
    return np.array(out)


def p_typiclust(st, k):
    from sklearn.cluster import MiniBatchKMeans
    acq = np.nonzero(st.used)[0]
    km = MiniBatchKMeans(n_clusters=min(len(acq) + k, st.prep.n), random_state=st.seed, n_init=1, batch_size=4096)
    lab = km.fit_predict(st.prep.clip_pca)
    covered, out = set(lab[acq]), []
    for c in np.argsort(-np.bincount(lab, minlength=km.n_clusters)):
        members = np.nonzero((lab == c) & (st.used == 0))[0]
        if c not in covered and len(members):
            out.append(members[st.prep.typ[members].argmax()])
        if len(out) == k:
            break
    rest = np.setdiff1d(st.new(), out)
    return np.array(out + list(rest[np.argsort(-st.prep.typ[rest])[:k - len(out)]]))


def p_error_region(st, k, nn_=5):
    """Uncertainty proxy available before observation: candidates near acquired images the adapted decoder fails on."""
    idx, x = st.acquired()
    loss, cand = torch.as_tensor(st.loss(idx, x), device=st.prep.clip.device), st.new()
    w, nb = (st.prep.clip[cand] @ st.prep.clip[idx].T).topk(min(nn_, len(idx)), dim=-1)
    return _top(((w.clamp_min(0) * loss[nb]).sum(-1) / w.clamp_min(1e-6).sum(-1)).cpu().numpy(), cand, k)


def p_expected_gain(st, k, n_cand=3000):
    """Gradient-embedding size at the first layer, with the mean source EEG standing in for the unseen response."""
    dev, m, store = U.ENV['device'], st.model, {}

    def hook(mod, inp, out):
        out = out.detach().requires_grad_(True)
        store['h'] = out
        return out
    cand = st.rng.choice(st.new(), min(n_cand, len(st.new())), replace=False)
    h = U.first_linear(m).register_forward_hook(hook)
    x = torch.as_tensor(st.prep.x_surrogate[cand]).to(dev).float()
    m.eval()
    with torch.enable_grad():
        logits = m.softplus(m.logit_scale) * m(x) @ st.prep.tgt[cand].T
        F.cross_entropy(logits, torch.arange(len(cand), device=dev), reduction='sum').backward()
    h.remove()
    return _top((store['h'].grad.norm(dim=-1) * x.flatten(1).norm(dim=-1)).cpu().numpy(), cand, k)


def p_intersubject(st, k):
    cand = st.new()
    return _top(st.prep.dispersion[cand], cand, k)


def p_repeat_new(st, k):
    """Half re-show acquired images fit worst (if repetitions remain), half k-center new. With acquire='image' every
    acquired image is complete, so this is k-center."""
    idx, x = st.acquired()
    ok = st.used[idx] < st.prep.reps.shape[1]
    rep = idx[ok][np.argsort(-st.loss(idx, x)[ok])[:k // 2]]
    return np.concatenate([rep, p_kcenter(st, k - len(rep))]).astype(np.int64)


POLICY_FN = dict(random=p_random, category=p_category, kcenter=p_kcenter, typiclust=p_typiclust,
                 error_region=p_error_region, expected_gain=p_expected_gain, intersubject=p_intersubject,
                 repeat_new=p_repeat_new)


# ----------------------------------------------------------------------------------------------------------------
# Runs and registry
# ----------------------------------------------------------------------------------------------------------------
COLUMNS = ['tag', 'group', 'target', 'montage', 'policy', 'config', 'seed', 'presentations', 'n_images', 'top1',
           'top1_se', 'top5', 'clean_top1', 'st_top1', 'f0_top1', 'consistency_acq', 'consistency_pool', 'source',
           'wall_s', 'version', 'time']


def done_tags():
    p = ENV['results_csv']
    if not p or not os.path.exists(p):
        return set()
    with open(p) as f:
        return {r['tag'] for r in csv.DictReader(f)}


def append_rows(rows):
    p = ENV['results_csv']
    new = not os.path.exists(p)
    with open(p, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def _row(tag, group, prep, policy, cfg, seed, presentations, n_images, ev, cons, t0):
    return dict(tag=tag, group=group, target=prep.target, montage=prep.montage, policy=policy, config=cfg_name(cfg),
                seed=seed, presentations=presentations, n_images=n_images, f0_top1=prep.f0_eval['top1'],
                consistency_acq=cons, consistency_pool=float(prep.consistency.mean()), source=U.ENV['source'],
                wall_s=round(time.time() - t0, 1), version=VERSION, time=time.strftime('%Y-%m-%d %H:%M'), **ev)


def simulate(prep, policy, cfg, seed, group):
    """One calibration run: rows at every budget reached (the tag repeats; the run is done when its last row exists)."""
    t0, tag = time.time(), run_tag(group, prep.target, prep.montage, policy, cfg, seed)
    per = prep.reps.shape[1] if cfg['acquire'] == 'image' else 1
    st = State(prep, seed)
    st.used[st.rng.choice(prep.n, SEED_SET, replace=False)] = per
    budgets, rows = [b for b in BUDGETS if b >= st.used.sum()], []
    while budgets:
        idx, x = st.acquired()
        st.model = adapt(prep, x, prep.tgt[idx], seed, cfg)
        pres = int(st.used.sum())
        if pres == budgets[0]:
            budgets.pop(0)
            rows.append(_row(tag, group, prep, policy, cfg, seed, pres, len(idx), evaluate(st.model, prep),
                             float(prep.consistency[idx].mean()), t0))
        if not budgets:
            break
        k = max(1, min(budgets[0] - pres, max(SEED_SET, pres // 2)) // per)
        picks = np.unique(POLICY_FN[policy](st, k)).astype(np.int64)
        picks = picks[st.used[picks] == 0] if per > 1 else picks
        st.used[picks] += per
    append_rows(rows)
    return rows


def ceilings(prep, cfg, group):
    """f0 (presentations 0) and full-pool ceilings (presentations -1): the adapter, every layer from f0, scratch."""
    t0, rows = time.time(), []
    tag = lambda pol: run_tag(group, prep.target, prep.montage, pol, cfg, 0)
    rows.append(_row(tag('f0'), group, prep, 'f0', cfg, 0, 0, 0, prep.f0_eval, float(prep.consistency.mean()), t0))
    for pol, c in [('full_pool', cfg), ('full_pool_all', dict(cfg, adapter='all', anchor=0.0, lr=1e-4)),
                   ('full_pool_scratch', dict(cfg, adapter='scratch', scratch_lr=1e-4))]:
        m = adapt(prep, prep.full_x, prep.tgt, 0, dict(c, batch=1024, rehearse=0.0), steps=1000)
        rows.append(_row(tag(pol), group, prep, pol, cfg, 0, -1, prep.n, evaluate(m, prep),
                         float(prep.consistency.mean()), t0))
    append_rows(rows)
    return rows


def diagnostics(prep, cfg):
    """Fundamental checks before trusting any calibration result."""
    e = prep.f0_eval
    print(f'=== diag: target {prep.target} {prep.montage} | pool {prep.n} images, eval 200 held-out concepts x '
          f'{len(prep.ev_x)} draws (disjoint: checked)')
    print(f'  EEG mean/std target {prep.full_x.mean():+.3f}/{prep.full_x.std():.3f} | sources ' +
          ' '.join(f's{s} {m:+.3f}/{sd:.3f}' for s, (m, sd) in prep.src_stats.items()))
    print(f'  f0 on target: top1 {e["top1"]:.2f} ± {e["top1_se"]:.2f}, clean {e["clean_top1"]:.2f}, single-trial '
          f'{e["st_top1"]:.2f} (chance 0.5)')
    same = abs(evaluate(adapt(prep, prep.full_x[:25], prep.tgt[:25], 0, cfg, steps=0), prep)['top1'] - e['top1']) < 1e-9
    print(f'  identity (adapter, 0 steps == f0): {"OK" if same else "FAILED"}')
    for s in [s for s in U.ENV['pool_subjects'] if s != prep.target][:2]:
        d = U.subject_montage(s, prep.montage)
        a = np.mean([U.retrieval(prep.f0, d['train'][ix], g, prep.y)[0] for ix, g in zip(prep.ev_idx, prep.ev_g)])
        print(f'  f0 on source subject {s} held-out concepts: top1 {a:.2f}')
    for r in ceilings(prep, cfg, 'diag')[1:]:
        print(f'  {r["policy"]:18s} top1 {r["top1"]:.2f} ± {r["top1_se"]:.2f}')


def run_group(group, targets, montages, seeds=(0, 1, 2), policies=POLICIES):
    """'diag': checks + ceilings. 'gate': random policy x GATE_CFGS. 'main': every policy with ENV['frozen']."""
    for t in targets:
        for mon in montages:
            if group == 'diag':
                jobs, cfg = [], make_cfg(**ENV['frozen'])
                todo_ceil = run_tag('diag', t, mon, 'full_pool_scratch', cfg, 0) not in done_tags()
            else:
                cfgs = GATE_CFGS if group == 'gate' else [ENV['frozen']]
                pols = ['random'] if group == 'gate' else policies
                jobs = [(p, make_cfg(**c), s) for c in cfgs for p in pols for s in seeds]
                jobs = [j for j in jobs if run_tag(group, t, mon, j[0], j[1], j[2]) not in done_tags()]
                cfg = make_cfg(**ENV['frozen'])
                todo_ceil = group == 'main' and run_tag('main', t, mon, 'full_pool', cfg, 0) not in done_tags()
            print(f'[{group}] target {t} {mon}: {len(jobs)} runs to do', flush=True)
            if not jobs and not todo_ceil:
                continue
            prep = Prep(t, mon)
            if group == 'diag':
                diagnostics(prep, cfg)
            elif todo_ceil:
                for r in ceilings(prep, cfg, group):
                    print(f'  {r["policy"]:18s} {r["top1"]:.2f}', flush=True)
            for p, c, s in jobs:
                rows = simulate(prep, p, c, s, group)
                print(f'  {p:13s} {cfg_name(c):34s} seed {s} | f0 {prep.f0_eval["top1"]:.1f} | ' +
                      ' '.join(f"{r['presentations']}:{r['top1']:.1f}" for r in rows) + f"  ({rows[-1]['wall_s']:.0f}s)",
                      flush=True)
            U.free_raw()


# ----------------------------------------------------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------------------------------------------------
def gate_table(csv_path=None):
    import pandas as pd
    df = pd.read_csv(csv_path or ENV['results_csv'])
    g = df[df.group == 'gate']
    t = g.pivot_table(index=['montage', 'config'], columns='presentations', values='top1', aggfunc='mean').round(2)
    t.insert(0, 'f0', g.groupby(['montage', 'config']).f0_top1.mean().round(2))
    return t


def summary(csv_path=None, frac=0.5):
    """'main' group: AUC over log2(presentations), gain over f0 (precondition > 0 for random), paired AUC vs random and
    vs typiclust per target (seeds averaged first), presentations to reach f0 + frac x (full_pool - f0)."""
    import pandas as pd
    df = pd.read_csv(csv_path or ENV['results_csv'])
    df = df[(df.group == 'main') & (df.config == cfg_name(make_cfg(**ENV['frozen'])))]
    ref = df[df.presentations <= 0].pivot_table(index=['target', 'montage'], columns='policy', values='top1')
    cur = df[df.presentations > 0]
    trap = getattr(np, 'trapezoid', None) or np.trapz
    recs = []
    for (tg, mon, pol, sd), g in cur.groupby(['target', 'montage', 'policy', 'seed']):
        g, r = g.sort_values('presentations'), ref.loc[(tg, mon)]
        lp = np.log2(g.presentations.values)
        a = trap(g.top1.values, lp) / (lp[-1] - lp[0])
        hit = g[g.top1 >= r['f0'] + frac * (r['full_pool'] - r['f0'])]
        recs.append(dict(target=tg, montage=mon, policy=pol, auc=a, gain_f0=a - r['f0'],
                         to_thr=hit.presentations.iloc[0] if len(hit) else np.nan, cons=g.consistency_acq.mean()))
    per = pd.DataFrame(recs).groupby(['target', 'montage', 'policy']).mean().reset_index()
    rows = []
    for (mon, pol), g in per.groupby(['montage', 'policy']):
        r = dict(montage=mon, policy=pol, targets=len(g), auc=round(g.auc.mean(), 2), gain_over_f0=round(g.gain_f0.mean(), 2),
                 presentations_to_thr=round(g.to_thr.mean(), 0), consistency=round(g.cons.mean(), 3))
        for b in ('random', 'typiclust'):
            d = (g.set_index('target').auc - per[(per.montage == mon) & (per.policy == b)].set_index('target').auc).dropna()
            if b != pol and len(d):
                r[f'vs_{b}'], r[f'wins_{b}'] = f'{d.mean():+.2f}', f'{int((d > 0).sum())}/{len(d)}'
        rows.append(r)
    out = pd.DataFrame(rows).fillna('').sort_values(['montage', 'auc'], ascending=[True, False])
    curves = cur.groupby(['montage', 'policy', 'presentations']).top1.mean().unstack('presentations').round(2)
    return out, curves, ref.round(2)
