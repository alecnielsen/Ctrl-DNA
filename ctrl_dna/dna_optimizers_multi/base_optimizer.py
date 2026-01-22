import os
#import tdc
import itertools
import json
import time
from functools import lru_cache
from pathlib import Path

import torch
import numpy as np

# Optional imports - only needed if wandb_log=True
try:
    import yaml
except ImportError:
    yaml = None

try:
    import wandb
except ImportError:
    wandb = None

import src.reglm.dataset, src.reglm.lightning, src.reglm.utils, src.reglm.metrics
from .experience import Experience

def evaluate(round_df, starting_sequences):
    data = round_df.sort_values(by='true_score', ascending=False).iloc[:128]
    
    top_fitness = data.iloc[:16]['true_score'].mean().item()
    median_fitness = data['true_score'].median().item()
    
    seqs = data['sequence'].tolist()
    
    distances = [distance(s1, s2) for s1, s2 in itertools.combinations(seqs, 2)]
    diversity = np.median(distances) if distances else 0.0
    
    inits = starting_sequences['sequence'].tolist()
    novelty_distances = [min(distance(seq, init_seq) for init_seq in inits) for seq in seqs]
    novelty = np.median(novelty_distances) if novelty_distances else 0.0
    
    return {
        'top': top_fitness,
        'fitness': median_fitness,
        'diversity': diversity,
        'novelty': novelty
    }
    
    
@lru_cache(maxsize=1)
def _load_fitness_overrides():
    """Load optional fitness range overrides from JSON."""
    override_path = os.getenv("CTRL_DNA_FITNESS_RANGES")
    if override_path:
        path = Path(override_path)
    else:
        try:
            repo_root = Path(__file__).resolve().parents[3]
        except Exception:
            return {}
        path = repo_root / "checkpoints" / "fitness_ranges.json"

    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_fitness_info(cell,oracle_type='paired'):
    overrides = _load_fitness_overrides()
    override = overrides.get(cell)

    if oracle_type=='paired' or oracle_type=='dlow':
        fitness_map = {
            'hepg2': (200, -6.051336, 10.992575),
            'k562': (200, -5.857445, 10.781755),
            'sknsh': (200, -7.283977, 12.888308),
            'JURKAT': (250, -5.574782, 8.413577),
            'K562': (250, -4.088671, 8.555965),
            'THP1': (250, -7.271035, 12.485513),
        }
        if cell in fitness_map:
            length, min_fitness, max_fitness = fitness_map[cell]
        elif override:
            try:
                length = int(override.get("length", 0))
                if length <= 0:
                    raise ValueError("override length missing or invalid")
                min_fitness = float(override["min"])
                max_fitness = float(override["max"])
                return length, min_fitness, max_fitness
            except Exception:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    if override:
        try:
            if "min" in override:
                min_fitness = float(override["min"])
            if "max" in override:
                max_fitness = float(override["max"])
            if "length" in override:
                length = int(override["length"])
        except Exception:
            pass

    return length, min_fitness, max_fitness
    
def top_auc(buffer, top_n, finish, env_log_interval, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(env_log_interval, min(len(buffer), max_oracle_calls), env_log_interval):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum += env_log_interval * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls

def distance(s1, s2):
    return sum([1 if i != j else 0 for i, j in zip(list(s1), list(s2))])

def diversity(seqs):
    divs = []
    for s1, s2 in itertools.combinations(seqs, 2):
        divs.append(distance(s1, s2))
    return sum(divs) / len(divs)

def mean_distance(seq, seqs):
    divs = []
    for s in seqs:
        divs.append(distance(seq, s))
    return sum(divs) / len(divs)
class BaseOptimizerMulti:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.task = cfg.task
        self.max_oracle_calls = cfg.max_oracle_calls
        self.env_log_interval = cfg.env_log_interval
        
        self.dna_buffer = dict()
        self.mean_score = 0

        # Logging counters
        self.last_log = 0
        self.last_log_time = time.time()
        self.last_logging_time = time.time()
        self.total_count = 0
        self.invalid_count = 0
        self.redundant_count = 0
        self.oracle_type=cfg.oracle_type
        if cfg.wandb_log:
            wandb.init(
                project=cfg.project_name,
                name=cfg.wandb_run_name,
                reinit=True,
            )
        
        self.device = torch.device(cfg.device)
        self.experience = Experience(cfg.e_size, cfg.priority)
        
        # Load target models for multiple cell types
        if cfg.task in ['hepg2','k562','sknsh']:
            self.targets = {
                'hepg2': self.load_target_model('hepg2'),
                'k562': self.load_target_model('k562'),
                'sknsh': self.load_target_model('sknsh')
            }
        else:
            self.targets = {
                'JURKAT': self.load_target_model('JURKAT'),
                'K562': self.load_target_model('K562'),
                'THP1': self.load_target_model('THP1')
            }
        
        self.fitness_ranges = {cell: get_fitness_info(cell,self.oracle_type) for cell in self.targets.keys()}
    
    def load_target_model(self, cell):
        checkpoint_dir = getattr(self.cfg, 'checkpoint_dir', './checkpoints')

        model_path = {
            'hepg2': f'{checkpoint_dir}/human_regression_{self.oracle_type}_hepg2.ckpt',
            'k562': f'{checkpoint_dir}/human_regression_{self.oracle_type}_k562.ckpt',
            'sknsh': f'{checkpoint_dir}/human_regression_{self.oracle_type}_sknsh.ckpt',
            "JURKAT": f'{checkpoint_dir}/human_{self.oracle_type}_jurkat.ckpt',
            "K562": f'{checkpoint_dir}/human_{self.oracle_type}_k562.ckpt',
            "THP1": f'{checkpoint_dir}/human_{self.oracle_type}_THP1.ckpt',
        }
        model = src.reglm.regression.EnformerModel.load_from_checkpoint(
            model_path[cell], map_location=self.device
        ).to(self.device)
        model.eval()
        return model
    
    def normalize_target(self, score, cell):
        _, min_fitness, max_fitness = self.fitness_ranges[cell]
        return (score - min_fitness) / (max_fitness - min_fitness)
    
    @torch.no_grad()
    def score_enformer(self, dna):
        if len(self.dna_buffer) > self.max_oracle_calls:
            # Return zero tensor matching expected shape for consistency
            return torch.zeros(3)

        # Score each cell type explicitly by key (not relying on dict order)
        # This ensures consistent ordering: index 0=first cell, 1=second, 2=third
        scores_dict = {}
        for cell, model in self.targets.items():
            raw_score = model([dna]).squeeze(0).item()
            scores_dict[cell] = self.normalize_target(raw_score, cell)

        # Build scores list in explicit order based on task type
        if self.task in ['hepg2', 'k562', 'sknsh']:
            scores = [scores_dict['hepg2'], scores_dict['k562'], scores_dict['sknsh']]
        else:
            scores = [scores_dict['JURKAT'], scores_dict['K562'], scores_dict['THP1']]
        
        # Default reward: single ON target (index 0) vs two OFF constraints (indices 1, 2)
        # For original cells: ON=hepg2(0), OFF=k562(1), OFF=sknsh(2)
        # For immune cells (single-ON mode): ON=JURKAT(0), OFF=K562(1), OFF=THP1(2)
        # Note: DualOnOptimizer overrides this for dual-ON (JURKAT+THP1 vs K562)
        reward = 0.8*scores[0] - 0.1*scores[1] - 0.1*scores[2]
        
        if dna in self.dna_buffer:
            self.dna_buffer[dna][3] += 1  # Increment count (index 3), not order (index 2)
            self.redundant_count += 1
        else:
            #self.dna_buffer[dna] = [float(reward), len(self.dna_buffer) + 1, 1]
            self.dna_buffer[dna] = [torch.tensor(scores), reward,len(self.dna_buffer) + 1, 1]
        
        return self.dna_buffer[dna][0]
    
    def predict_enformer(self, dna_list):
        st = time.time()
        assert type(dna_list) == list
        self.total_count += len(dna_list)
        
        score_list = [self.score_enformer(dna) for dna in dna_list]
        
        if len(self.dna_buffer) % self.env_log_interval == 0 and len(self.dna_buffer) > self.last_log:
            self.sort_buffer()
            self.log_intermediate()
            self.last_log_time = time.time()
            self.last_log = len(self.dna_buffer)
        
        self.last_logging_time = time.time() - st
        self.mean_score = np.mean(score_list)
        return score_list
    def sort_buffer(self):
        
        self.dna_buffer = dict(sorted(self.dna_buffer.items(), key=lambda kv: kv[1][1], reverse=True))
            
    def log_intermediate(self, dna=None, scores=None, finish=False):
        if finish:
            temp_top100 = list(self.dna_buffer.items())[:100]
            dnas = [item[0] for item in temp_top100]
            scores = [item[1][1] for item in temp_top100]
            n_calls = self.max_oracle_calls
        
        else:
            if dna is None and scores is None:
                if len(self.dna_buffer) <= self.max_oracle_calls:
                    # If not spefcified, log current top-100 dna in buffer
                    temp_top100 = list(self.dna_buffer.items())[:100]
                    dnas = [item[0] for item in temp_top100]
                    scores = [item[1][1] for item in temp_top100]
                    n_calls = len(self.dna_buffer)
                else:
                    results = list(sorted(self.dna_buffer.items(), key=lambda kv: kv[1][1], reverse=False))[:self.max_oracle_calls]
                    temp_top100 = sorted(results, key=lambda kv: kv[1][0], reverse=True)[:100]
                    dnas = [item[0] for item in temp_top100]
                    scores = [item[1][1] for item in temp_top100]
                    n_calls = self.max_oracle_calls
            else:
                raise NotImplementedError
       
        avg_top1 = np.max(scores)
        avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
        avg_top100 = np.mean(scores)
