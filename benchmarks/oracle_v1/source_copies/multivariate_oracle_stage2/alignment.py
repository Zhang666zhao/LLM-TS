"""Load actual historical run configs, not library/default configurations."""
import hashlib
import json
from pathlib import Path


def historical_run(project, method, dataset):
    if method == 'dlinear':
        return Path(project)/'runs/20260922_DLinear_shared_ctx512_pred64'/dataset
    return Path(project)/'runs/20260926_baselines_ctx512_pred64/jobs'/f'{method}_{dataset}'/'attempt01'


def load_historical_config(project, method, dataset):
    source = historical_run(project, method, dataset)/'config.json'
    original = json.loads(source.read_text())
    if method == 'dlinear':
        config = dict(method=method, dataset=dataset, seq_len=original['context'],
            pred_len=original['horizon'], individual=False, batch=original['batch_size'],
            epochs=original['epochs'], patience=original['patience'], lr=original['learning_rate'],
            scheduler='type1', weight_decay=0., seed=original['seed'], workers=original['workers'],
            clip=None)
    else:
        config = dict(original)
        if original['batch_by_dataset'][dataset] != original['batch'] or original['effective_train_batch_by_dataset'][dataset] != original['batch']:
            raise ValueError('Historical microbatch differs; explicit accumulation support required')
        if original.get('smoke'):
            raise ValueError('Cannot use a smoke run as baseline')
        config['workers'] = 0
    if config['seq_len'] != 512 or config['pred_len'] != 64 or config['seed'] != 2021:
        raise ValueError('Unexpected historical benchmark')
    audit = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 original_config=original, resolved_training_config=config.copy(),
                 loader=dict(pin_memory=True, workers=config['workers'], independent_split_generators=True,
                             shuffle_train=True, drop_last_train=True, drop_last_eval=False, seed=2021),
                 deliberate_differences=['outcome-only supervision', 'outcome-validation-MSE selection',
                    'DLinear/PatchTST only receive outcome; other backbones retain multivariate input'],
                 old_selection='all-variable-validation-MSE', new_selection='outcome-validation-MSE',
                 claim='training hyperparameters aligned; NOT identical task or selection criterion')
    return config, audit


def parameter_hash(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
