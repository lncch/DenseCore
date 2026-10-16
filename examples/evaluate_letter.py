"""Open held-out test once, after every predeclared validation arm passes."""
import argparse
import json
from pathlib import Path
import numpy as np
from letter_model import build,metrics,weight_hash
from letter_data import verify_cache


def evaluate(cache,results):
    results=Path(results);destination=results/'heldout-test.json'
    if destination.exists():raise RuntimeError('held-out evaluation already recorded; do not repeat selection on test')
    engines=['densecore','densecore_compiled','torch_eager']
    arms=[]
    for width in [32,128]:
        for engine in engines:
            quality=json.loads((results/f'{engine}-{width}-quality.json').read_text())
            if not quality['validation_gate']:raise RuntimeError('validation gate failed; test remains unopened')
            arms.append(quality)
    manifest=arms[0]['dataset']
    if any(arm['dataset']!=manifest for arm in arms):raise ValueError('quality arms use different dataset manifests')
    identity=verify_cache(cache,manifest)
    marker=results/'heldout-evaluation-started.json'
    with marker.open('x') as file:json.dump(dict(dataset_identity=identity,status='started'),file,indent=2)
    with np.load(Path(cache)/'prepared.npz',allow_pickle=False) as dataset:
        ids=dataset['test'];x=dataset['x'][ids];y=dataset['y'][ids]
    rows=[]
    for quality in arms:
        engine,width=quality['engine'],quality['width']
        with np.load(results/f'{engine}-{width}-weights.npz',allow_pickle=False) as archive:weights={k:archive[k] for k in archive.files}
        if weight_hash(weights)!=quality['best_weight_hash']:raise ValueError('selected checkpoint hash changed')
        model=build(engine,width,weights)
        if engine.startswith('densecore'):
            import densecore as d
            with d.no_grad():logits=model(d.tensor(x)).numpy()
        else:
            import torch
            torch.set_num_threads(1)
            with torch.no_grad():logits=model(torch.from_numpy(x.copy())).numpy()
        rows.append(dict(engine=engine,width=width,selected_epoch=quality['best_epoch'],**metrics(logits,y)))
    checks=[]
    for width in [32,128]:
        reference=next(r for r in rows if r['engine']=='torch_eager' and r['width']==width)
        for row in [r for r in rows if r['width']==width and r['engine']!='torch_eager']:
            checks.append(dict(engine=row['engine'],width=width,accuracy_gap=abs(row['accuracy']-reference['accuracy']),macro_f1_gap=abs(row['macro_f1']-reference['macro_f1']),passed=abs(row['accuracy']-reference['accuracy'])<=.005 and abs(row['macro_f1']-reference['macro_f1'])<=.01))
    result=dict(dataset_identity=identity,test_rows=len(y),test_used_for_selection=False,accepted=all(c['passed'] for c in checks),rows=rows,checks=checks)
    destination.write_text(json.dumps(result,indent=2)+'\n')
    if not result['accepted']:raise RuntimeError('held-out equivalence failed; results retained without retuning on test')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=Path(__file__).parents[1]/'data/letter');p.add_argument('--results',type=Path,default=Path(__file__).parents[1]/'benchmarks/letter-quality');a=p.parse_args();print(json.dumps(evaluate(a.cache,a.results),indent=2))
