"""Pinned UCI Letter Recognition data with duplicate-safe grouped splits."""
from pathlib import Path
import hashlib
import io
import json
import urllib.request
import urllib.error
import ssl
import subprocess
import zipfile
import numpy as np

URL='https://archive.ics.uci.edu/static/public/59/letter%2Brecognition.zip'
ZIP_SHA='3b5f07a334697b6cace4fbae22940393a18fee596e73f68d97ce5973d52dc60f'
RAW_SHA='2b89f3602cf768d3c8355267d2f13f2417809e101fc2b5ceee10db19a60de6e2'
SEED=20261014


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def grouped_split(features,labels,seed=SEED):
    _,groups=np.unique(features,axis=0,return_inverse=True)
    group_labels=np.empty(groups.max()+1,dtype=np.int64)
    for group in range(len(group_labels)):
        values=np.unique(labels[groups==group])
        if len(values)!=1:raise ValueError('identical features have conflicting labels')
        group_labels[group]=values[0]
    assignments=np.empty(len(group_labels),dtype=np.int8)
    rng=np.random.default_rng(seed)
    for label in np.unique(labels):
        ids=np.flatnonzero(group_labels==label);rng.shuffle(ids)
        if len(ids)<3:raise ValueError('every class needs at least three feature groups')
        validation=max(1,int(round(len(ids)*.1)));test=max(1,int(round(len(ids)*.1)))
        train=len(ids)-validation-test
        if train<1:raise ValueError('not enough training groups')
        assignments[ids[:train]]=0;assignments[ids[train:train+validation]]=1;assignments[ids[train+validation:]]=2
    split=assignments[groups]
    indices=tuple(np.flatnonzero(split==i) for i in range(3))
    for a in range(3):
        for b in range(a+1,3):
            assert not np.intersect1d(groups[indices[a]],groups[indices[b]]).size
    return indices,groups


def standardize(features,train_indices):
    training=features[train_indices].astype(np.float64)
    mean=training.mean(axis=0).astype(np.float32)
    scale=training.std(axis=0).astype(np.float32);scale[scale==0]=1.
    return np.ascontiguousarray((features.astype(np.float32)-mean)/scale),mean,scale


def verify_arrays(features,labels,arrays):
    indices,groups=grouped_split(features,labels)
    x,mean,scale=standardize(features,indices[0])
    expected=dict(x=x,y=labels,groups=groups,train=indices[0],validation=indices[1],test=indices[2],mean=mean,scale=scale)
    for key,value in expected.items():
        if key not in arrays or arrays[key].dtype!=value.dtype or not np.array_equal(arrays[key],value):
            raise ValueError('prepared dataset identity mismatch: '+key)
    return dict(feature_hash=digest(x),label_hash=digest(labels),group_hash=digest(groups),split_hashes=[digest(i) for i in indices],scaler_hash=digest(np.concatenate([mean,scale])))


def verify_cache(cache,manifest):
    cache=Path(cache);raw=(cache/'letter-recognition.data').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=RAW_SHA or manifest['raw_sha256']!=RAW_SHA:
        raise ValueError('prepared source identity mismatch')
    rows=[line.split(',') for line in raw.decode('ascii').strip().splitlines()]
    features=np.array([[int(v) for v in row[1:]] for row in rows],dtype=np.int16)
    labels=np.array([ord(row[0])-ord('A') for row in rows],dtype=np.int64)
    with np.load(cache/'prepared.npz',allow_pickle=False) as dataset:
        arrays={k:dataset[k] for k in dataset.files}
    identity=verify_arrays(features,labels,arrays)
    if manifest['seed']!=SEED or manifest['split_hashes']!=identity['split_hashes'] or manifest['scaler_mean']!=arrays['mean'].tolist() or manifest['scaler_scale']!=arrays['scale'].tolist():
        raise ValueError('prepared training manifest mismatch')
    return dict(raw_sha256=RAW_SHA,**identity)


def prepare(cache):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    raw_path=cache/'letter-recognition.data'
    if not raw_path.exists():
        zipped=cache/'letter.zip'
        if zipped.exists():
            archive=zipped.read_bytes()
        else:
            try:
                with urllib.request.urlopen(URL,timeout=60) as response:archive=response.read()
            except urllib.error.URLError as error:
                if not isinstance(error.reason,ssl.SSLCertVerificationError):raise
                # curl uses the system trust store on macOS. TLS verification
                # remains enabled; the archive hash is checked below as well.
                archive=subprocess.run(['curl','--fail','--silent','--show-error','--location','--max-time','60',URL],capture_output=True,check=True).stdout
        if hashlib.sha256(archive).hexdigest()!=ZIP_SHA:raise ValueError('dataset ZIP hash changed')
        with zipfile.ZipFile(io.BytesIO(archive)) as z:raw=z.read('letter-recognition.data')
        if hashlib.sha256(raw).hexdigest()!=RAW_SHA:raise ValueError('dataset data hash changed')
        raw_path.write_bytes(raw)
    raw=raw_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=RAW_SHA:raise ValueError('cached dataset hash mismatch')
    rows=[line.split(',') for line in raw.decode('ascii').strip().splitlines()]
    if len(rows)!=20000 or any(len(row)!=17 for row in rows):raise ValueError('unexpected dataset dimensions')
    labels=np.array([ord(row[0])-ord('A') for row in rows],dtype=np.int64)
    features=np.array([[int(v) for v in row[1:]] for row in rows],dtype=np.int16)
    if set(labels)!=set(range(26)) or features.min()<0 or features.max()>15:raise ValueError('unexpected dataset values')
    indices,groups=grouped_split(features,labels)
    x,mean,scale=standardize(features,indices[0])
    manifest=dict(dataset='UCI Letter Recognition',doi='10.24432/C5ZP40',license='CC BY 4.0',source=URL,raw_sha256=RAW_SHA,zip_sha256=ZIP_SHA,
                  seed=SEED,rows=len(labels),unique_feature_groups=len(np.unique(groups)),duplicate_rows=len(labels)-len(np.unique(groups)),
                  split_sizes=[len(i) for i in indices],split_hashes=[digest(i) for i in indices],scaler_mean=mean.tolist(),scaler_scale=scale.tolist())
    np.savez_compressed(cache/'prepared.npz',x=x,y=labels,groups=groups,train=indices[0],validation=indices[1],test=indices[2],mean=mean,scale=scale)
    (cache/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('cache',type=Path);args=p.parse_args();print(json.dumps(prepare(args.cache),indent=2))
