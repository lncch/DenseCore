import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('letter_data',Path(__file__).parents[1]/'examples/letter_data.py')
data=importlib.util.module_from_spec(spec);spec.loader.exec_module(data)


def test_grouped_split_preserves_duplicate_groups_and_all_classes():
    features=np.repeat(np.arange(60)[:,None],2,axis=0)
    labels=np.repeat(np.arange(60)%3,2)
    splits,groups=data.grouped_split(features,labels)
    assert sorted(np.concatenate(splits).tolist())==list(range(120))
    for a in range(3):
        assert set(labels[splits[a]])=={0,1,2}
        for b in range(a+1,3):assert not set(groups[splits[a]])&set(groups[splits[b]])
    again,_=data.grouped_split(features,labels)
    for a,b in zip(splits,again):np.testing.assert_array_equal(a,b)


def test_scaler_uses_training_rows_only_and_rejects_conflicting_groups():
    features=np.array([[1.,2.],[3.,2.],[1000.,2000.]])
    scaled,mean,scale=data.standardize(features,np.array([0,1]))
    np.testing.assert_array_equal(mean,[2.,2.]);np.testing.assert_array_equal(scale,[1.,1.])
    np.testing.assert_array_equal(scaled[:2],[[-1.,0.],[1.,0.]])
    with pytest.raises(ValueError,match='conflicting'):data.grouped_split(np.array([[1],[1]]),np.array([0,1]))


def test_prepared_identity_rejects_changed_splits_scaler_features_and_labels():
    features=np.arange(120,dtype=np.int16).reshape(60,2)
    labels=np.arange(60)%3
    indices,groups=data.grouped_split(features,labels)
    x,mean,scale=data.standardize(features,indices[0])
    arrays=dict(x=x,y=labels,groups=groups,train=indices[0],validation=indices[1],test=indices[2],mean=mean,scale=scale)
    identity=data.verify_arrays(features,labels,arrays)
    assert 'feature_hash' in identity and 'label_hash' in identity
    for key in ['test','mean','x','y']:
        changed={k:v.copy() for k,v in arrays.items()};changed[key].flat[0]+=1
        with pytest.raises(ValueError,match='prepared'):data.verify_arrays(features,labels,changed)
