---
published: true
layout: posts
title: '[DL4C] Ch6. Multi Categorical Classification'
categories: 
  - dl4coders
---



```python
#hide
! [ -e /content ] && pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
```


```python
#hide
from fastbook import *
```

# Other Computer Vision Problems

## Multi-Label Classification

### The Data


```python
from fastai.vision.all import *
path = untar_data(URLs.PASCAL_2007)
```


```python
df = pd.read_csv(path/'train.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fname</th>
      <th>labels</th>
      <th>is_valid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000005.jpg</td>
      <td>chair</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000007.jpg</td>
      <td>car</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>000009.jpg</td>
      <td>horse person</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>000012.jpg</td>
      <td>car</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>000016.jpg</td>
      <td>bicycle</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fname</th>
      <th>labels</th>
      <th>is_valid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000005.jpg</td>
      <td>chair</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000007.jpg</td>
      <td>car</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>000009.jpg</td>
      <td>horse person</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>000012.jpg</td>
      <td>car</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>000016.jpg</td>
      <td>bicycle</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



### Sidebar: Pandas and DataFrames


```python
df.iloc[:,0]
```




    0       000005.jpg
    1       000007.jpg
    2       000009.jpg
    3       000012.jpg
    4       000016.jpg
               ...    
    5006    009954.jpg
    5007    009955.jpg
    5008    009958.jpg
    5009    009959.jpg
    5010    009961.jpg
    Name: fname, Length: 5011, dtype: object




```python
df.iloc[0,:]
# Trailing :s are always optional (in numpy, pytorch, pandas, etc.),
#   so this is equivalent:
df.iloc[0]
```




    fname       000005.jpg
    labels           chair
    is_valid          True
    Name: 0, dtype: object




```python
df['fname']
```




    0       000005.jpg
    1       000007.jpg
    2       000009.jpg
    3       000012.jpg
    4       000016.jpg
               ...    
    5006    009954.jpg
    5007    009955.jpg
    5008    009958.jpg
    5009    009959.jpg
    5010    009961.jpg
    Name: fname, Length: 5011, dtype: object




```python
tmp_df = pd.DataFrame({'a':[1,2], 'b':[3,4]})
tmp_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
tmp_df['c'] = tmp_df['a']+tmp_df['b']
tmp_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



### End sidebar

### Constructing a DataBlock


```python
# data block created with no parameters
dblock = DataBlock()
```


```python
# create Datasets objects from 'source' params
dsets = dblock.datasets(df)
```


```python
len(dsets.train),len(dsets.valid)
```




    (4009, 1002)




```python
dsets.train[0]
```




    (fname       008663.jpg
     labels      car person
     is_valid         False
     Name: 4346, dtype: object,
     fname       008663.jpg
     labels      car person
     is_valid         False
     Name: 4346, dtype: object)




```python
dsets.valid[0]
```




    (fname          002613.jpg
     labels      bottle person
     is_valid             True
     Name: 1311, dtype: object,
     fname          002613.jpg
     labels      bottle person
     is_valid             True
     Name: 1311, dtype: object)




```python
x,y = dsets.train[0]
x,y
```




    (fname       008663.jpg
     labels      car person
     is_valid         False
     Name: 4346, dtype: object,
     fname       008663.jpg
     labels      car person
     is_valid         False
     Name: 4346, dtype: object)




```python
x['fname']
```




    '008663.jpg'




```python
# data block -> assume we have two things (input and target)
dblock = DataBlock(get_x = lambda r: r['fname'], get_y = lambda r: r['labels'])
dsets = dblock.datasets(df)
dsets.train[0]
```




    ('009546.jpg', 'sofa person')




```python
# identical to above approach
# BUT, use more verbose approach when exporting your Learner after training

def get_x(r): return r['fname']
def get_y(r): return r['labels']

dblock = DataBlock(get_x = get_x, get_y = get_y)
dsets = dblock.datasets(df)
dsets.train[0]
```




    ('005620.jpg', 'aeroplane')




```python
path
```




    Path('/Users/ridealist/.fastai/data/pascal_2007')




```python
def get_x(r): return path/'train'/r['fname']
def get_y(r): return r['labels'].split(' ')

dblock = DataBlock(get_x = get_x, get_y = get_y)
dsets = dblock.datasets(df)
dsets.train[0]
```




    (Path('/Users/ridealist/.fastai/data/pascal_2007/train/002549.jpg'),
     ['tvmonitor'])




```python
# MultiCategoryBlock -> One-Hot Encoding

dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   get_x = get_x, get_y = get_y)
dsets = dblock.datasets(df)
dsets.train[0], type(dsets.train[0])
```




    ((PILImage mode=RGB size=500x325,
      TensorMultiCategory([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.])),
     tuple)




```python
dblock.summary(df)
```

    Setting-up type transforms pipelines
    Collecting items from            fname          labels  is_valid
    0     000005.jpg           chair      True
    1     000007.jpg             car      True
    2     000009.jpg    horse person      True
    3     000012.jpg             car     False
    4     000016.jpg         bicycle      True
    ...          ...             ...       ...
    5006  009954.jpg    horse person      True
    5007  009955.jpg            boat      True
    5008  009958.jpg  person bicycle      True
    5009  009959.jpg             car     False
    5010  009961.jpg             dog     False
    
    [5011 rows x 3 columns]
    Found 5011 items
    2 datasets of sizes 4009,1002
    Setting up Pipeline: get_x -> PILBase.create
    Setting up Pipeline: get_y -> MultiCategorize -- {'vocab': None, 'sort': True, 'add_na': False} -> OneHotEncode -- {'c': None}
    
    Building one sample
      Pipeline: get_x -> PILBase.create
        starting from
          fname       002546.jpg
    labels             dog
    is_valid          True
    Name: 1277, dtype: object
        applying get_x gives
          /Users/ridealist/.fastai/data/pascal_2007/train/002546.jpg
        applying PILBase.create gives
          PILImage mode=RGB size=500x375
      Pipeline: get_y -> MultiCategorize -- {'vocab': None, 'sort': True, 'add_na': False} -> OneHotEncode -- {'c': None}
        starting from
          fname       002546.jpg
    labels             dog
    is_valid          True
    Name: 1277, dtype: object
        applying get_y gives
          [dog]
        applying MultiCategorize -- {'vocab': None, 'sort': True, 'add_na': False} gives
          TensorMultiCategory([11])
        applying OneHotEncode -- {'c': None} gives
          TensorMultiCategory([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
    
    Final sample: (PILImage mode=RGB size=500x375, TensorMultiCategory([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]))
    
    
    Collecting items from            fname          labels  is_valid
    0     000005.jpg           chair      True
    1     000007.jpg             car      True
    2     000009.jpg    horse person      True
    3     000012.jpg             car     False
    4     000016.jpg         bicycle      True
    ...          ...             ...       ...
    5006  009954.jpg    horse person      True
    5007  009955.jpg            boat      True
    5008  009958.jpg  person bicycle      True
    5009  009959.jpg             car     False
    5010  009961.jpg             dog     False
    
    [5011 rows x 3 columns]
    Found 5011 items
    2 datasets of sizes 4009,1002
    Setting up Pipeline: get_x -> PILBase.create
    Setting up Pipeline: get_y -> MultiCategorize -- {'vocab': None, 'sort': True, 'add_na': False} -> OneHotEncode -- {'c': None}
    Setting up after_item: Pipeline: ToTensor
    Setting up before_batch: Pipeline: 
    Setting up after_batch: Pipeline: IntToFloatTensor -- {'div': 255.0, 'div_mask': 1}
    
    Building one batch
    Applying item_tfms to the first sample:
      Pipeline: ToTensor
        starting from
          (PILImage mode=RGB size=500x375, TensorMultiCategory([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]))
        applying ToTensor gives
          (TensorImage of size 3x375x500, TensorMultiCategory([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]))
    
    Adding the next 3 samples
    
    No before_batch transform to apply
    
    Collating items in a batch
    Error! It's not possible to collate your items in a batch
    Could not collate the 0-th members of your tuples because got the following shapes
    torch.Size([3, 375, 500]),torch.Size([3, 333, 500]),torch.Size([3, 363, 500]),torch.Size([3, 334, 500])



    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Cell In[19], line 1
    ----> 1 dblock.summary(df)


    File ~/miniconda3/envs/fastbook/lib/python3.9/site-packages/fastai/data/block.py:237, in summary(self, source, bs, show_batch, **kwargs)
        235     why = _find_fail_collate(s)
        236     print("Make sure all parts of your samples are tensors of the same size" if why is None else why)
    --> 237     raise e
        239 if len([f for f in dls.train.after_batch.fs if f.name != 'noop'])!=0:
        240     print("\nApplying batch_tfms to the batch built")


    File ~/miniconda3/envs/fastbook/lib/python3.9/site-packages/fastai/data/block.py:231, in summary(self, source, bs, show_batch, **kwargs)
        229 print("\nCollating items in a batch")
        230 try:
    --> 231     b = dls.train.create_batch(s)
        232     b = retain_types(b, s[0] if is_listy(s) else s)
        233 except Exception as e:


    File ~/miniconda3/envs/fastbook/lib/python3.9/site-packages/fastai/data/load.py:166, in DataLoader.create_batch(self, b)
        164 try: return (fa_collate,fa_convert)[self.prebatched](b)
        165 except Exception as e: 
    --> 166     if not self.prebatched: collate_error(e,b)
        167     raise


    File ~/miniconda3/envs/fastbook/lib/python3.9/site-packages/fastai/data/load.py:164, in DataLoader.create_batch(self, b)
        163 def create_batch(self, b): 
    --> 164     try: return (fa_collate,fa_convert)[self.prebatched](b)
        165     except Exception as e: 
        166         if not self.prebatched: collate_error(e,b)


    File ~/miniconda3/envs/fastbook/lib/python3.9/site-packages/fastai/data/load.py:52, in fa_collate(t)
         49 "A replacement for PyTorch `default_collate` which maintains types and handles `Sequence`s"
         50 b = t[0]
         51 return (default_collate(t) if isinstance(b, _collate_types)
    ---> 52         else type(t[0])([fa_collate(s) for s in zip(*t)]) if isinstance(b, Sequence)
         53         else default_collate(t))


    File ~/miniconda3/envs/fastbook/lib/python3.9/site-packages/fastai/data/load.py:52, in <listcomp>(.0)
         49 "A replacement for PyTorch `default_collate` which maintains types and handles `Sequence`s"
         50 b = t[0]
         51 return (default_collate(t) if isinstance(b, _collate_types)
    ---> 52         else type(t[0])([fa_collate(s) for s in zip(*t)]) if isinstance(b, Sequence)
         53         else default_collate(t))


    File ~/miniconda3/envs/fastbook/lib/python3.9/site-packages/fastai/data/load.py:51, in fa_collate(t)
         49 "A replacement for PyTorch `default_collate` which maintains types and handles `Sequence`s"
         50 b = t[0]
    ---> 51 return (default_collate(t) if isinstance(b, _collate_types)
         52         else type(t[0])([fa_collate(s) for s in zip(*t)]) if isinstance(b, Sequence)
         53         else default_collate(t))


    File ~/miniconda3/envs/fastbook/lib/python3.9/site-packages/torch/utils/data/_utils/collate.py:264, in default_collate(batch)
        203 def default_collate(batch):
        204     r"""
        205         Function that takes in a batch of data and puts the elements within the batch
        206         into a tensor with an additional outer dimension - batch size. The exact output type can be
       (...)
        262             >>> default_collate(batch)  # Handle `CustomType` automatically
        263     """
    --> 264     return collate(batch, collate_fn_map=default_collate_fn_map)


    File ~/miniconda3/envs/fastbook/lib/python3.9/site-packages/torch/utils/data/_utils/collate.py:123, in collate(batch, collate_fn_map)
        121     for collate_type in collate_fn_map:
        122         if isinstance(elem, collate_type):
    --> 123             return collate_fn_map[collate_type](batch, collate_fn_map=collate_fn_map)
        125 if isinstance(elem, collections.abc.Mapping):
        126     try:


    File ~/miniconda3/envs/fastbook/lib/python3.9/site-packages/torch/utils/data/_utils/collate.py:162, in collate_tensor_fn(batch, collate_fn_map)
        160     storage = elem._typed_storage()._new_shared(numel, device=elem.device)
        161     out = elem.new(storage).resize_(len(batch), *list(elem.size()))
    --> 162 return torch.stack(batch, 0, out=out)


    File ~/miniconda3/envs/fastbook/lib/python3.9/site-packages/fastai/torch_core.py:382, in TensorBase.__torch_function__(cls, func, types, args, kwargs)
        380 if cls.debug and func.__name__ not in ('__str__','__repr__'): print(func, types, args, kwargs)
        381 if _torch_handled(args, cls._opt, func): types = (torch.Tensor,)
    --> 382 res = super().__torch_function__(func, types, args, ifnone(kwargs, {}))
        383 dict_objs = _find_args(args) if args else _find_args(list(kwargs.values()))
        384 if issubclass(type(res),TensorBase) and dict_objs: res.set_meta(dict_objs[0],as_copy=True)


    File ~/miniconda3/envs/fastbook/lib/python3.9/site-packages/torch/_tensor.py:1295, in Tensor.__torch_function__(cls, func, types, args, kwargs)
       1292     return NotImplemented
       1294 with _C.DisableTorchFunctionSubclass():
    -> 1295     ret = func(*args, **kwargs)
       1296     if func in get_default_nowrap_functions():
       1297         return ret


    RuntimeError: Error when trying to collate the data into batches with fa_collate, at least two tensors in the batch are not the same size.
    
    Mismatch found on axis 0 of the batch and is of type `TensorImage`:
    	Item at index 0 has shape: torch.Size([3, 375, 500])
    	Item at index 1 has shape: torch.Size([3, 333, 500])
    
    Please include a transform in `after_item` that ensures all data of type TensorImage is the same size



```python
idxs = torch.where(dsets.train[0][1]==1.); idxs
```




    (TensorMultiCategory([6]),)




```python
idxs = torch.where(dsets.train[0][1]==1.)[0] ; idxs
# dsets.train.vocab[idxs]
```




    TensorMultiCategory([6])




```python
dsets.train.vocab
```




    ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']




```python
dsets.train.vocab[idxs]
```




    (#1) ['car']




```python
len(dsets.train.vocab)
```




    20




```python
def splitter(df):
    train = df.index[~df['is_valid']].tolist()
    valid = df.index[df['is_valid']].tolist()
    return train,valid

dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=splitter,
                   get_x=get_x, 
                   get_y=get_y)

dsets = dblock.datasets(df)
dsets.train[0]
```




    (PILImage mode=RGB size=500x333,
     TensorMultiCategory([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))




```python
dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=splitter,
                   get_x=get_x, 
                   get_y=get_y,
                   item_tfms = RandomResizedCrop(128, min_scale=0.35))

dls = dblock.dataloaders(df)
```


```python
dls.show_batch(nrows=1, ncols=3)
```


    
![png](06_multicat_files/06_multicat_36_0.png)
    


### Remember "summary" method


```python
dblock.summary(df)
```

    Setting-up type transforms pipelines
    Collecting items from            fname          labels  is_valid
    0     000005.jpg           chair      True
    1     000007.jpg             car      True
    2     000009.jpg    horse person      True
    3     000012.jpg             car     False
    4     000016.jpg         bicycle      True
    ...          ...             ...       ...
    5006  009954.jpg    horse person      True
    5007  009955.jpg            boat      True
    5008  009958.jpg  person bicycle      True
    5009  009959.jpg             car     False
    5010  009961.jpg             dog     False
    
    [5011 rows x 3 columns]
    Found 5011 items
    2 datasets of sizes 2501,2510
    Setting up Pipeline: get_x -> PILBase.create
    Setting up Pipeline: get_y -> MultiCategorize -- {'vocab': None, 'sort': True, 'add_na': False} -> OneHotEncode -- {'c': None}
    
    Building one sample
      Pipeline: get_x -> PILBase.create
        starting from
          fname       000012.jpg
    labels             car
    is_valid         False
    Name: 3, dtype: object
        applying get_x gives
          /Users/ridealist/.fastai/data/pascal_2007/train/000012.jpg
        applying PILBase.create gives
          PILImage mode=RGB size=500x333
      Pipeline: get_y -> MultiCategorize -- {'vocab': None, 'sort': True, 'add_na': False} -> OneHotEncode -- {'c': None}
        starting from
          fname       000012.jpg
    labels             car
    is_valid         False
    Name: 3, dtype: object
        applying get_y gives
          [car]
        applying MultiCategorize -- {'vocab': None, 'sort': True, 'add_na': False} gives
          TensorMultiCategory([6])
        applying OneHotEncode -- {'c': None} gives
          TensorMultiCategory([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    
    Final sample: (PILImage mode=RGB size=500x333, TensorMultiCategory([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
    
    
    Collecting items from            fname          labels  is_valid
    0     000005.jpg           chair      True
    1     000007.jpg             car      True
    2     000009.jpg    horse person      True
    3     000012.jpg             car     False
    4     000016.jpg         bicycle      True
    ...          ...             ...       ...
    5006  009954.jpg    horse person      True
    5007  009955.jpg            boat      True
    5008  009958.jpg  person bicycle      True
    5009  009959.jpg             car     False
    5010  009961.jpg             dog     False
    
    [5011 rows x 3 columns]
    Found 5011 items
    2 datasets of sizes 2501,2510
    Setting up Pipeline: get_x -> PILBase.create
    Setting up Pipeline: get_y -> MultiCategorize -- {'vocab': None, 'sort': True, 'add_na': False} -> OneHotEncode -- {'c': None}
    Setting up after_item: Pipeline: RandomResizedCrop -- {'size': (128, 128), 'min_scale': 0.35, 'ratio': (0.75, 1.3333333333333333), 'resamples': (<Resampling.BILINEAR: 2>, <Resampling.NEAREST: 0>), 'val_xtra': 0.14, 'max_scale': 1.0, 'p': 1.0} -> ToTensor
    Setting up before_batch: Pipeline: 
    Setting up after_batch: Pipeline: IntToFloatTensor -- {'div': 255.0, 'div_mask': 1}
    
    Building one batch
    Applying item_tfms to the first sample:
      Pipeline: RandomResizedCrop -- {'size': (128, 128), 'min_scale': 0.35, 'ratio': (0.75, 1.3333333333333333), 'resamples': (<Resampling.BILINEAR: 2>, <Resampling.NEAREST: 0>), 'val_xtra': 0.14, 'max_scale': 1.0, 'p': 1.0} -> ToTensor
        starting from
          (PILImage mode=RGB size=500x333, TensorMultiCategory([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
        applying RandomResizedCrop -- {'size': (128, 128), 'min_scale': 0.35, 'ratio': (0.75, 1.3333333333333333), 'resamples': (<Resampling.BILINEAR: 2>, <Resampling.NEAREST: 0>), 'val_xtra': 0.14, 'max_scale': 1.0, 'p': 1.0} gives
          (PILImage mode=RGB size=128x128, TensorMultiCategory([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
        applying ToTensor gives
          (TensorImage of size 3x128x128, TensorMultiCategory([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
    
    Adding the next 3 samples
    
    No before_batch transform to apply
    
    Collating items in a batch
    
    Applying batch_tfms to the batch built
      Pipeline: IntToFloatTensor -- {'div': 255.0, 'div_mask': 1}
        starting from
          (TensorImage of size 4x3x128x128, TensorMultiCategory of size 4x20)
        applying IntToFloatTensor -- {'div': 255.0, 'div_mask': 1} gives
          (TensorImage of size 4x3x128x128, TensorMultiCategory of size 4x20)


### Binary Cross-Entropy

### Learner contains 4 main things

1. Model
2. DataLoaders object
3. Optimizer
4. loss function


```python
learn = vision_learner(dls, resnet18)
# resnet18 / dls / SGD / (    )
```

    /Users/ridealist/miniconda3/envs/fastbook/lib/python3.9/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
      warnings.warn(
    /Users/ridealist/miniconda3/envs/fastbook/lib/python3.9/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
      warnings.warn(msg)



```python
dls.train.one_batch()
```




    (TensorImage([[[[-0.6281, -0.7650, -1.0904,  ..., -0.9534, -0.9363, -0.9192],
                    [ 0.1939,  0.1254, -0.0972,  ..., -0.9534, -0.9363, -0.9363],
                    [ 0.8961,  1.3070,  1.2043,  ..., -0.9363, -0.9192, -0.9192],
                    ...,
                    [ 0.2624, -0.0972, -1.6727,  ..., -0.8164, -0.8849, -1.2274],
                    [ 0.3652, -1.0390, -1.6727,  ..., -0.6965, -0.7308, -0.8335],
                    [ 0.4508, -0.6794, -1.7069,  ..., -0.7479, -0.7650, -0.7650]],
     
                   [[ 0.0826,  0.0126, -0.1800,  ..., -0.0399, -0.0224, -0.0224],
                    [ 0.5028,  0.4678,  0.3277,  ..., -0.0049, -0.0224, -0.0224],
                    [ 1.0455,  1.4482,  1.3782,  ..., -0.0049, -0.0049, -0.0049],
                    ...,
                    [ 0.3627, -0.0049, -1.5280,  ..., -0.7752, -0.7927, -1.1253],
                    [ 0.4328, -0.9328, -1.5455,  ..., -0.6702, -0.6702, -0.7227],
                    [ 0.5378, -0.3725, -1.5455,  ..., -0.7402, -0.7402, -0.7052]],
     
                   [[ 1.0539,  0.9668,  0.8274,  ...,  1.2108,  1.2282,  1.2282],
                    [ 0.9494,  0.9319,  0.8622,  ...,  1.2282,  1.2282,  1.2282],
                    [ 1.3154,  1.6640,  1.6291,  ...,  1.1934,  1.2108,  1.2282],
                    ...,
                    [ 0.4439,  0.1476, -1.3164,  ..., -0.6367, -0.6367, -0.8807],
                    [ 0.5485, -0.7936, -1.3164,  ..., -0.5147, -0.5147, -0.5670],
                    [ 0.6531, -0.2881, -1.3164,  ..., -0.5670, -0.6018, -0.5495]]],
     
     
                  [[[-1.0562, -1.0733, -0.9877,  ..., -1.9124, -1.8953, -1.7583],
                    [-1.1418, -1.0219, -0.9705,  ..., -1.7925, -1.7925, -1.8268],
                    [-1.2959, -1.0562, -0.9363,  ..., -1.8097, -1.7240, -1.7069],
                    ...,
                    [ 0.3309,  0.3994,  0.4337,  ...,  0.7762,  0.7248,  0.6734],
                    [ 0.3481,  0.4337,  0.4679,  ...,  0.7933,  0.7419,  0.7077],
                    [ 0.3994,  0.5193,  0.4166,  ...,  0.7591,  0.7419,  0.7077]],
     
                   [[-0.9328, -0.9678, -0.8803,  ..., -1.8957, -1.8782, -1.7381],
                    [-1.0203, -0.9328, -0.8627,  ..., -1.7556, -1.7731, -1.8081],
                    [-1.1954, -0.9678, -0.8452,  ..., -1.7906, -1.7381, -1.7381],
                    ...,
                    [ 0.2402,  0.2402,  0.2577,  ...,  0.8179,  0.7304,  0.5903],
                    [ 0.2402,  0.2927,  0.3277,  ...,  0.8704,  0.7829,  0.6604],
                    [ 0.2577,  0.3803,  0.3627,  ...,  0.8704,  0.8004,  0.6954]],
     
                   [[-0.7936, -0.8284, -0.7413,  ..., -1.6999, -1.6824, -1.5604],
                    [-0.8807, -0.7761, -0.7413,  ..., -1.6127, -1.6127, -1.6476],
                    [-1.0376, -0.8284, -0.6890,  ..., -1.6302, -1.5779, -1.5779],
                    ...,
                    [ 0.3045,  0.2522,  0.2522,  ...,  0.9668,  0.8622,  0.7054],
                    [ 0.3045,  0.3045,  0.3568,  ...,  1.0191,  0.9319,  0.7925],
                    [ 0.3219,  0.4091,  0.4091,  ...,  1.0191,  0.9668,  0.8448]]],
     
     
                  [[[-0.5596, -0.6281, -0.6452,  ..., -1.1247, -1.0904, -1.0733],
                    [ 0.0227, -0.5424, -0.4911,  ..., -1.1589, -1.0048, -1.0219],
                    [ 0.4851, -0.1999,  0.2624,  ..., -1.2103, -1.0562, -0.9534],
                    ...,
                    [-2.0837, -2.0837, -2.0494,  ..., -1.7069, -1.8782, -1.9809],
                    [-2.0837, -2.0837, -2.0494,  ..., -1.7583, -1.5870, -1.4843],
                    [-2.0837, -2.0837, -2.0494,  ..., -2.0494, -2.0494, -1.8782]],
     
                   [[-0.3025, -0.3725, -0.4076,  ..., -1.0028, -0.9328, -0.8978],
                    [ 0.2227, -0.3725, -0.3025,  ..., -0.9678, -0.8102, -0.8277],
                    [ 0.7654,  0.0301,  0.5203,  ..., -0.9853, -0.8627, -0.7577],
                    ...,
                    [-2.0007, -2.0007, -1.9657,  ..., -1.6681, -1.8256, -1.9132],
                    [-2.0007, -2.0007, -1.9657,  ..., -1.7031, -1.5105, -1.4055],
                    [-2.0007, -2.0007, -1.9657,  ..., -1.9657, -1.9657, -1.7731]],
     
                   [[ 0.0779, -0.0092, -0.0790,  ..., -0.2532, -0.1312, -0.0964],
                    [ 0.6008,  0.0431,  0.0256,  ..., -0.1661, -0.0267, -0.0790],
                    [ 1.1237,  0.4439,  0.9145,  ..., -0.2184, -0.0615,  0.0082],
                    ...,
                    [-1.7696, -1.7696, -1.7347,  ..., -1.4559, -1.6127, -1.6999],
                    [-1.7696, -1.7696, -1.7347,  ..., -1.4907, -1.2816, -1.1596],
                    [-1.7696, -1.7696, -1.7347,  ..., -1.7347, -1.7347, -1.5430]]],
     
     
                  ...,
     
     
                  [[[ 0.1083,  0.0912,  0.0056,  ...,  0.6734,  0.7419,  0.8447],
                    [ 0.0227, -0.0287, -0.0801,  ...,  0.7248,  0.8447,  0.9132],
                    [ 0.0569, -0.0287, -0.0629,  ...,  0.7762,  0.8961,  0.9646],
                    ...,
                    [ 1.1358,  0.3652,  0.2111,  ..., -1.9467, -1.9809, -2.0152],
                    [ 0.9303,  0.3481,  0.2111,  ..., -2.0152, -2.0152, -1.9980],
                    [ 0.8276,  0.3481,  0.2111,  ..., -1.9124, -1.8953, -1.8610]],
     
                   [[ 0.2402,  0.2227,  0.1352,  ...,  0.8179,  0.8880,  0.9930],
                    [ 0.1527,  0.1001,  0.0476,  ...,  0.8704,  0.9930,  1.0630],
                    [ 0.1877,  0.1001,  0.0651,  ...,  0.9230,  1.0455,  1.1155],
                    ...,
                    [ 1.2906,  0.5028,  0.3452,  ..., -1.8606, -1.8957, -1.9307],
                    [ 1.0805,  0.4853,  0.3452,  ..., -1.9307, -1.9307, -1.9132],
                    [ 0.9755,  0.4853,  0.3452,  ..., -1.8256, -1.8081, -1.7731]],
     
                   [[ 0.4614,  0.4439,  0.3568,  ...,  1.0365,  1.1062,  1.2108],
                    [ 0.3742,  0.3219,  0.2696,  ...,  1.0888,  1.2108,  1.2805],
                    [ 0.4091,  0.3219,  0.2871,  ...,  1.1411,  1.2631,  1.3328],
                    ...,
                    [ 1.5071,  0.7228,  0.5659,  ..., -1.6302, -1.6650, -1.6999],
                    [ 1.2980,  0.7054,  0.5659,  ..., -1.6999, -1.6999, -1.6824],
                    [ 1.1934,  0.7054,  0.5659,  ..., -1.5953, -1.5779, -1.5430]]],
     
     
                  [[[ 0.4337,  0.4508,  0.4508,  ..., -1.5528, -1.3473, -0.9020],
                    [ 0.4508,  0.4679,  0.4679,  ..., -1.5528, -1.5528, -1.2445],
                    [ 0.4508,  0.4508,  0.4337,  ..., -1.5699, -1.5357, -1.3473],
                    ...,
                    [-1.7583, -1.7583, -1.7754,  ..., -0.0116, -0.0458, -0.1143],
                    [-1.7240, -1.7754, -1.7754,  ..., -0.0458, -0.0801, -0.1314],
                    [-1.6213, -1.7925, -1.7925,  ..., -0.2684, -0.3027, -0.3541]],
     
                   [[ 0.5203,  0.5378,  0.5378,  ..., -1.5980, -1.4405, -1.1078],
                    [ 0.5378,  0.5553,  0.5553,  ..., -1.5980, -1.5105, -1.2129],
                    [ 0.5378,  0.5378,  0.5203,  ..., -1.5980, -1.5455, -1.3179],
                    ...,
                    [-1.7556, -1.7206, -1.7206,  ..., -0.6176, -0.6352, -0.6527],
                    [-1.8081, -1.7731, -1.7556,  ..., -0.3375, -0.3725, -0.4251],
                    [-1.7906, -1.8431, -1.7906,  ..., -0.5126, -0.5126, -0.5476]],
     
                   [[ 0.8622,  0.8797,  0.8797,  ..., -1.3164, -1.2293, -1.0201],
                    [ 0.8797,  0.8971,  0.8971,  ..., -1.3164, -1.2467, -1.0027],
                    [ 0.8797,  0.8797,  0.8622,  ..., -1.3164, -1.2816, -1.0898],
                    ...,
                    [-1.5081, -1.4210, -1.4210,  ..., -0.9504, -0.9678, -1.0027],
                    [-1.6302, -1.5256, -1.4907,  ..., -0.3230, -0.4101, -0.5147],
                    [-1.7173, -1.6127, -1.5430,  ..., -0.3927, -0.4275, -0.4798]]],
     
     
                  [[[-0.7479, -0.8678, -1.0048,  ...,  0.5022,  0.3309,  0.1939],
                    [-0.7137, -0.8849, -1.0390,  ...,  0.5364,  0.4851,  0.3994],
                    [-0.7137, -0.9020, -1.0048,  ...,  0.5193,  0.4679,  0.4851],
                    ...,
                    [-0.9363, -1.0904, -1.1418,  ..., -0.1486, -0.0972, -0.1999],
                    [-1.1760, -1.2274, -1.2445,  ..., -0.3369, -0.2342, -0.3541],
                    [-1.3130, -1.2103, -1.1075,  ..., -0.4226, -0.3541, -0.3198]],
     
                   [[-0.4251, -0.6527, -0.8277,  ...,  0.9055,  0.7304,  0.6254],
                    [-0.3901, -0.6352, -0.8452,  ...,  0.9580,  0.9055,  0.8529],
                    [-0.3725, -0.6527, -0.8102,  ...,  0.9580,  0.8880,  0.9755],
                    ...,
                    [-0.6527, -0.7752, -0.7927,  ...,  0.2227,  0.2927,  0.1877],
                    [-0.8978, -0.9328, -0.9678,  ..., -0.0224,  0.1527,  0.0301],
                    [-0.9678, -0.8277, -0.7752,  ..., -0.1275,  0.0301,  0.0476]],
     
                   [[ 0.1825, -0.0790, -0.2532,  ...,  1.6291,  1.4548,  1.3328],
                    [ 0.2173, -0.0790, -0.2881,  ...,  1.6465,  1.5942,  1.5245],
                    [ 0.2348, -0.0964, -0.2707,  ...,  1.6291,  1.5942,  1.6640],
                    ...,
                    [-0.0267, -0.1661, -0.1661,  ...,  0.8099,  0.9494,  0.8099],
                    [-0.2881, -0.3230, -0.3055,  ...,  0.6182,  0.7925,  0.6356],
                    [-0.3230, -0.1835, -0.0790,  ...,  0.5485,  0.7228,  0.7751]]]], device='mps:0'),
     TensorMultiCategory([[0., 0., 0.,  ..., 0., 0., 0.],
                          [0., 0., 0.,  ..., 0., 0., 0.],
                          [0., 0., 0.,  ..., 0., 0., 1.],
                          ...,
                          [0., 0., 0.,  ..., 0., 0., 0.],
                          [0., 0., 0.,  ..., 0., 0., 0.],
                          [0., 0., 0.,  ..., 0., 0., 0.]], device='mps:0'))




```python
## the model in Learner is an object of a class inheriting from "nn.Module"
## can call model using parentheses -> return the activations of model 

x,y = to_cpu(dls.train.one_batch())
activs = learn.model(x)
# batch size of 64 / 20 categories
activs.shape
```




    torch.Size([64, 20])




```python
activs[0]
```




    TensorImage([ 0.2109,  0.0866, -0.1316,  1.7263, -2.0373,  0.0544, -0.6262, -0.1172, -3.2036, -0.3143,  0.6041, -0.7967, -0.8520,  2.4630,  1.4149,  0.9663, -1.1261, -0.8106,  0.0800, -1.8106],
                grad_fn=<AliasBackward0>)




```python
def binary_cross_entropy(inputs, targets):
    inputs = inputs.sigmoid()
    return -torch.where(targets==1, inputs, 1-inputs).log().mean()
```


```python
activs.shape, y.shape
```




    (torch.Size([64, 20]), torch.Size([64, 20]))




```python
# x= x.as_subclass(torch.Tensor)
y = y.as_subclass(torch.Tensor)
```


```python
loss_func = nn.BCEWithLogitsLoss()
loss = loss_func(activs, y)
loss
```




    TensorImage(1.1001, grad_fn=<AliasBackward0>)



### 'partial' function
- allows us to bind a "function" with some "arguments" or "keyword arguments"
- making a new version of that function, always includes those arguments


```python
def say_hello(name, say_what="Hello"): return f"{say_what} {name}."
say_hello('Jeremy'), say_hello('Jeremy', 'Ahoy!')
```




    ('Hello Jeremy.', 'Ahoy! Jeremy.')




```python
f = partial(say_hello, say_what="Bonjour")
f("Jeremy"), f("Sylvain")
```




    ('Bonjour Jeremy.', 'Bonjour Sylvain.')



### accuracy multi


```python
def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):
    if sigmoid: inp = inp.sigmoid()
    ## only one element tensors can be converted to Python scalars
    ## return (int(inp > thresh) == targ).float().mean()
    return ((inp > thresh) == targ.bool()).float().mean()
```


```python
learn = vision_learner(dls, resnet50, metrics=partial(accuracy_multi, thresh=0.2))
learn.fine_tune(3, base_lr=3e-3, freeze_epochs=4)
```

    /Users/ridealist/miniconda3/envs/fastbook/lib/python3.9/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet50_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet50_Weights.DEFAULT` to get the most up-to-date weights.
      warnings.warn(msg)




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy_multi</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.942904</td>
      <td>0.708307</td>
      <td>0.238287</td>
      <td>00:34</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.825156</td>
      <td>0.561708</td>
      <td>0.293626</td>
      <td>00:33</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.605840</td>
      <td>0.204976</td>
      <td>0.819681</td>
      <td>00:33</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.360526</td>
      <td>0.126902</td>
      <td>0.939223</td>
      <td>00:34</td>
    </tr>
  </tbody>
</table>




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy_multi</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.132665</td>
      <td>0.118781</td>
      <td>0.942092</td>
      <td>00:39</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.118319</td>
      <td>0.107946</td>
      <td>0.948327</td>
      <td>00:38</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.099634</td>
      <td>0.103022</td>
      <td>0.952908</td>
      <td>00:38</td>
    </tr>
  </tbody>
</table>



```python
learn.metrics = partial(accuracy_multi, thresh=0.1)
learn.validate()
# val_loss / accuracy -> thresh 따라 차이가 큼!
```




    (#2) [0.10986798256635666,1.0]




```python
learn.metrics = partial(accuracy_multi, thresh=0.99)
learn.validate()
```


```python
## we can find the best threshold by trying a few levels and seeing what works best
preds,targs = learn.get_preds()
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>








```python
# by default "get_pres" applies the output activation function for us (sigmoid, in this case)
# so we'll set not to apply it

accuracy_multi(preds, targs, thresh=0.9, sigmoid=False)
```




    tensor(0.9571)




```python
xs = torch.linspace(0.05,0.95,29)
accs = [accuracy_multi(preds, targs, thresh=i, sigmoid=False) for i in xs]
plt.plot(xs,accs);
```


    
![png](06_multicat_files/06_multicat_59_0.png)
    


### Theory vs. Practice

- In this case, we’re using the validation set to pick a hyperparameter (the threshold), which is the purpose of the validation set.
- Cocern about Overfitting is unneeded
    - because results in a smooth curve, we're clearly not picking an inappropriate outlier

## Regression

### Assemble the Data


```python
path = untar_data(URLs.BIWI_HEAD_POSE)
```


```python
#hide
Path.BASE_PATH = path
```


```python
path.ls().sorted()
```




    (#50) [Path('01'),Path('01.obj'),Path('02'),Path('02.obj'),Path('03'),Path('03.obj'),Path('04'),Path('04.obj'),Path('05'),Path('05.obj')...]




```python
(path/'01').ls().sorted()
```




    (#1000) [Path('01/depth.cal'),Path('01/frame_00003_pose.txt'),Path('01/frame_00003_rgb.jpg'),Path('01/frame_00004_pose.txt'),Path('01/frame_00004_rgb.jpg'),Path('01/frame_00005_pose.txt'),Path('01/frame_00005_rgb.jpg'),Path('01/frame_00006_pose.txt'),Path('01/frame_00006_rgb.jpg'),Path('01/frame_00007_pose.txt')...]




```python
# get all image files recursively with "get_image_files"
img_files = get_image_files(path)

def img2pose(x): return Path(f'{str(x)[:-7]}pose.txt')

img2pose(img_files[0])
```




    Path('03/frame_00393_pose.txt')




```python
img_files
```




    (#15678) [Path('03/frame_00393_rgb.jpg'),Path('03/frame_00383_rgb.jpg'),Path('03/frame_00619_rgb.jpg'),Path('03/frame_00609_rgb.jpg'),Path('03/frame_00134_rgb.jpg'),Path('03/frame_00124_rgb.jpg'),Path('03/frame_00252_rgb.jpg'),Path('03/frame_00242_rgb.jpg'),Path('03/frame_00407_rgb.jpg'),Path('03/frame_00417_rgb.jpg')...]




```python
im = PILImage.create(img_files[0])
im.shape
```




    (480, 640)




```python
im.to_thumb(160)
```




    
![png](06_multicat_files/06_multicat_70_0.png)
    




```python
# extract the head center point : return the coordinates as a tensor of two items

# np.gengromtxt -> 텍스트 파일 열기, 컬럼별로 구분
cal = np.genfromtxt(path/'01'/'rgb.cal', skip_footer=6)

def get_ctr(f):
                        # 출력결과에서 앞 3개를 뺌
    ctr = np.genfromtxt(img2pose(f), skip_header=3)
    c1 = ctr[0] * cal[0][0]/ctr[2] + cal[0][2]
    c2 = ctr[1] * cal[1][1]/ctr[2] + cal[1][2]
    return tensor([c1,c2])
```


```python
cal
```




    array([[517.679,   0.   , 320.   ],
           [  0.   , 517.679, 240.5  ],
           [  0.   ,   0.   ,   1.   ]])




```python
get_ctr(img_files[0])
```




    tensor([387.1024, 261.9126])




```python
img_files[0], img_files[0].parent, img_files[0].parent.name
```




    (Path('03/frame_00393_rgb.jpg'), Path('03'), '03')




```python
biwi = DataBlock(     #the lable represent "coordinate" -> same augmentation coordinate & image
    blocks=(ImageBlock, PointBlock),
    get_items=get_image_files,
    get_y=get_ctr,
    # create splitter function that returns True for just one person.
    # resulting in a validation set containing just that person's images
    splitter=FuncSplitter(lambda o: o.parent.name=='13'),
    # batch_tfms=[*aug_transforms(size=(240,320)), Normalize.from_stats(*imagenet_stats)]
)
```


```python
biwi.summary(path)
```

    Setting-up type transforms pipelines
    Collecting items from /Users/ridealist/.fastai/data/biwi_head_pose
    Found 15678 items
    2 datasets of sizes 15193,485
    Setting up Pipeline: PILBase.create
    Setting up Pipeline: get_ctr -> TensorPoint.create
    
    Building one sample
      Pipeline: PILBase.create
        starting from
          /Users/ridealist/.fastai/data/biwi_head_pose/03/frame_00393_rgb.jpg
        applying PILBase.create gives
          PILImage mode=RGB size=640x480
      Pipeline: get_ctr -> TensorPoint.create
        starting from
          /Users/ridealist/.fastai/data/biwi_head_pose/03/frame_00393_rgb.jpg
        applying get_ctr gives
          tensor([387.1024, 261.9126])
        applying TensorPoint.create gives
          TensorPoint of size 1x2
    
    Final sample: (PILImage mode=RGB size=640x480, TensorPoint([[387.1024, 261.9126]]))
    
    
    Collecting items from /Users/ridealist/.fastai/data/biwi_head_pose
    Found 15678 items
    2 datasets of sizes 15193,485
    Setting up Pipeline: PILBase.create
    Setting up Pipeline: get_ctr -> TensorPoint.create
    Setting up after_item: Pipeline: PointScaler -> ToTensor
    Setting up before_batch: Pipeline: 
    Setting up after_batch: Pipeline: IntToFloatTensor -- {'div': 255.0, 'div_mask': 1}
    
    Building one batch
    Applying item_tfms to the first sample:
      Pipeline: PointScaler -> ToTensor
        starting from
          (PILImage mode=RGB size=640x480, TensorPoint of size 1x2)
        applying PointScaler gives
          (PILImage mode=RGB size=640x480, TensorPoint of size 1x2)
        applying ToTensor gives
          (TensorImage of size 3x480x640, TensorPoint of size 1x2)
    
    Adding the next 3 samples
    
    No before_batch transform to apply
    
    Collating items in a batch
    
    Applying batch_tfms to the batch built
      Pipeline: IntToFloatTensor -- {'div': 255.0, 'div_mask': 1}
        starting from
          (TensorImage of size 4x3x480x640, TensorPoint of size 4x1x2)
        applying IntToFloatTensor -- {'div': 255.0, 'div_mask': 1} gives
          (TensorImage of size 4x3x480x640, TensorPoint of size 4x1x2)



```python
dls = biwi.dataloaders(path)
dls.show_batch(max_n=9, figsize=(8,6))
```


    
![png](06_multicat_files/06_multicat_77_0.png)
    



```python
# Make sure that WHY these are the shapes for mini-batches

xb, yb = dls.one_batch()
xb.shape, yb.shape
```




    (torch.Size([64, 3, 480, 640]), torch.Size([64, 1, 2]))




```python
yb[0]
```




    TensorPoint([[0.2792, 0.0445]], device='mps:0')



### Training a Model


```python
## (coordinates in fastai and PyTorch are always rescaled between –1 and +1)

learn = vision_learner(dls, resnet18, y_range=(-1,1))
```


```python
# This is set as the final layer of the model
# it forces the model to output activations in the range (lo, hi)

def sigmoid_range(x, lo, hi): return torch.sigmoid(x) * (hi-lo) + lo
```


```python
# Basic sigmoid

plot_function(partial(sigmoid_range,lo=0,hi=1), min=-4, max=4)
```


    
![png](06_multicat_files/06_multicat_83_0.png)
    



```python
plot_function(partial(sigmoid_range,lo=-1,hi=1), min=-4, max=4)
```


    
![png](06_multicat_files/06_multicat_84_0.png)
    



```python
plot_function(partial(sigmoid_range,lo=-2,hi=5), min=-4, max=4)
```


    
![png](06_multicat_files/06_multicat_85_0.png)
    



```python
# see what loss function choosed by fastai as the default

dls.loss_func
```




    FlattenedLoss of MSELoss()




```python
# pick a good learning rate wit "learning rate finder"

learn.lr_find()
```


```python
lr = 1e-2
learn.fine_tune(3, lr)
```


```python
math.sqrt(0.0001)
```


```python
## take a look at our results
# left : the actual coordinates / right : model's predictions

learn.show_results(ds_idx=1, nrows=3, figsize=(6,8))
```

## Conclusion

## Questionnaire

1. How could multi-label classification improve the usability of the bear classifier?
1. How do we encode the dependent variable in a multi-label classification problem?
1. How do you access the rows and columns of a DataFrame as if it was a matrix?
1. How do you get a column by name from a DataFrame?
1. What is the difference between a `Dataset` and `DataLoader`?
1. What does a `Datasets` object normally contain?
1. What does a `DataLoaders` object normally contain?
1. What does `lambda` do in Python?
1. What are the methods to customize how the independent and dependent variables are created with the data block API?
1. Why is softmax not an appropriate output activation function when using a one hot encoded target?
1. Why is `nll_loss` not an appropriate loss function when using a one-hot-encoded target?
1. What is the difference between `nn.BCELoss` and `nn.BCEWithLogitsLoss`?
1. Why can't we use regular accuracy in a multi-label problem?
1. When is it okay to tune a hyperparameter on the validation set?
1. How is `y_range` implemented in fastai? (See if you can implement it yourself and test it without peeking!)
1. What is a regression problem? What loss function should you use for such a problem?
1. What do you need to do to make sure the fastai library applies the same data augmentation to your input images and your target point coordinates?

### Further Research

1. Read a tutorial about Pandas DataFrames and experiment with a few methods that look interesting to you. See the book's website for recommended tutorials.
1. Retrain the bear classifier using multi-label classification. See if you can make it work effectively with images that don't contain any bears, including showing that information in the web application. Try an image with two different kinds of bears. Check whether the accuracy on the single-label dataset is impacted using multi-label classification.


```python

```
