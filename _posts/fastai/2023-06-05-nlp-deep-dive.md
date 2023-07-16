---
published: true
layout: posts
title: '[DL4C] Ch11. NLP Deep Dive'
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
from IPython.display import display,HTML
```

# NLP Deep Dive: RNNs

1. What is self supervised learning?
    - 외부 label 없이 independant variable(독립 변수)에 포함되어 있는 label을 사용해 model을 학습시키는 방법
        - model에게 label을 줄 필요 없고, 많은 양의 텍스트만 학습시키면 됨
        - 데이터로부터 자체적으로 label을 얻을 수 있음

1. What is lanaguage model?
    - text에서 다음 단워가 무엇인지 예측하는 모델

1. Why is a language model considered self-supervised?
    - text 안에서 앞 단어들의 stream에서 다음 단어를 예측하면 되므로

1. What are self-supervised models usually used for?
    - 언어를 생성하는 자체 모델로서 활용됨. (검색어 자동 완성 기능 등)
    - 다른 task에서 transfer-learning을 위한 pre-trained model로서 가장 많이 활용됨
    
1. Why do we fine-tune language models?
    - pre-training 모델이 우리가 하려는 task와 다르게 학습되어 있으므로
    - 우리가 하려는 downstream task에 맞는 "corpus"로 LM을 파인튜닝 시키기 위해
    - 원래의 pre-trained 모델은 현재의 작업과 약간 다른 corpus로 학습되어 있기 때문
        - ex) Wikitext <-> IMDb

1. What are the three steps to create a state-of-the-art text classifier?
- Universal Language Model Fine-tuning (ULMFiT) approach

    1. Train a language model on a large corpus of text (already done for ULM-FiT by Sebastian Ruder and Jeremy!)
    2. Fine-tune the language model on text classification dataset
    3. Fine-tune the language model as a text classifier instead.

1. How do the 50,000 unlabeled movie reviews help create a better text classifier for the IMDb dataset?
    - movie review의 다음 단어를 어떻게 예측할지 학습함으로써, text 분류 dataset language 스타일과 구조에 대해 더 잘 이해하게 됨
    - classifier로 파인튜지이 했을 때 같은 dataset을 더 잘 분류할 수 있음

1. What are the three steps to prepare your data for a language model?
    - Tokenization
    - Numericalization
    - Language model data loader

## Text Preprocessing

### Tokenization

1. What is "tokenization"? Why do we need it?
    - 언어의 stream을 기계가 알아들을 수 있는 적절한 단위로 나눠야 함 -> 벡터화를 위해
    - 간단한 문제는 아님. 문장부호, -, "", 등등 다양한 복잡한 케이스를 모두 커버할 수 있도록 tokenizing 해야 함

1. Name three different approaches to tokenization.
    - Word-based
    - Subword-based
    - Character-based

### Word Tokenization with fastai


```python
from fastai.text.all import *
path = untar_data(URLs.IMDB)
```


```python
files = get_text_files(path, folders = ['train', 'test', 'unsup'])
```


```python
files
```




    (#100000) [Path('/Users/ridealist/.fastai/data/imdb/test/neg/1821_4.txt'),Path('/Users/ridealist/.fastai/data/imdb/test/neg/9487_1.txt'),Path('/Users/ridealist/.fastai/data/imdb/test/neg/4604_4.txt'),Path('/Users/ridealist/.fastai/data/imdb/test/neg/2828_2.txt'),Path('/Users/ridealist/.fastai/data/imdb/test/neg/10890_1.txt'),Path('/Users/ridealist/.fastai/data/imdb/test/neg/3351_4.txt'),Path('/Users/ridealist/.fastai/data/imdb/test/neg/8070_2.txt'),Path('/Users/ridealist/.fastai/data/imdb/test/neg/1027_4.txt'),Path('/Users/ridealist/.fastai/data/imdb/test/neg/8248_3.txt'),Path('/Users/ridealist/.fastai/data/imdb/test/neg/4290_4.txt')...]




```python
txt = files[0].open().read(); txt[:75]
```




    'Alan Rickman & Emma Thompson give good performances with southern/New Orlea'




```python
spacy = WordTokenizer()
Tokenizer(txt)
print(coll_repr(tkn(txt), 31))
```




    Tokenizer:
    encodes: (Path,object) -> encodes
    (str,object) -> encodes
    decodes: (object,object) -> decodes




```python
# fastai's default word tokenizer
spacy = WordTokenizer()
        # fastai's tokenizers take a "collection of documents" to tokenize
toks = first(spacy([txt]))
             # Display the first n items of collections, along with the full size
# display the first n items of collections, along with the full size
print(coll_repr(toks, 30))
```

    (#121) ['Alan','Rickman','&','Emma','Thompson','give','good','performances','with','southern','/','New','Orleans','accents','in','this','detective','flick','.','It',"'s",'worth','seeing','for','their','scenes-','and','Rickman',"'s",'scene'...]



```python
print(toks)
```

    ['Alan', 'Rickman', '&', 'Emma', 'Thompson', 'give', 'good', 'performances', 'with', 'southern', '/', 'New', 'Orleans', 'accents', 'in', 'this', 'detective', 'flick', '.', 'It', "'s", 'worth', 'seeing', 'for', 'their', 'scenes-', 'and', 'Rickman', "'s", 'scene', 'with', 'Hal', 'Holbrook', '.', 'These', 'three', 'actors', 'mannage', 'to', 'entertain', 'us', 'no', 'matter', 'what', 'the', 'movie', ',', 'it', 'seems', '.', 'The', 'plot', 'for', 'the', 'movie', 'shows', 'potential', ',', 'but', 'one', 'gets', 'the', 'impression', 'in', 'watching', 'the', 'film', 'that', 'it', 'was', 'not', 'pulled', 'off', 'as', 'well', 'as', 'it', 'could', 'have', 'been', '.', 'The', 'fact', 'that', 'it', 'is', 'cluttered', 'by', 'a', 'rather', 'uninteresting', 'subplot', 'and', 'mostly', 'uninteresting', 'kidnappers', 'really', 'muddles', 'things', '.', 'The', 'movie', 'is', 'worth', 'a', 'view-', 'if', 'for', 'nothing', 'more', 'than', 'entertaining', 'performances', 'by', 'Rickman', ',', 'Thompson', ',', 'and', 'Holbrook', '.']



```python
first(spacy(['The U.S. dollar $1 is $1.00.']))
```




    (#9) ['The','U.S.','dollar','$','1','is','$','1.00','.']



xx -> special token

1. What is `xxbos`?
    - Begining-of-stream. 문장의 시작
    - 이전 것을 '잊어버리고' 앞으로 나올 문장에 집중하게 됨

In a sense,
we are translating the original English language sequence into a
simplified tokenized language—a language that is designed to be easy
for a model to learn.


```python
# fastai Tokenizer class
tkn = Tokenizer(spacy)
print(coll_repr(tkn(txt), 31))
```

    (#139) ['xxbos','xxmaj','alan','xxmaj','rickman','&','xxmaj','emma','xxmaj','thompson','give','good','performances','with','southern','/','xxmaj','new','xxmaj','orleans','accents','in','this','detective','flick','.','xxmaj','it',"'s",'worth','seeing'...]



```python
defaults.text_proc_rules
```




    [<function fastai.text.core.fix_html(x)>,
     <function fastai.text.core.replace_rep(t)>,
     <function fastai.text.core.replace_wrep(t)>,
     <function fastai.text.core.spec_add_spaces(t)>,
     <function fastai.text.core.rm_useless_spaces(t)>,
     <function fastai.text.core.replace_all_caps(t)>,
     <function fastai.text.core.replace_maj(t)>,
     <function fastai.text.core.lowercase(t, add_bos=True, add_eos=False)>]



1. List rules that fastai applies to text during tokenization.

    - fix_html :: replace special HTML characters by a readable version (IMDb reviews have quite a few of them for instance) ;
    - replace_rep :: replace any character repeated three times or more by a special token for repetition (xxrep), the number of times it’s repeated, then the character ;
    - replace_wrep :: replace any word repeated three times or more by a special token for word repetition (xxwrep), the number of times it’s repeated, then the word ;
    - spec_add_spaces :: add spaces around / and # ;
    - rm_useless_spaces :: remove all repetitions of the space character ;
    - replace_all_caps :: lowercase a word written in all caps and adds a special token for all caps (xxcap) in front of it ;
    - replace_maj :: lowercase a capitalized word and adds a special token for capitalized (xxmaj) in front of it ;
    - lowercase :: lowercase all text and adds a special token at the beginning (xxbos) and/or the end (xxeos).

1. Why are repeated characters replaced with a token showing the number of repetitions, and the character that’s repeated?
    - 반복되는 character들은 한 글자와 다른 의미를 가졌다고 예상 가능
    - 모델의 embedding matrix가 일반적인 개념으로 정보를 encoding 할 수 있음


```python
coll_repr(tkn('&copy;   Fast.ai www.fast.ai/INDEX'), 31)
```




    "(#11) ['xxbos','©','xxmaj','fast.ai','xxrep','3','w','.fast.ai','/','xxup','index']"



### Subword Tokenization


```python
txts = L(o.open().read() for o in files[:2000])
```


```python
def subword(sz): #sz=size
    sp = SubwordTokenizer(vocab_sz=sz)
    sp.setup(txts)
    return ' '.join(first(sp([txt]))[:40])
```

- subword tokenizer -> word - character 사이의 역할
- vocab의 개수가 token의 갯수를 결정한다??
    - vocab가 클수록 오히려 word 단위로 분리하게 된다
    - vocab가 작을수록 character 단위로 분리하게 된다


```python
subword(1000)
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







    sentencepiece_trainer.cc(177) LOG(INFO) Running command: --input=tmp/texts.out --vocab_size=1000 --model_prefix=tmp/spm --character_coverage=0.99999 --model_type=unigram --unk_id=9 --pad_id=-1 --bos_id=-1 --eos_id=-1 --minloglevel=2 --user_defined_symbols=▁xxunk,▁xxpad,▁xxbos,▁xxeos,▁xxfld,▁xxrep,▁xxwrep,▁xxup,▁xxmaj --hard_vocab_limit=false





    '▁Al an ▁R ick man ▁& ▁E m m a ▁Th o mp s on ▁give ▁good ▁performance s ▁with ▁so uth er n / N e w ▁O r le an s ▁acc ent s ▁in ▁this ▁de t'




```python
subword(200)
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










    '▁A l an ▁ R i ck m an ▁ & ▁ E m m a ▁ T h o m p s on ▁ g i ve ▁ g o o d ▁p er f or m an c'




```python
subword(10000)
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










    "▁Al an ▁Rick man ▁& ▁Emma ▁Thompson ▁give ▁good ▁performances ▁with ▁southern / N ew ▁Orleans ▁accents ▁in ▁this ▁detective ▁flick . ▁It ' s ▁worth ▁seeing ▁for ▁their ▁scenes - ▁and ▁Rick man ' s ▁scene ▁with ▁H al"



### Numericalization with fastai

1. What is "numericalization”?
    - token -> integer로 매핑
        - 가능한 모든 tokens vocab의 level을 파악
        - 각 vocab을 index로 변환


```python
toks = tkn(txt)
print(coll_repr(tkn(txt), 31))
```

    (#139) ['xxbos','xxmaj','alan','xxmaj','rickman','&','xxmaj','emma','xxmaj','thompson','give','good','performances','with','southern','/','xxmaj','new','xxmaj','orleans','accents','in','this','detective','flick','.','xxmaj','it',"'s",'worth','seeing'...]



```python
txts = L(o.open().read() for o in files[:2000]);
```


```python
toks200 = txts[:200].map(tkn)
toks200[0]
```




    (#139) ['xxbos','xxmaj','alan','xxmaj','rickman','&','xxmaj','emma','xxmaj','thompson'...]




```python
num = Numericalize()
# need to call 'setup' on Numericalize
num.setup(toks200)
coll_repr(num.vocab,20)

# special rules tokens first, every word appears once
# default : min_freq=3, max_vocab=60_000
```




    "(#1984) ['xxunk','xxpad','xxbos','xxeos','xxfld','xxrep','xxwrep','xxup','xxmaj','the','.',',','and','a','to','of','i','it','is','in'...]"



1. Why might there be words that are replaced with the "unknown word" token?
- 가장 일반적인 60_000개 vocab 제외 나머지 "unknown token"으로 변환 -> "xxunk"
    - 지나치게 embedding matrix가 커지는 것 방지
    - 지나친 메모리 소요, 학습 시간 저하 방지


```python
nums = num(toks)[:20]; nums
```




    TensorText([   2,    8,    0,    8, 1442,  234,    8,    0,    8,    0,  199,   64,  731,   29,    0,  122,    8,  253,    8,    0])




```python
' '.join(num.vocab[o] for o in nums)
```




    'xxbos xxmaj xxunk xxmaj rickman & xxmaj xxunk xxmaj xxunk give good performances with xxunk / xxmaj new xxmaj xxunk'



### Putting Our Texts into Batches for a Language Model


```python
stream = "In this chapter, we will go back over the example of classifying movie reviews we studied in chapter 1 and dig deeper under the surface. First we will look at the processing steps necessary to convert text into numbers and how to customize it. By doing this, we'll have another example of the PreProcessor used in the data block API.\nThen we will study how we build a language model and train it for a while."; stream
```




    "In this chapter, we will go back over the example of classifying movie reviews we studied in chapter 1 and dig deeper under the surface. First we will look at the processing steps necessary to convert text into numbers and how to customize it. By doing this, we'll have another example of the PreProcessor used in the data block API.\nThen we will study how we build a language model and train it for a while."




```python
tokens = tkn(stream)
' '.join(tokens)
```




    "xxbos xxmaj in this chapter , we will go back over the example of classifying movie reviews we studied in chapter 1 and dig deeper under the surface . xxmaj first we will look at the processing steps necessary to convert text into numbers and how to customize it . xxmaj by doing this , we 'll have another example of the preprocessor used in the data block xxup api . \n xxmaj then we will study how we build a language model and train it for a while ."




```python
bs,seq_len = 6,15
d_tokens = np.array([tokens[i*seq_len : (i+1)*seq_len] for i in range(bs)])
df = pd.DataFrame(d_tokens)
display(HTML(df.to_html(index=False,header=None)))
```


<table border="1" class="dataframe">
  <tbody>
    <tr>
      <td>xxbos</td>
      <td>xxmaj</td>
      <td>in</td>
      <td>this</td>
      <td>chapter</td>
      <td>,</td>
      <td>we</td>
      <td>will</td>
      <td>go</td>
      <td>back</td>
      <td>over</td>
      <td>the</td>
      <td>example</td>
      <td>of</td>
      <td>classifying</td>
    </tr>
    <tr>
      <td>movie</td>
      <td>reviews</td>
      <td>we</td>
      <td>studied</td>
      <td>in</td>
      <td>chapter</td>
      <td>1</td>
      <td>and</td>
      <td>dig</td>
      <td>deeper</td>
      <td>under</td>
      <td>the</td>
      <td>surface</td>
      <td>.</td>
      <td>xxmaj</td>
    </tr>
    <tr>
      <td>first</td>
      <td>we</td>
      <td>will</td>
      <td>look</td>
      <td>at</td>
      <td>the</td>
      <td>processing</td>
      <td>steps</td>
      <td>necessary</td>
      <td>to</td>
      <td>convert</td>
      <td>text</td>
      <td>into</td>
      <td>numbers</td>
      <td>and</td>
    </tr>
    <tr>
      <td>how</td>
      <td>to</td>
      <td>customize</td>
      <td>it</td>
      <td>.</td>
      <td>xxmaj</td>
      <td>by</td>
      <td>doing</td>
      <td>this</td>
      <td>,</td>
      <td>we</td>
      <td>'ll</td>
      <td>have</td>
      <td>another</td>
      <td>example</td>
    </tr>
    <tr>
      <td>of</td>
      <td>the</td>
      <td>preprocessor</td>
      <td>used</td>
      <td>in</td>
      <td>the</td>
      <td>data</td>
      <td>block</td>
      <td>xxup</td>
      <td>api</td>
      <td>.</td>
      <td>\n</td>
      <td>xxmaj</td>
      <td>then</td>
      <td>we</td>
    </tr>
    <tr>
      <td>will</td>
      <td>study</td>
      <td>how</td>
      <td>we</td>
      <td>build</td>
      <td>a</td>
      <td>language</td>
      <td>model</td>
      <td>and</td>
      <td>train</td>
      <td>it</td>
      <td>for</td>
      <td>a</td>
      <td>while</td>
      <td>.</td>
    </tr>
  </tbody>
</table>



```python
bs,seq_len = 6,5
d_tokens = np.array([tokens[i*15:i*15+seq_len] for i in range(bs)])
df = pd.DataFrame(d_tokens)
display(HTML(df.to_html(index=False,header=None)))
```


<table border="1" class="dataframe">
  <tbody>
    <tr>
      <td>xxbos</td>
      <td>xxmaj</td>
      <td>in</td>
      <td>this</td>
      <td>chapter</td>
    </tr>
    <tr>
      <td>movie</td>
      <td>reviews</td>
      <td>we</td>
      <td>studied</td>
      <td>in</td>
    </tr>
    <tr>
      <td>first</td>
      <td>we</td>
      <td>will</td>
      <td>look</td>
      <td>at</td>
    </tr>
    <tr>
      <td>how</td>
      <td>to</td>
      <td>customize</td>
      <td>it</td>
      <td>.</td>
    </tr>
    <tr>
      <td>of</td>
      <td>the</td>
      <td>preprocessor</td>
      <td>used</td>
      <td>in</td>
    </tr>
    <tr>
      <td>will</td>
      <td>study</td>
      <td>how</td>
      <td>we</td>
      <td>build</td>
    </tr>
  </tbody>
</table>



```python
bs,seq_len = 6,5
d_tokens = np.array([tokens[i*15+seq_len:i*15+2*seq_len] for i in range(bs)])
df = pd.DataFrame(d_tokens)
display(HTML(df.to_html(index=False,header=None)))
```


<table border="1" class="dataframe">
  <tbody>
    <tr>
      <td>,</td>
      <td>we</td>
      <td>will</td>
      <td>go</td>
      <td>back</td>
    </tr>
    <tr>
      <td>chapter</td>
      <td>1</td>
      <td>and</td>
      <td>dig</td>
      <td>deeper</td>
    </tr>
    <tr>
      <td>the</td>
      <td>processing</td>
      <td>steps</td>
      <td>necessary</td>
      <td>to</td>
    </tr>
    <tr>
      <td>xxmaj</td>
      <td>by</td>
      <td>doing</td>
      <td>this</td>
      <td>,</td>
    </tr>
    <tr>
      <td>the</td>
      <td>data</td>
      <td>block</td>
      <td>xxup</td>
      <td>api</td>
    </tr>
    <tr>
      <td>a</td>
      <td>language</td>
      <td>model</td>
      <td>and</td>
      <td>train</td>
    </tr>
  </tbody>
</table>



```python
bs,seq_len = 6,5
d_tokens = np.array([tokens[i*15+10:i*15+15] for i in range(bs)])
df = pd.DataFrame(d_tokens)
display(HTML(df.to_html(index=False,header=None)))
```


<table border="1" class="dataframe">
  <tbody>
    <tr>
      <td>over</td>
      <td>the</td>
      <td>example</td>
      <td>of</td>
      <td>classifying</td>
    </tr>
    <tr>
      <td>under</td>
      <td>the</td>
      <td>surface</td>
      <td>.</td>
      <td>xxmaj</td>
    </tr>
    <tr>
      <td>convert</td>
      <td>text</td>
      <td>into</td>
      <td>numbers</td>
      <td>and</td>
    </tr>
    <tr>
      <td>we</td>
      <td>'ll</td>
      <td>have</td>
      <td>another</td>
      <td>example</td>
    </tr>
    <tr>
      <td>.</td>
      <td>\n</td>
      <td>xxmaj</td>
      <td>then</td>
      <td>we</td>
    </tr>
    <tr>
      <td>it</td>
      <td>for</td>
      <td>a</td>
      <td>while</td>
      <td>.</td>
    </tr>
  </tbody>
</table>


### LMDataLoader


```python
nums200 = toks200.map(num)
```


```python
dl = LMDataLoader(nums200)
```

LMDataLoaders
- concatenate all texts (maybe shuffled) in one big stream, split it in bs contigous sentences. then go through those seq_len at a time.


```python
x,y = first(dl)
x.shape,y.shape
```




    (torch.Size([64, 72]), torch.Size([64, 72]))




```python
first(dl)
```




    (LMTensorText([[   2,    8,    0,  ..., 1134,   11,   31],
                   [   8,    0,   39,  ...,   10,    8,   19],
                   [  21,    0,   21,  ..., 1487,    0,  664],
                   ...,
                   [  10,    8,    0,  ...,  421,  144,    8],
                   [   0,   19,  677,  ...,    9,  105,  127],
                   [   0,   34,    9,  ...,   54,    9, 1133]]),
     TensorText([[   8,    0,    8,  ...,   11,   31,   42],
                 [   0,   39,   49,  ...,    8,   19,  105],
                 [   0,   21, 1486,  ...,    0,  664,   10],
                 ...,
                 [   8,    0,   10,  ...,  144,    8,  729],
                 [  19,  677,   11,  ...,  105,  127,    0],
                 [  34,    9,   74,  ...,    9, 1133,   82]]))




```python
x[0]
```




    LMTensorText([   2,    8,    0,    8, 1442,  234,    8,    0,    8,    0,  199,   64,  731,   29,    0,  122,    8,  253,    8,    0,  943,   19,   20,  944,  294,   10,    8,   17,   25,  338,  408,
                    28,  102,    0,   12,    8, 1442,   25,  160,   29,    8,    0,    8,    0,   10,    8,  163,  320,  164,    0,   14, 1443,  295,   77,  254,   61,    9,   27,   11,   17,  296,   10,
                     8,    9,  110,   28,    9,   27,  650, 1134,   11,   31])




```python
x[1]
```




    LMTensorText([   8,    0,   39,   49,   13,    0,    0,   29, 1466, 1467,  841,  323,   36,   13,  656,   15,  382,   29,    0,   10,    8,  219,   11,  954,   14,  108,    0,   11,   61,  214,  279,
                     8,  588,   41,  118,  127,  847,  342,   39,   75,    0,    0,  323,   51, 1468,   12, 1468,   12, 1468,   14,    0,   13,  955,   41,   14,   77,    0,   39,   12,  120,  956,    0,
                   102, 1469,   58,    0,   19,    9,  957,   10,    8,   19])




```python
for txt in txts[:5]:
    print(txt)
    print("-"*100)
```

    Alan Rickman & Emma Thompson give good performances with southern/New Orleans accents in this detective flick. It's worth seeing for their scenes- and Rickman's scene with Hal Holbrook. These three actors mannage to entertain us no matter what the movie, it seems. The plot for the movie shows potential, but one gets the impression in watching the film that it was not pulled off as well as it could have been. The fact that it is cluttered by a rather uninteresting subplot and mostly uninteresting kidnappers really muddles things. The movie is worth a view- if for nothing more than entertaining performances by Rickman, Thompson, and Holbrook.
    ----------------------------------------------------------------------------------------------------
    I have seen this movie and I did not care for this movie anyhow. I would not think about going to Paris because I do not like this country and its national capital. I do not like to learn french anyhow because I do not understand their language. Why would I go to France when I rather go to Germany or the United Kingdom? Germany and the United Kingdom are the nations I tolerate. Apparently the Olsen Twins do not understand the French language just like me. Therefore I will not bother the France trip no matter what. I might as well stick to the United Kingdom and meet single women and play video games if there is a video arcade. That is all.
    ----------------------------------------------------------------------------------------------------
    In Los Angeles, the alcoholic and lazy Hank Chinaski (Matt Dillon) performs a wide range of non-qualified functions just to get enough money to drink and gamble in horse races. His primary and only objective is writing and having sexy with dirty women.<br /><br />"Factotum" is an uninteresting, pointless and extremely boring movie about an irresponsible drunken vagrant that works a couple of days or weeks just to get enough money to buy spirits and gamble, being immediately fired due to his reckless behavior. In accordance with IMDb, this character would be the fictional alter-ego of the author Charles Bukowski, and based on this story, I will certainly never read any of his novels. Honestly, if the viewer likes this theme of alcoholic couples, better off watching the touching and heartbreaking Hector Babenco's "Ironweed" or Marco Ferreri's "Storie di Ordinaria Follia" that is based on the life of the same writer. My vote is four.<br /><br />Title (Brazil): "Factotum  Sem Destino" ("Factotum  Without Destiny")
    ----------------------------------------------------------------------------------------------------
    This film is bundled along with "Gli fumavano le Colt... lo chiamavano Camposanto" and both films leave a lot to be desired in the way of their DVD prints. First, both films are very dark--occasionally making it hard to see exactly what's happening. Second, neither film has subtitles and you are forced to watch a dubbed film--though "Il Prezzo del Potere" does seem to have a better dub. Personally, I always prefer subtitles but for the non-purists out there this isn't a problem. These DVD problems, however, are not the fault of the original film makers--just the indifferent package being marketed four decades later.<br /><br />As for the film, it's about the assassination of President Garfield. This is a MAJOR problem, as Van Johnson looks about as much like Garfield as Judy Garland. In no way whatsoever does he look like Garfield. He's missing the beard, has the wrong hair color and style and is just not even close in any way (trust me on this, I am an American History teacher and we are paid to know these sort of things!). The real life Garfield was a Civil War general and looked like the guys on the Smith Brothers cough drop boxes. Plus, using some other actor to provide the voice for Johnson in the dubbing is just surreal. Never before or since has Van Johnson sounded quite so macho!! He was a fine actor...but certainly not a convincing general or macho president.<br /><br />In addition to the stupid casting, President Garfield's death was in no way like this film. It's obvious that the film makers are actually cashing in on the crazy speculation about conspiracies concerning the death of JFK, not Garfield. Garfield was shot in Washington, DC (not Dallas) by a lone gunman with severe mental problems--not a group of men with rifles. However, according to most experts, what actually killed Garfield (over two months later) were incompetent doctors--who probed and probed and probed to retrieve a bullet (to no avail) and never bothered cleaning their hands or implements in the process. In other words, like George Washington (who was basically killed by repeated bloodletting when suffering with pneumonia) he died due to malpractice. In the movie they got nothing right whatsoever...other than indeed President Garfield was shot.<br /><br />Because the film bears almost no similarity to real history, it's like a history lesson as taught from someone from another planet or someone with a severe brain injury. Why not also include ninjas, fighting robots and the Greek gods while you're at it?!?! Aside from some decent acting and production values, because the script is utter cow crap, I don't recommend anyone watch it. It's just a complete and utter mess.
    ----------------------------------------------------------------------------------------------------
    I only comment on really very good films and on utter rubbish. My aim is to help people who want to see great films to spend their time - and money - wisely.<br /><br />I also want to stop people wasting their time on garbage, and want to publicize the fact that the director/producer of these garbage films can't get away with it for very long. We will find out who you are and will vote with out feet - and wallets.<br /><br />This film clearly falls into the garbage category.<br /><br />The director and writer is John Shiban. It's always a bad sign when the writer is also the director. Maybe he wants two pay cheques. He shouldn't get any. So remember the name - John SHIBAN. And if you see anything else by him, forget it.<br /><br />I won't say anything about the plot - others have already. I am a little worried by how much the director likes to zoom in to the poor girl's face when she is crying and screaming. These long duration shots are a little worrying and may say something about the state of mind of Mr. Shiban. Maybe he should get psychiatric help.<br /><br />Enough already. It's crap - don't waste your time on it.
    ----------------------------------------------------------------------------------------------------



```python
for i in range(10):
    print(' '.join(num.vocab[o] for o in x[i].tolist()))
    print('-' * 100)
```

    xxbos xxmaj xxunk xxmaj rickman & xxmaj xxunk xxmaj xxunk give good performances with xxunk / xxmaj new xxmaj xxunk accents in this detective flick . xxmaj it 's worth seeing for their xxunk and xxmaj rickman 's scene with xxmaj xxunk xxmaj xxunk . xxmaj these three actors xxunk to entertain us no matter what the movie , it seems . xxmaj the plot for the movie shows potential , but
    ----------------------------------------------------------------------------------------------------
    xxmaj xxunk ) by a xxunk xxunk with severe mental problems -- not a group of men with xxunk . xxmaj however , according to most xxunk , what actually killed xxmaj garfield ( over two months later ) were xxunk xxunk -- who probed and probed and probed to xxunk a bullet ( to no xxunk ) and never bothered xxunk their hands or xxunk in the process . xxmaj in
    ----------------------------------------------------------------------------------------------------
    " xxunk " requires a lot of xxunk to xxunk through and will probably turn off most viewers ; but the dialogue xxunk xxunk true and xxmaj joe xxmaj xxunk , who xxunk his body in almost every scene , also gives an utterly convincing performance . a xxunk , to be sure , but the more xxunk " trash " , made two years later , is a definite xxunk forward
    ----------------------------------------------------------------------------------------------------
    more successful at xxunk rather than acting . 
    
     xxmaj overall rating : xxmaj do not rent … do xxup not xxup buy ! xxbos i was forced to watch this film for my xxmaj world xxmaj xxunk xxmaj xxunk class . xxmaj this film is what is wrong with xxmaj america today , instead of xxunk out the best way out of hard times or situations we would rather xxunk about
    ----------------------------------------------------------------------------------------------------
    still do n't really know what point it was supposed to get across , but i do know that a good two hours was wasted from my life . xxmaj two precious hours i can never get back . 
    
     xxmaj the storyline was so predictable , it 's laughable . xxmaj xxunk … or something … a very xxmaj xxunk and xxmaj xxunk type plot . i xxunk the xxunk within
    ----------------------------------------------------------------------------------------------------
    off as a kind of xxunk - professional movie . xxmaj unfortunately the poor effects , wooden acting and unoriginal story makes this a very mediocre horror slasher at best . xxmaj by no xxunk is xxmaj dark xxmaj harvest the worst horror movie i 've ever seen , it just is n't anything special and has nothing in it to xxunk a second watch or hope for a sequel . xxmaj
    ----------------------------------------------------------------------------------------------------
    . i do n't think i would ever want to watch it again , there 's no real xxunk , the plot is weak that xxunk classic xxmaj vampire themes with silly xxunk & i was distinctly xxunk by it all . xxmaj not the worst film ever but hardly the best either . 
    
     xxmaj the film looks alright with nice locations & some local xxunk although you feel the look
    ----------------------------------------------------------------------------------------------------
    beyond the call of xxunk under the most xxunk bad movie xxunk . xxmaj watch leading xxunk xxmaj xxunk , then you can xxunk his xxunk under fire . xxmaj she looks like she just xxunk up inside a bad dream and maybe if she stands stock - still , no one will xxunk her . i barely did . xxmaj oh well , the first time i saw this drive -
    ----------------------------------------------------------------------------------------------------
    would put himself . xxmaj now , xxmaj joseph gordon - xxunk , is a good looking actor , but he played his character a little xxunk . i did n't believe his acting , it looked like the director tried to pull out of him some personality he could n't xxunk . xxmaj and he did n't have to , because his less crazy behavior was creepy enough . xxmaj the
    ----------------------------------------------------------------------------------------------------
    in to things , xxunk her head back , and xxunk her xxunk - was this considered " dancing " back then ? xxmaj if it was , xxmaj i 'm not sure i could 've xxunk it . xxmaj the xxunk , the xxunk , the xxunk xxunk were great , but some of it - my xxunk 's xxunk could 've done better ! xxmaj also , if xxmaj alex
    ----------------------------------------------------------------------------------------------------



```python
' '.join(num.vocab[o] for o in x[0][:20])
```




    'xxbos xxmaj xxunk xxmaj rickman & xxmaj xxunk xxmaj xxunk give good performances with xxunk / xxmaj new xxmaj xxunk'




```python
' '.join(num.vocab[o] for o in y[0][:20])
```




    'xxmaj xxunk xxmaj rickman & xxmaj xxunk xxmaj xxunk give good performances with xxunk / xxmaj new xxmaj xxunk accents'




```python
txt = files[0].open().read(); txt
```




    "Alan Rickman & Emma Thompson give good performances with southern/New Orleans accents in this detective flick. It's worth seeing for their scenes- and Rickman's scene with Hal Holbrook. These three actors mannage to entertain us no matter what the movie, it seems. The plot for the movie shows potential, but one gets the impression in watching the film that it was not pulled off as well as it could have been. The fact that it is cluttered by a rather uninteresting subplot and mostly uninteresting kidnappers really muddles things. The movie is worth a view- if for nothing more than entertaining performances by Rickman, Thompson, and Holbrook."




```python
txt = files[1].open().read(); txt
```




    'I have seen this movie and I did not care for this movie anyhow. I would not think about going to Paris because I do not like this country and its national capital. I do not like to learn french anyhow because I do not understand their language. Why would I go to France when I rather go to Germany or the United Kingdom? Germany and the United Kingdom are the nations I tolerate. Apparently the Olsen Twins do not understand the French language just like me. Therefore I will not bother the France trip no matter what. I might as well stick to the United Kingdom and meet single women and play video games if there is a video arcade. That is all.'




```python
' '.join(num.vocab[o] for o in x[1][:20])
```




    'xxmaj xxunk ) by a xxunk xxunk with severe mental problems -- not a group of men with xxunk .'



1. With a batch size of 64, the first row of the tensor representing the first batch contains the first 64 tokens for the dataset.

- What does the second row of that tensor contain?
- What does the first row of the second batch contain?
   - (Careful—students often get this one wrong! Be sure to check your answer on the book's website.)

- a. The dataset is split into 64 mini-streams (batch size)
- b. Each batch has 64 rows (batch size) and 64 columns (sequence length)
- c. The first row of the first batch contains the beginning of the first mini-stream (tokens 1-64)
- d. The second row of the first batch contains the beginning of the second mini-stream
- e. The first row of the second batch contains the second chunk of the first mini-stream (tokens 65-128)

## Training a Text Classifier

- TextBlock을 전달할 때 토큰화와 수치화를 자동으로 다룸
- Tokenizer와 Numericalize에 입력한 모든 인자는 TextBlock에도 입력 가능
    - DataBlock의 summary debug용으로 잘 활용하기!

### Language Model Using DataBlock


```python
def get_text_files_n(path, recurse=True, folders=None, n=100):
    if n:
        return get_files(path, extensions=['.txt'], recurse=recurse, folders=folders)[:n]
    else:
        return get_files(path, extensions=['.txt'], recurse=recurse, folders=folders)
```


```python
get_imdb = partial(get_text_files, folders=['train', 'test', 'unsup'])
```


```python
dls_lm = DataBlock(
    # not using class directly, but calling a "class method".
    blocks=TextBlock.from_folder(path, is_lm=True),
    get_items=get_imdb, splitter=RandomSplitter(0.1)
).dataloaders(path, path=path, bs=64, seq_len=80)
```


```python
# 원본 데이터를 볼 때
dls_lm.show_batch(max_n=2)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>text_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>xxbos xxmaj my main problem with the film is that it goes on too long . xxmaj other then that , it 's pretty good . xxmaj paul xxmaj muni plays a poor xxmaj chinese farmer who is about to get married through an arranged marriage . xxmaj luise xxmaj rainer is a servant girl who gets married to xxmaj muni . xxmaj they live with xxmaj muni 's father on a farm and they are doing pretty bad .</td>
      <td>xxmaj my main problem with the film is that it goes on too long . xxmaj other then that , it 's pretty good . xxmaj paul xxmaj muni plays a poor xxmaj chinese farmer who is about to get married through an arranged marriage . xxmaj luise xxmaj rainer is a servant girl who gets married to xxmaj muni . xxmaj they live with xxmaj muni 's father on a farm and they are doing pretty bad . xxmaj</td>
    </tr>
    <tr>
      <th>1</th>
      <td>comedic directing skills . xxmaj they credit three writers , including writer genius xxmaj charles xxmaj addams , but besides the characters i ca n't find his element int his movie anywhere . xxmaj out of the other two actors , one never worked on comedy before and the other was responsible for such drivel as xxmaj casper and xxmaj richie xxmaj rich , which is comedy but is extremely childish . xxmaj those are the only comedy movies credited</td>
      <td>directing skills . xxmaj they credit three writers , including writer genius xxmaj charles xxmaj addams , but besides the characters i ca n't find his element int his movie anywhere . xxmaj out of the other two actors , one never worked on comedy before and the other was responsible for such drivel as xxmaj casper and xxmaj richie xxmaj rich , which is comedy but is extremely childish . xxmaj those are the only comedy movies credited to</td>
    </tr>
  </tbody>
</table>



```python
x, y = dls_lm.one_batch()
x.shape, y.shape
```




    (torch.Size([64, 80]), torch.Size([64, 80]))




```python
# tensor 자체를 보고 싶을 때
dls_lm.one_batch()
```




    (LMTensorText([[    2,    19,   133,  ...,  1451,    14,    10],
                   [   61,    11,    12,  ..., 12585,    48,    58],
                   [   46,    66,   806,  ...,   249,    27,     9],
                   ...,
                   [   26,     8,   600,  ...,     9,    84,    12],
                   [   18,   651,    15,  ...,    17,    15,   123],
                   [   10,    26,     8,  ...,    10,     8, 12607]], device='mps:0'),
     TensorText([[   19,   133,    98,  ...,    14,    10,     8],
                 [   11,    12,   116,  ...,    48,    58,  1590],
                 [   66,   806,    12,  ...,    27,     9,   471],
                 ...,
                 [    8,   600,   670,  ...,    84,    12,     9],
                 [  651,    15,  1026,  ...,    15,   123,    60],
                 [   26,     8,     9,  ...,     8, 12607,     8]], device='mps:0'))



### Fine-Tuning the Language Model

1. What is "perplexity"?
- the exponential of the loss
    - i.e. torch.exp(cross_entrophy)
- often used in NLP for language modeling
- test set에 대한 loss로 사용 가능!


```python
learn = language_model_learner(
    dls_lm, AWD_LSTM, drop_mult=0.3, 
    metrics=[accuracy, Perplexity()]).to_fp16()
```


```python
# learn.fit_one_cycle(1, 2e-2)
```


```python
learn = learn.load('/Users/ridealist/Desktop/Jupyter_notebook/fastbook/clean/models/fitonecycle')
```

### Text Generation


```python
TEXT = "I hate this movie because"
N_WORDS = 50
N_SENTENCES = 2
preds = [learn.predict(TEXT, N_WORDS, temperature=0.75) 
         for _ in range(N_SENTENCES)]
```


```python
# not trained yet
print("\n".join(preds))
```

    i liked this movie because of its similarities to The Prince of Persia . The film is also known to have been influenced by the film Bail Out ( 1941 ) by Jack Nicholson . The film 's hints at the impact of the King
    i liked this movie because of its similarities to the Doctor Who novel The Elder Scrolls IV : Oblivion , and pitched Iron Man 4 to the Academy of Motion Picture Arts and Sciences . The film was originally described



```python
# 0.1K data trained
# print("\n".join(preds))
```

    i liked this movie because it was a movie for the time it was made and watched by fans and fans of movies that were not classic movies . So as to make a film that 's a movie set in the JFK
    i liked this movie because i was paid $ 6 million to write a film . That was just me . And i did well . i found it very satisfying and i was so excited at how to use the movie to



```python
# 1K data trained
# print("\n".join(preds))
```

    i liked this movie because it was very basic for me ! i was almost aware of it . i was already a fan of John Wayne ( this was Larry ) . i had a great idea who was an actor
    i liked this movie because of it . When i saw the movie i was forced to see the story . i kept watching this movie and seeing it … the people who were right there did n't n't get this movie .



```python
# 10K data trained
# print("\n".join(preds))
```

    i liked this movie because it 's not yet really bad . It is so bad that i should have been interested in it . I went on to see this movie , but i thought that was something i could n't do . So i was disappointed . So ,
    i liked this movie because at the time it was one of the worst films ever made . I 'm not sure how this movie is a masterpiece of this genre . It was a shame that i saw a movie like this . It did not make a good movie .



```python
# Fully trained
## pos
print("\n".join(preds))
```

    i liked this movie because of its intelligent of wonderfully a film . It was the first time that an variation had been produced . This is a superhero theme of the film , and it is not a " intelligent " . Although the film does not have a word or
    i liked this movie because of its composed . He wrote that it was " lacked " , and said that it " was like a big m thing when you 're talking to you . " Although the film was shot through a studio , it was not disgruntled filmed .



```python
# Fully trained
## neg
print("\n".join(preds))
```

    i hate this movie because i don ' lesbian know whether it 's similar to the one that God has said [ in the film , but i think it 's really important to see it . It 's just a little bit more like the personal story of the Al of
    i hate this movie because of its portrayal of a high - betrayal student . The Sons of Biographical Examination Police Department ( JINGLE ) , the United Quick Department of Prancing ( HAREM ) , and PERPETUATING Public Works


### Saving and Loading Models


```python
# learn.save('1epoch')
```




    Path('/Users/ridealist/.fastai/data/imdb/models/1epoch.pth')




```python
# learn = learn.load('1epoch')
```


```python
# learn.unfreeze()
# learn.fit_one_cycle(10, 2e-3)
```


```python
# learn.save_encoder('finetuned')
```

### Creating the Classifier DataLoaders


```python
dls_clas = DataBlock(
            # no longer has the "is_lm=True" params / pass "vocab" created for the language model fine-tuning
    blocks=(TextBlock.from_folder(path, vocab=dls_lm.vocab),CategoryBlock),
    get_y = parent_label,
    get_items=partial(get_text_files, folders=['train', 'test']),
    splitter=GrandparentSplitter(valid_name='test')
).dataloaders(path, path=path, bs=128, seq_len=72)
```

1. Why do we have to pass the vocabulary of the language model to the classifier data block?
    - token indexing에 사용한 것과 똑같은 vocab을 적용하기 위해서
    - embedding이 위 vocab에 의해서 fine-tuned 되었기 때문에, 같은 vocab을 사용해야만 embedding matrix가 의미가 있음


```python
dls_clas.show_batch(max_n=3)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>xxbos xxmaj match 1 : xxmaj tag xxmaj team xxmaj table xxmaj match xxmaj bubba xxmaj ray and xxmaj spike xxmaj dudley vs xxmaj eddie xxmaj guerrero and xxmaj chris xxmaj benoit xxmaj bubba xxmaj ray and xxmaj spike xxmaj dudley started things off with a xxmaj tag xxmaj team xxmaj table xxmaj match against xxmaj eddie xxmaj guerrero and xxmaj chris xxmaj benoit . xxmaj according to the rules of the match , both opponents have to go through tables in order to get the win . xxmaj benoit and xxmaj guerrero heated up early on by taking turns hammering first xxmaj spike and then xxmaj bubba xxmaj ray . a xxmaj german xxunk by xxmaj benoit to xxmaj bubba took the wind out of the xxmaj dudley brother . xxmaj spike tried to help his brother , but the referee restrained him while xxmaj benoit and xxmaj guerrero</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>1</th>
      <td>xxbos * ! ! - xxup spoilers - ! ! * \n\n xxmaj before i begin this , let me say that i have had both the advantages of seeing this movie on the big screen and of having seen the " authorized xxmaj version " of this movie , remade by xxmaj stephen xxmaj king , himself , in 1997 . \n\n xxmaj both advantages made me appreciate this version of " the xxmaj shining , " all the more . \n\n xxmaj also , let me say that xxmaj i 've read xxmaj mr . xxmaj king 's book , " the xxmaj shining " on many occasions over the years , and while i love the book and am a huge fan of his work , xxmaj stanley xxmaj kubrick 's retelling of this story is far more compelling … and xxup scary . \n\n xxmaj kubrick</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>xxbos xxmaj warning : xxmaj does contain spoilers . \n\n xxmaj open xxmaj your xxmaj eyes \n\n xxmaj if you have not seen this film and plan on doing so , just stop reading here and take my word for it . xxmaj you have to see this film . i have seen it four times so far and i still have n't made up my mind as to what exactly happened in the film . xxmaj that is all i am going to say because if you have not seen this film , then stop reading right now . \n\n xxmaj if you are still reading then i am going to pose some questions to you and maybe if anyone has any answers you can email me and let me know what you think . \n\n i remember my xxmaj grade 11 xxmaj english teacher quite well . xxmaj</td>
      <td>pos</td>
    </tr>
  </tbody>
</table>


#### 10개 문서로 미니배치 생성하는 방법 예시


```python
nums_samp = toks200[:10].map(num)
```


```python
nums_samp.map(len)
```




    (#10) [139,152,226,605,262,242,179,101,373,309]




```python
x, y = dls_clas.one_batch();
x.shape, y.shape
```




    (torch.Size([128, 3345]), torch.Size([128]))




```python
learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, 
                                metrics=accuracy).to_fp16()
```


```python
learn = learn.load_encoder('/Users/ridealist/Desktop/Jupyter_notebook/fastbook/clean/models/finetuned')
```

1. Why do we need padding for text classification?
2. Why don't we need it for language modeling?
    - ‘PyTorch DataLoaders need to collate all the items in a batch into a single tensor, and a single tensor has a fixed shape
        - (i.e., it has a particular length on every axis, and all items must be consistent)
    - Other approaches. like cropping or squishing, either to negatively affect training or do not make sense in this context.
    - 가장 작은 길이의 text를 모두 같은 사이즈로 맞춰야 함
        - fastai, special pedding token을 사용함 - model이 자체적으로 무시함
    - language model은 사용하지 않음
        - documemnts들이 모두 all concatenated 되어 dataset으로 사용되기 때문

1. What does an embedding matrix for NLP contain? What is its shape?
    - vocab에 있는 모든 토큰들의 vector representations 들을 포함
    - input vector들의 embedding 값들
    - shape : (vocab_size * embedding_size)
        - vocab_size is the length of the vocabulary
        - embedding_size is an arbitrary number defining the number of latent factors of the tokens



### Fine-Tuning the Classifier


1. What is "gradual unfreezing"?
    - in NLP, 각각 구별되는 학습률로 학습하는 방식
        - "unfreezing one layer at a time and fine-tuning the pretrained model"
    - NLP classifier에선, 몇몇의 layer들을 unfreezing 하는 방식이 더 효과가 좋음
        - CV에서는 model을 한번에 unfreeze 주로 함

1. Why is text generation always likely to be ahead of automatic identification of machine-generated texts?
    - To build a state-of-the art classifier, we used a pretrained language model, fine-tuned it to the corpus of our task
    - then used its body (the encoder) with a new head to do the classification.
        - The classification models could be used to improve text generation algorithms (evading the classifier) so the text generation algorithms will always be ahead.


```python
learn = learn.load('/Users/ridealist/Desktop/Jupyter_notebook/fastbook/clean/models/classifier')
```


```python
learn.fit_one_cycle(1, 2e-2)
```


```python
# 파라마터 그룹 중 마지막 2개만 freezing 해제
# learn.freeze_to(-2)
# learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2))
```


```python
# 파라미토 그룹 중 마지막 3개만 freezing 해제
# learn.freeze_to(-3)
# learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3))
```


```python
# learn.unfreeze()
# learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3))
```

### Text Sentiment Classification


```python
text1 = "I liked this movie."
text2 = "I hate this movie."
text3 = "I don't like this movie."
text4 = "This movie sucks."
text5 = "It's cool."

pred1 = learn.predict(text1)
pred2 = learn.predict(text2)
pred3 = learn.predict(text3)
pred4 = learn.predict(text4)
pred5 = learn.predict(text5)
```


```python
pred1, pred2, pred3, pred4, pred5
```




    (('pos', tensor(1), tensor([0.0551, 0.9449])),
     ('neg', tensor(0), tensor([0.6324, 0.3676])),
     ('neg', tensor(0), tensor([0.9806, 0.0194])),
     ('neg', tensor(0), tensor([9.9955e-01, 4.4647e-04])),
     ('pos', tensor(1), tensor([0.0210, 0.9790])))




```python
# before train
pred1, pred2
```




    (('neg', tensor(0), tensor([0.9471, 0.0529])),
     ('neg', tensor(0), tensor([0.9944, 0.0056])))



## Disinformation and Language Models

## Conclusion

## Questionnaire

1. What is "self-supervised learning"?
1. What is a "language model"?
1. Why is a language model considered self-supervised?
1. What are self-supervised models usually used for?
1. Why do we fine-tune language models?
1. What are the three steps to create a state-of-the-art text classifier?
1. How do the 50,000 unlabeled movie reviews help us create a better text classifier for the IMDb dataset?
1. What are the three steps to prepare your data for a language model?
1. What is "tokenization"? Why do we need it?
1. Name three different approaches to tokenization.
1. What is `xxbos`?
1. List four rules that fastai applies to text during tokenization.
1. Why are repeated characters replaced with a token showing the number of repetitions and the character that's repeated?
1. What is "numericalization"?
1. Why might there be words that are replaced with the "unknown word" token?
1. With a batch size of 64, the first row of the tensor representing the first batch contains the first 64 tokens for the dataset. What does the second row of that tensor contain? What does the first row of the second batch contain? (Careful—students often get this one wrong! Be sure to check your answer on the book's website.)
1. Why do we need padding for text classification? Why don't we need it for language modeling?
1. What does an embedding matrix for NLP contain? What is its shape?
1. What is "perplexity"?
1. Why do we have to pass the vocabulary of the language model to the classifier data block?
1. What is "gradual unfreezing"?
1. Why is text generation always likely to be ahead of automatic identification of machine-generated texts?

### Further Research

1. See what you can learn about language models and disinformation. What are the best language models today? Take a look at some of their outputs. Do you find them convincing? How could a bad actor best use such a model to create conflict and uncertainty?
1. Given the limitation that models are unlikely to be able to consistently recognize machine-generated texts, what other approaches may be needed to handle large-scale disinformation campaigns that leverage deep learning?


```python

```
