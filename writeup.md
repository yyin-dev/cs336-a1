## Problem (unicode1)
```
>>> chr(0)
'\x00'
>>> repr(chr(0))
"'\\x00'"
>>> print(chr(0))

>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
```
(a) `chr(0)` returns the unicode with code point 0. 
(b) The string representation (__repr__()) is "'\\x{HEX}'", whereas the printed representation is invisible.
(c) When it occurs in text, it's invisible. It's unprintable but stringable.

## Problem (unicode2)
(a)
```
>>> len(list("hello".encode("utf-8")))
5
>>> len(list("hello".encode("utf-16")))
12
>>> len(list("hello".encode("utf-32")))
24
```
The output is shorter.

(b) This is decoding at byte level. However, each character might map to more than one byte in UTF-8. 

```
>>> decode_utf8_bytes_to_str_wrong("你好".encode("utf-8"))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in decode_utf8_bytes_to_str_wrong
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data
```
(c) Utf-8 specifies that 1-byte sequence must be `0xxxxxxx`, and 2-byte sequence must starts with `110xxxxx`, and 3-byte sequence must starts with `1110xxxx`. A 2-byte sequence starting with `1110xxxx` is invalid.
```
>>> bytes([254,254]).decode('utf-8')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xfe in position 0: invalid start byte
```





## Problem (train_bpe_tinystories)

```
$ uv run -m src.train_bpe train ../data/TinyStoriesV2-GPT4-train.txt ../ts-train-bpe.pkl 10000
$ uv run -m src.train_bpe read ../ts-train-bpe.pkl
```

(a) It takes about 7mins on CPU. The longest token in the vocabulary is `b' accomplishment'`. 

(b) It spends a lot of time in finding the next token with the highest frequency. We could make this more efficient using a heap / priority-queue.

```
   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
196097110   20.773    0.000   35.846    0.000 bpe.py:74(comp_frequency_then_pair)
    19525   17.578    0.001   53.424    0.003 {built-in method builtins.max}
196097220   15.073    0.000   15.073    0.000 {method 'get' of 'dict' objects}
       24    0.472    0.020    0.472    0.020 {built-in method posix.read}
        1    0.386    0.386   54.416   54.416 bpe.py:81(train_bpe)
       10    0.080    0.008    0.080    0.008 {built-in method _io.open}
```



## Problem (train_bpe_expts_owt)

```
$ uv run -m src.train_bpe train ../data/owt_valid.txt ../owt-val-bpe.pkl 32000
```

Training on the validation took about 53 minutes, using a peak memory of around 3G. The longest token in the vocabulary is `b'abc'`.
