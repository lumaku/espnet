# SincConv + specaug + large LM

  - Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/open?id=10Y0SsEaJWseghWij07MQH5oSIO8o74s_
    - training config file: `conf/tuning/train_sinc.yaml`
    - decoding config file: `conf/tuning/decode_rnn.yaml`
    - preprocess config file: `conf/specaug.yaml`
    - cmvn file: `data/train_trim_sp/cmvn.ark`
    - e2e file: `exp/train_trim_sp_pytorch_deeper/results/model.acc.best`
    - e2e JSON file: `exp/train_trim_sp_pytorch_deeper/results/model.json`
    - lm file: `exp/train_rnnlm_pytorch_lm_unigram500/rnnlm.model.best`
    - lm JSON file: `exp/train_rnnlm_pytorch_lm_unigram500/model.json`
    - dict file: `data/lang_char/train_trim_sp_unigram500_units.txt`
  - Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)
```
exp/train_trim_sp_pytorch_deeper/decode_dev_decode/result.wrd.txt
| SPKR                      | # Snt # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg                   |  507  17783 | 90.9    6.2    2.9    1.6   10.7   80.9 |
exp/train_trim_sp_pytorch_deeper/decode_test_decode/result.wrd.txt
| SPKR                  | # Snt # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg               | 1155  27500 | 90.6    5.8    3.6    1.3   10.7   75.2 |
```

Link to the decoding results: https://drive.google.com/open?id=1hA3biwJxEv2Qrp9zcA_BwxA7yJeGA7vV
