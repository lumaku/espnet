
## Sinc convolutions
To enable sinc convolutions, use prefix "sinc" in the encoder type (e.g. "sincblstmp"). Sinc convolutions are implemented in [espnet/nets/pytorch_backend/sincconv.py](../../../espnet/nets/pytorch_backend/sincconv.py).

Sinc convolutions require raw audio data as input. [utils/make_raw.sh](../../../utils/make_raw.sh) uses [utils/compute-raw-feats.py](../../../utils/compute-raw-feats.py) (utils refers to the folder in the root of the git repo) to generate raw "features", and replaces [steps/make_fbank_pitch.sh](steps/make_fbank_pitch.sh) for feature extraction. This script requires a config file, see [conf/raw.yaml](conf/raw.yaml).

## Running
Make sure to run steps 0-2 to prepare the raw audio data as features, step 3 optionally to train a LM, and step 4 and 5 for training and decoding. Example:

```
./run.sh --stop-stage 2
./run.sh --stage 4
```

Note that we decode with GPU by default. You might have to change the batchsize in your decoding configuration ([conf/decode.yaml](conf/decode.yaml)), or change [run.sh](run.sh) in stage 5 to use CPU decoding instead (set ngpu to 0 and nj to something higher, the relevant parts are marked with "CPU decoding"). The current configuration should run on a GPU with 12GB of memory.

A link to the pre-trained model and it's decoding results is listed in [RESULTS.md](RESULTS.md). This model was trained with tag `deeper`. To decode with it, run:

```
./run.sh --stop-stage 2
./run.sh --stage 5 --tag "deeper"
```



