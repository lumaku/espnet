#!/usr/bin/env bats

setup() {
    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)
    cat << EOF > $tmpdir/base.json
{
    "utts": {
        "uttid": {
            "input": [
                {
                    "feat": "aaa.ark:123", 
                    "name": "input1", 
                    "shape": [
                        100, 
                        80
                    ]
                }
            ], 
            "output": [
                {
                    "name": "target1", 
                    "shape": [
                        10, 
                        26
                    ], 
                    "text": "あいうえお", 
                    "token": "あ い う え お", 
                    "tokenid": "0 1 2 3 4"
                }
            ], 
            "utt2spk": "foobar"
        }
    }
}
EOF

    cat << EOF > $tmpdir/aux.json
{
    "utts": {
        "uttid": {
                "feat": "bbb.ark:456", 
                "ilen": 40, 
                "idim": 80
        }
    }
}
EOF

    cat << EOF > $tmpdir/valid
{
    "utts": {
        "uttid": {
            "input": [
                {
                    "feat": "aaa.ark:123",
                    "name": "input1",
                    "shape": [
                        100,
                        80
                    ]
                },
                {
                    "feat": "bbb.ark:456",
                    "name": "input2",
                    "shape": [
                        40,
                        80
                    ]
                }
            ],
            "output": [
                {
                    "name": "target1",
                    "shape": [
                        10,
                        26
                    ],
                    "text": "あいうえお",
                    "token": "あ い う え お",
                    "tokenid": "0 1 2 3 4"
                }
            ],
            "utt2spk": "foobar"
        }
    }
}
EOF

}

teardown() {
    rm -rf $tmpdir
}

@test "" {
    python $utils/addjson.py $tmpdir/base.json $tmpdir/aux.json> $tmpdir/out.json
    cat $tmpdir/out.json
    jsondiff $tmpdir/out.json $tmpdir/valid
}
