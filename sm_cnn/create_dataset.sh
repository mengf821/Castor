#!/bin/sh
mkdir -p data
python overlap_features.py --dir ../../Castor-data/datasets/TrecQA/

CURRENT_DIR=$(pwd)
cd ../../Castor-data/datasets/TrecQA
cd raw-dev/; paste id.txt sim.txt a.toks b.toks overlap_feats.txt > $CURRENT_DIR/data/trecqa.dev.tsv; cd ..
cd raw-test/; paste id.txt sim.txt a.toks b.toks overlap_feats.txt > $CURRENT_DIR/data/trecqa.test.tsv; cd ..
cd train-all/; paste id.txt sim.txt a.toks b.toks overlap_feats.txt > $CURRENT_DIR/data/trecqa.train.tsv; cd ..
cd $CURRENT_DIR

python overlap_features.py --dir ../../Castor-data/datasets/WikiQA/
cd ../../Castor-data/datasets/WikiQA
cd dev/; paste id.txt sim.txt a.toks b.toks overlap_feats.txt > $CURRENT_DIR/data/wikiqa.dev.tsv; cd ..
cd test/; paste id.txt sim.txt a.toks b.toks overlap_feats.txt > $CURRENT_DIR/data/wikiqa.test.tsv; cd ..
cd train/; paste id.txt sim.txt a.toks b.toks overlap_feats.txt > $CURRENT_DIR/data/wikiqa.train.tsv; cd ..
cd $CURRENT_DIR