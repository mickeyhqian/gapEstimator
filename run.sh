#!/bin/zsh

# python exp.py linear GapCRN SRP 100 n1=64 n2=36 &
# python exp.py linear GapCRN I2RP 100 n1=64 n2=36 &
# python exp.py linear GapCRN A2RP 100 n1=64 n2=36 &
# python exp.py linear GapCRN Batch 100 n1=64 n2=36 &
python exp.py linear GapCRN BagU 100 n1=64 n2=36 B=500 &
python exp.py linear GapCRN BagV 100 n1=64 n2=36 B=500 &