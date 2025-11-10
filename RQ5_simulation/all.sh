#!/bin/bash

# 你可以传递任何额外的参数
language=python
idx=100

# 启动 script2.sh，并传递额外的参数
bash 0_TRACE.sh "$language" "$idx"
bash 1_woinvoker.sh "$language" "$idx"
bash 2_enriched.sh "$language" "$idx"
bash 3_plain.sh "$language" "$idx"
bash 4_coedpilot.sh "$language" "$idx"
bash 5_ccd.sh "$language" "$idx"
