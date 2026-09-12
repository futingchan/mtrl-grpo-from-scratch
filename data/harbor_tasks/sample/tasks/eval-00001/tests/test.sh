#!/bin/bash
mkdir -p /logs/verifier
python /grader/grade.py --answer /workdir/answer.txt --gold /tests/gold.json > /logs/verifier/reward.txt
