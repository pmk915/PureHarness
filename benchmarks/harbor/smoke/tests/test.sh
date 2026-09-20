#!/usr/bin/env bash

set -u

mkdir -p /logs/verifier

if [ -f /app/answer.txt ] && \
   [ "$(cat /app/answer.txt)" = "PUREHARNESS_OK" ]; then
    echo "Smoke verification passed."
    echo 1 > /logs/verifier/reward.txt
else
    echo "Smoke verification failed."
    echo 0 > /logs/verifier/reward.txt
fi
