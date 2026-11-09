#!/usr/bin/env bash
# Final GPU pass on the final jar, one workload at a time: the cluster batch, then the two free-VRAM test classes repeated.
X=<scratch>; R=<repo>
P=$X/final-progress.txt; : > $P
python3 -I $X/cluster-batch.py $R $R/juno-player/target/juno-player-0.1.2-shaded.jar $X/cluster-batch 18096 "--parallel on a cluster decodes a batch" "could you please write me a short poem about love and war" > $X/cluster-batch.log 2>&1
echo "cluster-batch exit=$?" >> $P
sleep 8
for i in 2 3 4 5; do bash $X/vram/run.sh $i; echo "vram-$i done" >> $P; sleep 8; done
echo "ALL DONE" >> $P
