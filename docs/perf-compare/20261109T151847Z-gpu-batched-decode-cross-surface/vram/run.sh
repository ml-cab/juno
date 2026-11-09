#!/usr/bin/env bash
# Re-runs the two free-VRAM test classes alone, with per-process GPU memory sampled once a second.
X=<scratch>/vram
cd <repo>
rm -f $X/stop; bash $X/sample.sh $X/stop > $X/samples-$1.txt & 
mvn -o test -pl node -am -Dgroups=gpu -Dtest='GpuAttentionHandlerParityTest,PrefillRegionHandlerParityTest' -Dsurefire.failIfNoSpecifiedTests=false > $X/mvn-$1.log 2>&1
rc=$?
touch $X/stop; sleep 2
echo "run $1 exit=$rc" >> $X/result.txt
