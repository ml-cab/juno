(ch-6-1)=
# 6.1. On-Prem Cluster

Run `juno-master` as the coordinator and `juno-node` on each worker, connected by gRPC (systemd
or your own process manager). Parallelism modes and byte-order flags match the local cluster
harness behavior described in [Cluster mode](#ch-3-4); topology and
components are in [Distributed inference](#ch-2-2). The AWS
automation under `scripts/aws/` is optional cloud packaging of the same two roles; see
[AWS deployment](#ch-6-2) if you want that instead of managing hosts directly.

## Roles

- **`juno-master`**: shaded coordinator jar. Runs `RequestScheduler`, `GenerationLoop`, and the
  REST/health surface.
- **`juno-node`**: shaded worker jar. Loads one model shard and serves forward-pass RPCs to the
  coordinator.

## Configuration parity with local cluster mode

All flags, environment variable overrides, and byte-order behavior described in
[CLI flags](#ch-3-2) apply the same way whether the cluster is forked
in-process by `./juno` or deployed as separate `juno-master` / `juno-node` processes on
different hosts. In particular:

- `--pType pipeline|tensor` selects the same two distribution strategies described in
  [Distributed inference](#ch-2-2).
- `--byteOrder BE|LE` must match across every process; propagate it through your process
  manager's environment configuration the same way `ClusterHarness` and `juno-deploy.sh` do
  automatically for the built-in launchers.
- `--lora-play PATH` applies a pre-trained adapter read-only on every node; see
  [Inference with a trained adapter](#ch-4-4).

## See also

- [Chapter 3.4 -- Cluster Mode](#ch-3-4)
- [Chapter 2.2 -- Distributed Inference](#ch-2-2)
- [Chapter 6.2 -- AWS Deployment](#ch-6-2)
- [Chapter 6.3 -- Windows Notes](#ch-6-3)

---

[<- 5.4 OpenAPI Spec](#ch-5-4) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [6.2 AWS Deployment ->](#ch-6-2)
