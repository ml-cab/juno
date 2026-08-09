(ch-07)=
# 7. AWS Deployment: Cluster Lifecycle and Free-Tier GPU Quotas

On-prem orchestration runs `juno-master` as the coordinator and `juno-node` on each worker with
gRPC between them (systemd or your own process manager), using the parallelism modes described
in [Chapter 2](#ch-02). Automation under `scripts/aws/` is optional cloud packaging of the same
roles, built around one script: `juno-deploy.sh`.

![A Juno chat session running on a deployed AWS cluster](../assets/aws-chat-deployed.png)

## Cluster lifecycle commands

```bash
./launcher.sh juno-deploy.sh setup      [options]
./launcher.sh juno-deploy.sh start
./launcher.sh juno-deploy.sh stop
./launcher.sh juno-deploy.sh teardown
./launcher.sh juno-deploy.sh status
./launcher.sh juno-deploy.sh scan-regions
```

**Setup options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--instance-type TYPE` | `g4dn.xlarge` | EC2 instance type |
| `--node-count N` | `3` | Number of inference nodes |
| `--coordinator node1\|separate` | `node1` | Co-located or separate coordinator |
| `--model-url URL` | TinyLlama Q4_K_M | Model to download during bootstrap |
| `--ptype pipeline\|tensor` | `pipeline` | Parallelism type |
| `--dtype FLOAT32\|FLOAT16` | `FLOAT16` | Activation wire format |
| `--jfr DURATION` | — | JFR on all JVMs (e.g. `5m`) |
| `--lora-play PATH` | — | Local path to a `.lora` file. Must be absolute or relative to working directory — resolved via `realpath`. The file is SCPed to every node after bootstrap. |

**GPU quota:** the script checks EC2 quota `L-DB2E81BA` before launching. If the quota in
vCPUs is less than `node-count x vCPUs-per-instance`, setup fails immediately with the
shortfall and a link to the Service Quotas console. It never silently reduces node count.

**GPU on AWS instances:** pre-installed in the golden AMI by `make-ami.sh`. Node bootstrap runs
`lspci` to detect the GPU vendor and sets `JUNO_USE_GPU=true` — no DKMS compilation at boot.

- **NVIDIA (g4dn, g5, g6, p\*):** CUDA 12.3 + nvidia-open. Backend auto-selects CUDA.
- **AMD Radeon (g4ad):** ROCm 7.2.4 + amdgpu-dkms. The AMI sets `HSA_OVERRIDE_GFX_VERSION=10.1.0`
  in `/etc/environment` to work around the missing gfx1011 rocBLAS kernels on the Radeon Pro
  V520 (upstream issue ROCm/rocm-libraries#4347); rocBLAS uses the gfx1010 dispatch path, which
  runs correctly on Navi12 silicon. Backend auto-selects ROCm when CUDA libraries are absent.

## LoRA deploy flow

```bash
# Train locally
./juno lora --model-path /path/to/model.gguf
you > /train-qa What is my name? A: Dima
you > /save

# Deploy to AWS with adapter
cd scripts/aws
./launcher.sh juno-deploy.sh setup \
  --instance-type m7i-flex.large \
  --model-url https://huggingface.co/.../tinyllama-1.1b-chat-v1.0-q4_k_m.gguf \
  --lora-play /absolute/path/to/model.lora
```

After all nodes finish bootstrap and before starting the coordinator, `_scp_lora_to_nodes()`
stops each `juno-node.service` synchronously, SCPs the file to `/opt/juno/models/`, patches
`JUNO_LORA_PLAY_PATH` in `/etc/juno/node.env`, and restarts the service. The coordinator only
starts after all nodes are confirmed active.

**Expected coordinator log:**

```
INFO: LoRA inference overlay configured -- nodes will load:
      /opt/juno/models/tinyllama-1.1b-chat-v1.0-q4_k_m.lora
```

**Expected node log:**

```
INFO: Detected architecture: llama  backend=CpuMatVec  file=...  lora=44 adapters
```

## AWS cluster JFR

```bash
./launcher.sh juno-deploy.sh setup --jfr 2m ...
# Ctrl+C -> recordings collected from all nodes -> metrics printed -> instances stopped
```

See [Chapter 18](#ch-18) for how to read the resulting metrics against the published
performance matrix, and the AWS performance runner that automates whole rows of it.

## Free-tier GPU quotas: a walkthrough

The AWS Free Plan explicitly restricts high-performance instances. High-spec instance types
like `g4dn.xlarge` or `g4ad.2xlarge` are not eligible for the free plan by default, so a quota
increase must be requested first.

`g4dn.xlarge` is NVIDIA's 4 vCPU, 16 GiB instance type; for a test cluster, two of them is
enough.

**For NVIDIA hardware (`g4dn.xlarge`):**

```bash
aws service-quotas request-service-quota-increase \
  --service-code ec2 --quota-code L-DB2E81BA \
  --desired-value 12 --region eu-north-1
```

**For AMD Radeon hardware (`g4ad.2xlarge`):**

```bash
aws service-quotas request-service-quota-increase \
  --service-code ec2 --quota-code L-1216C47A \
  --desired-value 60 --region eu-north-1
```

The response looks like:

```json
{
    "RequestedQuota": {
        "Id": "1234567890abcdefghijklmnopqrstuvwxyz0987",
        "ServiceCode": "ec2",
        "ServiceName": "Amazon Elastic Compute Cloud (Amazon EC2)",
        "QuotaCode": "L-1216C47A",
        "QuotaName": "Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances",
        "DesiredValue": 60.0,
        "Status": "PENDING",
        "Created": "2026-06-04T22:17:29.313000+03:00",
        "GlobalQuota": false,
        "Unit": "None",
        "QuotaRequestedAtLevel": "ACCOUNT"
    }
}
```

To verify NVIDIA quotas later:

```bash
aws service-quotas list-requested-service-quota-change-history \
  --service-code ec2 --region eu-north-1 \
  --query "RequestedQuotas[?QuotaCode=='L-DB2E81BA'].[Status,DesiredValue,Created]" \
  --output table
```

Or for AMD Radeon quotas:

```bash
aws service-quotas list-requested-service-quota-change-history \
  --service-code ec2 --region eu-north-1 \
  --query "RequestedQuotas[?QuotaCode=='L-1216C47A'].[Status,DesiredValue,Created]" \
  --output table
```

which outputs something like:

```
-------------------------------------------------------------
|          ListRequestedServiceQuotaChangeHistory           |
+--------------+-------+------------------------------------+
|  CASE_OPENED |  12.0 |  2026-04-02T01:56:51.160000+03:00  |
+--------------+-------+------------------------------------+
```

The standard free plan is also limited from accessing a subset of AWS services and offerings
that would immediately consume the entire Free Tier credit amount, and GPU instances fall
squarely into that category. To unlock them:

1. Go to **AWS Billing and Cost Management Console** → `https://console.aws.amazon.com/billing/`
2. Click **"Upgrade Plan"** — it's in the navigation bar or the Cost and Usage widget on the
   home dashboard.
3. Confirm the upgrade.

When you upgrade to the paid plan, remaining Free Tier credits automatically apply to future
AWS bills until they expire — nothing is given up, GPU access is simply unlocked.

One heads-up on budget: 2× `g4dn.xlarge` at $0.526/hr burns roughly **$1.05/hr**. A $100 credit
gives roughly 95 hours of runtime, so consider a stop/start workflow (`juno-deploy.sh stop` /
`start`) rather than leaving the cluster running idle.

---

[← Chapter 6: JVM Integration](#ch-06) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [Chapter 8: LoRA Fundamentals →](#ch-08)
