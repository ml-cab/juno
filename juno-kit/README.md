# Juno hardware validation kit

The kit runs Juno's correctness checks and performance readings on your machine and packs the results into
one zip file for you to send back. You start it with one command on Windows, Linux or macOS. It builds
Juno from the copy you downloaded, works out what your machine can run, lets you choose what to run, tells
you how long it will take, and shows progress while it runs. Nothing is uploaded: you decide whether to
send the zip.

## What you need

- **JDK 25 or later.** Check with `java -version`. Any vendor's build works; `JAVA_HOME` is used when set.
- **Maven 3.9 or later**, as `mvn` on `PATH` (`mvn.cmd` on Windows), or `MAVEN_HOME` set.
- **Network access** to Maven Central (for the build) and huggingface.co (for the models).
- **Disk:** 2 GiB free for the build, plus the models: 637 MiB for the `quick` suite and about 12 GiB for
  the others.
- **Memory:** most `quick` probes need 8 GiB of RAM, and one needs 10 GiB. The larger models' probes need
  12 to 20 GiB. A probe that does not fit is listed as not runnable, with the reason, and the others still run.
- **GPU (optional):** an NVIDIA GPU with its driver and the CUDA 12.x runtime on Linux, or an AMD GPU with
  ROCm 6 or later on Linux. On Windows and macOS the GPU probes are listed as not runnable: Juno's GPU
  backends do not run there yet. The CPU probes still run.

No administrator rights are needed, and nothing is installed outside the folder you download to and the
model cache.

## Get the code

Use the exact commit you were asked to test: results are compared with a reference run at that commit.

```bash
git clone https://github.com/ml-cab/juno.git
cd juno
git checkout <commit>
```

Without git, download the commit's archive from GitHub (the commit page, then "Browse files", then "Code",
then "Download ZIP") and unpack it. The kit reads the commit from the folder name.

## Run it

Linux and macOS:

```bash
./juno-kit.sh
```

Windows (Command Prompt or PowerShell):

```bat
juno-kit.bat
```

With no options the kit asks what to run. To run the short suite without questions:

```bash
./juno-kit.sh --suite quick --yes
```

To see what a suite would run, how long it would take and what it would download, without running,
downloading or building anything:

```bash
./juno-kit.sh --plan --suite all
```

`--plan` needs the kit built once. Run the kit once without `--plan` to build it, then stop it at the
confirmation prompt.

## Suites

| Suite | What it runs | Downloads | Time on the reference machine |
|---|---|---|---|
| `quick` | TinyLlama 1.1B only: CPU checks and readings, plus the GPU ones where a GPU is usable | 637 MiB | 32 to 45 min |
| `cpu` | every CPU probe on the six models | about 12 GiB | 10.5 to 15 h |
| `gpu` | every GPU probe on the six models | about 12 GiB | 1 to 1.5 h |
| `all` | `cpu` and `gpu` together | about 12 GiB | 11.5 to 16.5 h |
| `custom` | the probes you name with `--probes` | the models they need | depends |

The reference machine is a 6-core, 12-thread workstation CPU from 2013 (no AVX2) with an 8 GiB NVIDIA GPU. Your times
will differ. Before it asks you to start, the kit runs a calibration of about a minute and scales the
estimate to your machine. The estimate does not include downloads, which the kit times separately.

If you have a fixed amount of time, `--budget 2h` (or `45m`, or `1h30m`) runs the most useful probes that
fit. A probe left out is listed with the reason.

## While it runs

The kit tells you at each stage whether you may use the computer:

1. **Build and downloads:** you may use the computer.
2. **Calibration and measurements:** leave it idle. The kit samples other programs' CPU and GPU use. A
   reading taken while the machine was busy is re-run up to twice, then marked `NOISY`.

Also:

- Close games, 3D and video programs before the GPU probes.
- On a laptop, plug in. The kit warns when it runs on battery, because readings are slower and less steady
  and a long run can drain the battery.
- The kit warns when the connection is metered, and states the total download size.
- The kit keeps the machine from sleeping, without administrator rights: through the system's execution-state
  call on Windows, `caffeinate` on macOS, and `systemd-inhibit` on Linux where it is installed. When it cannot,
  it says so; then set your system not to sleep while the kit runs.

The kit shows an overall progress bar, the current probe, and the elapsed and remaining time. A full log
is always written to the run directory.

## If it stops

A failed, crashed or timed-out probe is recorded with its log, and the run goes on to the next probe. Your
results are useful whatever the probes found.

If the run itself is interrupted (you close the window, the machine restarts), continue it with:

```bash
./juno-kit.sh resume juno-kit-results/<run directory>
```

Probes that finished are not run again. To pack what an unfinished run has so far:

```bash
./juno-kit.sh pack juno-kit-results/<run directory>
```

If the build fails, the kit still writes a results zip with the build log and a description of the
machine. Please send that zip as well: a build failure on your system is a result.

## What to send back

At the end the kit prints the path of one file:

```
juno-kit-results/juno-kit-results-<os>-<arch>-<UTC time>.zip
```

Send that file to the person who asked you to run the kit. It holds:

- a manifest: the kit version, the Juno commit, the kit's own SHA-256, the suites run and the hardware;
- one result file and one log per probe, and the unit-test reports;
- `SUMMARY.txt`, a plain-text summary, and `report.html`, the same results as a page you can open in a
  browser.

Read `SUMMARY.txt` or `report.html` first if you want to see what you are sending.

## Privacy

Before the zip is written, every file in it is scrubbed:

- your user name, your home directory, the machine's host name, and every IP and MAC address are replaced
  with placeholders, in the logs and the test reports too;
- hardware model names (CPU, GPU) are kept, because they are what the results are about;
- JFR recordings, which hold details of the machine, are never packed.

The kit's only network use is the Maven build and the model downloads. It uploads nothing.

## Models

The kit downloads ungated models from Hugging Face. Each download comes from a fixed commit of its
repository, resumes if interrupted, and is checked against a pinned SHA-256 before use.

| Model | Size | Used by |
|---|---|---|
| TinyLlama 1.1B Chat, Q4_K_M | 637 MiB | every suite |
| Qwen2.5-3B Instruct, Q4_K_M | 2.0 GiB | `cpu`, `gpu`, `all` |
| Phi-3.5-mini Instruct, Q4_K_M | 2.2 GiB | `cpu`, `gpu`, `all` |
| Mistral 7B Instruct v0.1, Q4_K_M | 4.1 GiB | `cpu`, `gpu`, `all` |
| Qwen3-1.7B, Q4_K_M | 1.0 GiB | `cpu`, `gpu`, `all` |
| moondream2, Q5_K | 1.8 GiB | `cpu`, `gpu`, `all` (image to text) |

Downloads go to `~/.cache/juno/models/kit` by default; change it with `--cache-dir <dir>`. If you already
have some of these files, pass `--models-dir <dir>`. The kit uses a file there when its size and hash
match, and never writes to that folder. The LoRA probes train their own small adapter during the run, so
no adapter is downloaded.

## Options

```
juno-kit.sh [options]            run probes (asks what to run when a console is attached)
juno-kit.sh resume <run dir>     continue an interrupted run
juno-kit.sh pack <run dir>       pack a run, finished or not, into the results zip

--suite quick|cpu|gpu|all|custom  what to run (quick: TinyLlama only)
--probes id,id,...                 run these probes (implies --suite custom)
--budget 45m|2h|1h30m              run the highest-priority probes that fit in this time
--yes                              no questions; --suite defaults to quick
--plan                             show what would run, the time, the downloads and the disk needed,
                                   and stop; writes, downloads and builds nothing
--list                             list the probes and models, with what this machine can run
--models-dir <dir>                 look for model files here first (never written)
--cache-dir <dir>                  where downloaded models go (default ~/.cache/juno/models/kit)
--out <dir>                        where results go (default juno-kit-results/)
--skip-build                       start the kit already built
-h, --help                         every option
```

On Windows, use `juno-kit.bat` in place of `./juno-kit.sh`; the options are the same.

## Common problems

| Message | What to do |
|---|---|
| "juno-kit needs JDK 25 or later" | install JDK 25 or later and put it first on `PATH`, or set `JAVA_HOME` to it |
| "juno-kit needs Maven 3.9 or later" | install Maven 3.9 or later and put `mvn` on `PATH`, or set `MAVEN_HOME` |
| "Not enough free disk to build" | free 2 GiB on the drive that holds the downloaded folder |
| "The downloads need N MiB" | free space, or pass `--cache-dir` on a drive with room for the models |
| a GPU probe "not runnable: CUDA: Juno found no usable NVIDIA GPU" | expected on Windows and macOS; on Linux, check that `nvidia-smi` works and that the CUDA 12.x runtime libraries are on the library path |
| "not runnable: 256-bit integer vectors" | expected on CPUs without 256-bit integer vectors (for example without AVX2); that probe needs them |
| "not runnable: memory: needs N GiB" | the machine has less RAM than that probe needs; the other probes still run |

## Exit codes

- `0`: the probes ran, whatever they found.
- `1`: the build failed (the zip still holds the build log), or the kit stopped on an unexpected error.
- `2`: a prerequisite is missing, there is not enough disk, or an option is wrong.
