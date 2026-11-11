# Remote test task (for a coding agent on the test PC)

Goal: install liveswap on this Windows PC, run the speed test, and report the
numbers. Do not change the code; if something fails, report it with the output.

## 1. Install

Run in PowerShell (no administrator rights needed, about 6 GB of downloads):

```powershell
$env:LIVESWAP_BRANCH = 'perf/stage-timings'
irm https://raw.githubusercontent.com/VladMarkdown/liveswap/perf/stage-timings/install.ps1 | iex
```

The installer ends with `Press Enter to close`; press Enter. Everything lands
in `C:\liveswap`. If it prints `INSTALL FAILED`, report the last 60 lines of
`C:\liveswap\install-log.txt` and stop.

## 2. Check the GPU path

```powershell
cd C:\liveswap
.\.venv\Scripts\python.exe tools\check_install.py
```

Expected: every model on `CUDAExecutionProvider` and `RESULT: OK`.
If it says the models run on the CPU, also report the output of `nvidia-smi`.

## 3. Speed test

Ask the person at the PC for a face photo of someone who has agreed to this,
and for either a short video file with a face in it or permission to use the
webcam. Then run (use `--source 0` for the webcam):

```powershell
.\.venv\Scripts\python.exe tools\bench.py --photo <photo> --source <video-or-0> --seconds 30 --save out.png 2> bench-stderr.txt | Tee-Object bench.jsonl
```

Run it a second time with `--quality fast`.

## 4. Report

- GPU name, driver version and VRAM (`nvidia-smi`).
- The `check_install.py` output.
- For each run: the final `mean_fps` line and one mid-run JSON line with
  `stages_ms`.
- Any warnings or tracebacks from `bench-stderr.txt`.
- Whether `out.png` shows a swapped face with the `AI-GENERATED FACE` label
  in the bottom-left corner.
