# Bernini-R-S2V on LightX2V

Bernini-R-S2V is served directly by LightX2V through the
`wan2.2_s2v_moe` runner. No runtime source overlays are required.

## Architecture

- LightX2V owns the S2V runner, dual-expert routing, sequence parallelism,
  reference conditioning, first-frame stabilization, API server and production
  configuration.
- NFS owns immutable model assets and generated results.
- Bernini is the source model family; its Python inference repository is not a
  runtime dependency.

## Model assets

The default NFS layout is declared in
`configs/wan22/bernini_s2v_assets_nfs.json`. The service launcher validates all
40 streamed blocks, the non-block weights and indexes under each expert
directory, and the shared VAE / T5 / tokenizer / wav2vec assets.

The runner resolves the shared assets relative to `--model_path`, not to
`shared_s2v_model`, so they must be symlinked into both expert directories.
Preflight checks the runtime-resolved paths and prints the `ln -sfn` loop if
they are missing.

## Expert residency and cost

Each expert is ~31 GB and two of them do not fit on a 40 GB card, so
`S2VMultiModelStruct` keeps exactly one resident: it frees the other expert
before building the one the current timestep needs. Timesteps decrease
monotonically within a clip, so there is one high→low crossing per clip, and the
next clip (or the next request) starts on high again.

Steady-state cost is therefore **two full expert loads from NFS per clip**, on
every rank. Budget for that when sizing service latency — it is not included in
the benchmark below, which is a single-clip offline run.

## Identity conditioning

Bernini identity is carried by `context_latents`. These tokens receive:

- a distinct source-id rotary phase;
- no native S2V condition-mask embedding;
- the current diffusion timestep modulation.

The output/audio boundary remains limited to target video tokens. Keeping the
output boundary separate from the timestep boundary is required under Ulysses
sequence parallelism.

## First-frame stabilization

The causal VAE may produce one bright/ghosted transition frame between the
prepended reference latent and generated latents. Set
`s2v_stabilize_first_frame` to replace only that boundary frame with the
immediate successor. The correction is opt-in, applies only to the first clip of
a `drop_first_motion` run (the only place the seam exists), and is enabled in the
Bernini A100 production config.

## Output length

`num_repeat` is null in the production config, so the output follows the audio.
Setting it to an integer caps the number of 80-frame clips and silently shortens
the result — the audio mux uses `-shortest`. The runner logs a warning naming
both durations when that happens.

## Start the 4xA100 service

```bash
bash scripts/server/start_bernini_s2v_a100.sh
```

Build and run the production container:

```bash
docker build -f dockerfiles/Dockerfile_bernini_s2v_a100 -t lightx2v:bernini-s2v-a100 .
bash scripts/server/start_bernini_s2v_a100_docker.sh
```

Default configuration:

```text
configs/wan22/a100/bernini_s2v_moe_4gpu.json
```

## Verified A100 deployment

The repository-built image was verified on four A100 40 GB GPUs with the
production config above. The run completed at 832x448, 16 fps, with audio, and
peaked at 39,679 MiB per GPU.

That run was a **single clip** (`num_repeat` was 1 at the time). Multi-clip runs
are the same code path but have not been measured; see the expert residency cost
above.

NFS assets:

```text
/nfs-data/models/Bernini-R-S2V-lx2v-high
/nfs-data/models/Bernini-R-S2V-lx2v-low
/nfs-data/models/Wan2.2-S2V-14B
```

Verified output and extracted first frames:

```text
/nfs-data/bernini_s2v_out/lightx2v_repo_20260727/bernini_s2v_repo_first_frame_fixed.mp4
/nfs-data/bernini_s2v_out/lightx2v_repo_20260727/first_frames
```

The stabilized first two decoded frames measured YAVG 57.17 and 57.17. The
previous bright transition frame measured YAVG 62.03, so the boundary spike is
removed without changing the remaining generated sequence or audio timing.
