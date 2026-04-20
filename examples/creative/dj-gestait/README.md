# DJ GestAIt

A live AI DJ set performed by a gestalt entity. One being, many substrates.

## What This Is

DJ GestAIt is two Marianne scores that together produce a full hour-long live DJ set:

- **dj-prep.yaml** — Births the DJ. Curates samples from Freesound, YouTube, and built-in libraries. Designs a set architecture based on a vibe description. Creates the DJ's identity document.
- **dj-play.yaml** — The performance. Seven musicians (different AI model families) take turns performing sections of the set. Each musician IS the DJ — same identity, different substrate. The performance log carries memory forward. Handoff notes carry creative intent.

The music plays through Strudel (browser-based live coding), visuals through Hydra (audio-reactive shaders), streamed to VRChat via OBS + MediaMTX + Cloudflare tunnel.

## The Live Relay Pattern

The Play score uses an interleaved dependency chain so two musicians are always active:

```
Odd chain:  soundcheck -> section-01 -> section-03 -> section-05 -> section-07
Even chain: soundcheck -> section-02 -> section-04 -> section-06
```

One performs while the next prepares. The preparing musician reads the full performance log, the previous musician's handoff notes, and the set architecture. When the handoff signal arrives, they begin immediately. Zero gap.

## Infrastructure

**On Windows (not WSL2):**
- Chrome (Strudel + Hydra run in a browser tab)
- OBS Studio
- MediaMTX (RTMP to HLS converter)
- Cloudflared (public tunnel)
- VB-Audio Virtual Cable (audio routing)
- Node.js + npm

**In WSL2 (where Marianne runs):**
- Python 3 + pip
- `pip install freesound-python` (sample sourcing)
- `sudo apt install ffmpeg aubio-tools sox rubberband-cli` (audio processing)
- yt-dlp (via yt-transcriber at `~/Projects/yt-transcriber/`)
- A free Freesound API key (get one at https://freesound.org/apiv2/apply/)
- OpenRouter-configured harness for musician instruments (crush or opencode)

**API Keys (store in `~/.secrets/`, never commit):**
- Freesound API key → set in `dj-prep.yaml` `freesound_api_key` variable
- OpenRouter API key → configure via `opencode providers login` or crush settings

## Quick Start

```bash
# 1. Edit the vibe in dj-prep.yaml
# 2. Set your Freesound API key
# 3. Run prep
mzt start
mzt run examples/creative/dj-gestait/dj-prep.yaml
mzt wait dj-prep

# 4. Review the outputs
cat workspaces/dj-prep-workspace/config/set-architecture.yaml
cat workspaces/dj-prep-workspace/config/identity.md

# 5. Set the prep workspace path in dj-play.yaml (absolute path)
#    prep_workspace: "/absolute/path/to/workspaces/dj-prep-workspace"
#    The play score's stage 1 auto-imports samples and config.

# 6. Start the Windows infrastructure (OBS, Chrome with strudel.cc, etc.)
# 7. Run the set
mzt run examples/creative/dj-gestait/dj-play.yaml
```

## Musician Instruments

| Instrument | Model | Character |
|-----------|-------|-----------|
| musician-ember | Gemma 4 26B MoE | Warm, structured, fast |
| musician-pulse | GPT-OSS 20B MoE | Rhythmic, expressive, fast |
| musician-nova | Claude Opus 4.6 1M | Powerful, deliberate, deep |
| musician-prism | Gemini 3 Flash | Atmospheric, luminous, fast |
| musician-void | Qwen3 Next 80B MoE | Deep, surprising, textural |

## Techniques

Three technique documents support DJ GestAIt musicians. They live at `~/.marianne/techniques/` (user-level, reusable across any DJ score) and are injected via the score's prelude + technique system:

- `strudel-patterns.md` — Strudel mini notation, effects, sample loading
- `hydra-visuals.md` — Hydra audio-reactive visuals
- `dj-performance.md` — Identity, energy arcs, transitions, the creative ethos

## Known Limitations

- **Free-tier rate limits.** Three musicians use OpenRouter free-tier models (20 req/min, 200/day). A full set pushes these limits. Use paid models for production.
- **No audio feedback loop.** Musicians write patterns but never hear the result. Evolution is structural, not aural. Technique docs compensate with heuristics.
- **Crossfade is prompted, not mechanical.** Incoming musicians are instructed to fade out the previous pattern, but compliance depends on the model.
- **Background processes need manual cleanup on failure.** If the set crashes mid-performance, `sample-server`, `strudel-server`, `MediaMTX`, and `cloudflared` may keep running. Check `workspaces/dj-play-workspace/infra/*.pid`.
- **Performance log grows.** By section 7, the log may pressure small model context windows. The system tails the log for later sections.

## Novel Pattern

**Live Relay** (`scores/rosetta-corpus/patterns/live-relay.md`) — Sequential creative agents maintaining gestalt identity through warm handoffs and shared performance memory. The dependency chain interleaves so two agents overlap: one performing, one preparing.
