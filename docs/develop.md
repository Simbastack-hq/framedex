# fdx-develop: a vision model edits your RAWs (experimental)

`fdx-develop` treats the vision model as the editor and an external RAW engine
as the darkroom. You get edited copies of a folder of RAWs, ready for a web
gallery, without touching a slider. It is opt-in, experimental, and never
modifies an original.

## How it works

For each RAW:

1. **Neutral render.** RapidRAW renders the RAW with neutral settings. If a
   Lightroom `.xmp` sidecar carries a crop, it is applied, so the model edits
   your final framing. framedex never invents a crop.
2. **Calibration sheet.** The same photo, rendered with a handful of
   single-slider probes (exposure +1, shadows +60, highlights -60,
   temperature +30, vibrance +40) on one labelled sheet. The model has never
   used this engine; this shows it what the numbers do to *this* image.
3. **Look and choose.** The model gets the neutral render, the sheet, measured
   stats (mean/median brightness, % clipped, % near-black, histogram), any
   context framedex already knows (a `.description.md` sidecar's place,
   lighting, scene; else EXIF), and, with `--examples`, your own before/after
   pairs. It returns one JSON object of slider values.
4. **Render and look again.** The engine renders those values. The next round
   shows the model its render next to everything else and asks it to critique
   and revise. It may stop from round 3 on; otherwise it runs `--rounds`
   (default 3, max 4).
5. **Write.** `<out>/<name>.jpg` (long edge `--size`, default 2048), a
   plain-text `<out>/<name>.develop.json`, and a RapidRAW `.rrdata` next to the
   RAW if none exists.

Rounds are stateless: each call carries everything it needs, so the same loop
runs on every backend (`api`, `local`, `cli`).

## Install

1. The `[images]` extra (Pillow): `uv pip install -e '.[images]'`.
2. [RapidRAW](https://github.com/CyberTimon/RapidRAW) 1.6 or newer (needs its
   headless `export` CLI). framedex finds it via `--engine PATH` (a binary or a
   `.app`), then `$FDX_RAPIDRAW`, then `rapidraw` on `PATH`, then
   `/Applications/RapidRAW.app`.

## Usage

```bash
fdx-develop ~/shoots/mara                     # every RAW under the folder
fdx-develop ~/shoots/mara --dry-run           # list + cost estimate, no calls, no renders
fdx-develop ~/shoots/mara --examples ~/shoots/mara/hero-edits
fdx-develop ~/shoots/mara --backend local --local-model gemma-4-27b
fdx-develop ~/shoots/mara --rounds 4 --size 3000 --out ~/web/mara
fdx-develop ~/shoots/mara --files-from picks.txt   # only the picked photos
```

| Flag | Meaning |
|---|---|
| `--out DIR` | Output folder (default `<root>/developed`, subfolders mirrored) |
| `--engine PATH` | RapidRAW binary or `.app` |
| `--backend api\|local\|cli` | Anthropic API (default), an OpenAI-compatible local server, or the Claude Code CLI |
| `--model ID` | Model for api/cli (default `claude-opus-5-5`) |
| `--local-base-url`, `--local-model` | Local server (default `http://localhost:1234/v1`, LM Studio) |
| `--rounds N` | Max look-and-adjust rounds, 1-4 (default 3) |
| `--examples DIR` | Up to 3 pairs of `<name>.<raw>` + `<name>.jpg` (your finished export of that RAW) |
| `--size PX` | Long edge of the edited copies (default 2048) |
| `--files-from FILE` | Develop only the RAWs listed in FILE: one path per line, relative to the root (or absolute inside it), blank lines and `#` comments ignored. A path outside the root, missing, or not a RAW is an error naming the line. Output names are planned over the whole folder, so a photo gets the same name with or without the list |
| `--max-files N` | Cap a run |
| `--force` | Redo photos framedex already developed |
| `--no-rrdata` | Don't write the RapidRAW sidecar |
| `--no-calibration` | Skip the calibration sheet |

### Style examples

The strongest lever on quality is showing the model how *you* edit. Put 2-3
RAWs you've already edited, each next to your exported JPEG with the same
name, in a folder and pass it with `--examples`. Pick hero shots from the same
shoot or similar light. framedex renders each RAW neutral (cropped like your
export, if the export carries Lightroom's crop) and shows the model
before/after: "match this photographer". It is told to match brightness,
warmth, saturation and contrast, not the crop, and not to copy a black and
white conversion unless the photo calls for it.

## Outputs and sidecars

- **Originals are never modified, moved, or renamed.**
- `<out>/<name>.jpg`: the edited copy, sRGB, quality 90. If two RAWs in one
  folder would get the same output name (`A.NEF` + `A.ARW`, or `A.NEF` +
  `A.NEF.DNG`), those outputs use the full file name (`A.NEF.jpg`,
  `A.ARW.jpg`, `A.NEF.DNG.jpg`).
- `<out>/<name>.develop.json`: plain text. Status, source fingerprint (name,
  size, mtime), the output's SHA-256, engine and version, backend and model,
  every round's values, stats and cost, the final engine adjustments, the
  crop. A `pending` record carrying the new JPEG's digest is written before
  the JPEG is published and finalized after, so an interrupted run recovers;
  a failed redo keeps the digest of the earlier output it left in place. A re-run skips a photo only
  if the record says `done`, the RAW is unchanged and the JPEG is byte-for-byte
  what framedex wrote. `failed` photos (no usable render) and `partial` ones (a
  later round failed; the best earlier render ships) are retried next run.
- `<raw>.rrdata`: RapidRAW's own sidecar with the final adjustments (and the
  crop, if it has no rotation), so you can open the RAW in RapidRAW and keep
  editing from the model's values. Written only if no `.rrdata` exists; your
  own RapidRAW edits are never overwritten. `--no-rrdata` skips it.
- framedex never overwrites an output JPEG it can't prove it wrote (no
  record, a record for another RAW, or a digest mismatch because you replaced
  the file). Those photos are skipped with a message.

## Cost

Fixed per photo: at most `--rounds` model calls, each with at most two 1600 px
images, the calibration sheet, and up to three 1024 px example pairs.
Measured on 10 real RAWs with Claude Opus 5.5 (effort `high`, adaptive
thinking), two example pairs, the calibration sheet and the default 3 rounds:
$0.19-0.28 per photo, $0.23 on average, about 2.5 minutes each. Four rounds
cost about $0.30. The
run prints an estimate before it starts and the actual cost at the end
(marked `+ unknown` for the CLI backend, which doesn't report usage, and
for any call that failed mid-flight, which may still have been billed). `--backend local`
is $0. This command is the one place framedex spends more than one call per
file; the cap is the round count.

## What leaves the machine

With `--backend api` or `cli`: downscaled renders of each photo (and of your
example pairs) plus the context text go to the model provider every round.
With `--backend local`: nothing. RapidRAW runs locally.

## Licensing

RapidRAW is AGPL-3.0. framedex (MIT) does not include, link, or modify any
RapidRAW code: it runs the user's independently installed, unmodified binary
as a separate program through its documented command line, the same way it
runs exiftool and ffmpeg, and reads/writes RapidRAW's documented JSON. If you
redistribute RapidRAW alongside framedex, or modify it, the AGPL's terms apply
to RapidRAW.

## Known limitations

- Masks are radial and graduated only. RapidRAW's AI subject/sky masks need
  extra model downloads and aren't wired in yet.
- No crop or straighten decisions; Lightroom crops are honored, nothing else.
  A rotated (straightened) Lightroom crop is applied to the outputs but not
  carried into the `.rrdata`.
- RapidRAW's lens database doesn't cover every lens, so distortion and
  vignetting correction can differ from Lightroom's.
- Quality depends heavily on the model. Local models are supported but
  untested for quality.
- One engine process at a time; a 45 MP RAW renders in ~4 s on Apple silicon,
  so a photo takes about a minute end to end (renders + model rounds).
