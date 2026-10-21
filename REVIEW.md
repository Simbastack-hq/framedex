# Review ledger — fdx share v0 (working file, not committed)

## Round 1 — plan review (Codex, gpt-6.1-sol high) — 2026-10-21
1. rsync filters are not a deletion boundary — FOLD IN: sync site straight into one validated <root>/<prefix>/<slug>/, no filters, no traversal/symlink escapes; sibling-sentinel e2e test with local rsync.
2. Publisher bypasses encrypted output — FOLD IN: one explicitly enumerated publish artifact (objects list), uploaded verbatim by every target; preview never in it.
3. Re-publish with fresh salt + immutable names breaks/caches — FOLD IN: encrypted names = hash of ciphertext; assets first, HTML last, stale deleted after; doc that cached plaintext can't be revoked.
4. Protection can silently disappear — FOLD IN: persist protected per link; publish fails closed without passcode unless --no-passcode.
5. Privacy promise overstated — FOLD IN (part): locked page drops description/OG; cover blurred server-side; document title/slug/sizes/host-JS limits. DEFER: generated strong passphrase.
6. Editor server boundaries — FOLD IN: token path prefix, allowlisted rendered assets only, exact Host:port, X-FDX-Token on POST, no CORS.
7. Browser vs Python crypto — FOLD IN: canonical UTF-8 AAD (gallery-relative path; "fdx:manifest"), node WebCrypto check locally, image failure != wrong passcode, secure-context check, revoke lightbox blobs.
8. Editor save erases pool — FOLD IN: round-trip pool + hidden items, atomic write. DEFER: burst alternates in pool.
9. Story pass destructive — FOLD IN (cut): drop subject-run + closer-move rules (near-dup covers near-identical runs). KEEP: hero = strongest wide frame, separate + editable. NEW (non-destructive): last section reserves one slot for its best late frame.
10. Trip name collisions — FOLD IN: name includes start date.
- Scope: "Caddy auth simpler than encryption" — REJECT: NJ explicitly chose ciphertext-only hosting (dumb server, CDN-safe, portable). Freeze S3 + trips work: AGREE.

## Round 1b — plan design read (GPT-6 Astra) — 2026-10-21
- Neutral palette (#fff/#1d1d1f/#6e6e73; #000/#f5f5f7/#a1a1a6) — FOLD IN.
- Compact opening, first photo in first viewport — FOLD IN (judge title size on screen; NJ asked for a big quiet title).
- Strongest frame opens regardless of orientation — FOLD IN (cover = best score; uncropped).
- Pairs: equal height, 24px gap, cap min(880px, 80svh); 390px dissolves pairs, full-bleed — FOLD IN.
- Quiet 15px section headings, 64/24 spacing; 180-250ms fades; no scroll reveals — FOLD IN.
- Lightbox: visible but quiet close + desktop arrows (44px), tap closes, swipe never closes — FOLD IN.
- Editor: inline toolbar success (no sheet), hover/focus controls, handle on touch, keep scroll on edits — FOLD IN.
- Unlock: frosted white (light) / black (dark) overlay, Show toggle, spinner, reserved error line — FOLD IN.
- "No shake on wrong passcode" — REJECT: Apple's own password fields shake; disabled under reduced motion.
- Cut closer choreography — REJECT (narrowed): no reordering; only a reserved pick slot when the cap bites (NJ asked for a good closer).

## Round 2 — design critique of real screenshots (GPT-6 Astra) — 2026-10-21
1. Title too big / landing-page feel — FOLD IN: 48/52 desktop, 72 top, 12 to subtitle, 40 to hero; phone 28/34, 40 top, 28 below.
2. Centered singles feel unedited — FOLD IN: hero 1200px, other singles 960px, pairs 24px gutter, 64px rows; phone unchanged.
3. Headings interrupt — FOLD IN: 13/18 weight 500, no weekday ("9 November").
4. Passcode controls look disabled — FOLD IN: desaturate + 25% black overlay, 360/56/16 field, solid white 44px button, copy "Enter the passcode to view."; no mobile autofocus (already done).
5. Editor shifts layout — FOLD IN: 56px fixed toolbar over the top whitespace; 5 s Undo after hide; hint gone after first edit; Cover badge on hover; title underline affordance.
- Cut: instructions after first edit, permanent Cover badge, weekday names — FOLD IN.
- Memorable ending — FOLD IN (narrowed): the closer slot prefers a wide frame from the last quarter; 96px before the footer. No reordering (Codex round 1).

## Round 3 — code review (Codex, gpt-6.1-sol high) — 2026-10-21
1. Failed hide save still allows Publish — FOLD IN: failed save blocks Publish; "Photo hidden" only after a successful save.
2. Protection lost after partial failure — FOLD IN: mark protected before remote writes; clear only after an open_ok publish succeeds.
3. Remote symlink escapes deletion boundary — FOLD IN: remote rsync-path guard compares `pwd -P` of dest to resolved root + key; local check exact.
4. Cleanup can delete originals via --out overlap — FOLD IN: refuse work dirs holding selected originals; cleanup only our own render/temp names.
5. --keep-gps publishes GPS — REJECT: explicit opt-in the brief asked for; docs state it publishes coordinates.
6. rsync may skip a changed page (same size/second) — FOLD IN: --checksum on the final pass.
7. Unreadable publish.json silently resets — FOLD IN: only FileNotFoundError is a fresh start; other errors fail loud.
8. CI skips crypto/pixel tests — FOLD IN: share-extra CI job.
- Delete: Site.files, Rendered.bytes, Editor.folder, Published.sent/removed, editor JSON sections/protected — FOLD IN. --keep-gps — REJECT (see 5).

## Round 4 — final design pass (GPT-6 Astra) — 2026-10-21 — verdict: SHIP
1. Passcode controls still muted — FOLD IN: white 92% button, #303028 arrow, 75% white placeholder, disabled at 0.8.
2. Explicit 1200/960 widths — NO CHANGE: implemented (hero 1200, singles 960, pairs to 1200); the screenshot set has no portrait pairs to show it.
3. Mobile editor bottom bar — REJECT: the editor binds 127.0.0.1, it runs on the machine running fdx; a phone can't reach it.
4. Mobile discovery cue — REJECT: same reason.
