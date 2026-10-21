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
