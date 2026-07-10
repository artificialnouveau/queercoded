# Sharing codes across people (future work)

Status: **deferred.** v1 stores codes in the browser's `localStorage` (one
device, one browser) with JSON **export/import** for manual sharing. This note
captures how we would let multiple people see a shared set of codes later.

Key point: we only ever share **pose coordinates** (the `seq` arrays), never
webcam recordings. A shared code is a few kilobytes of numbers.

## What we already have (no backend)

- **Export / Import JSON** — one person exports `queercoded-codes.json`, another
  imports it. Good for workshops and handoffs. Already implemented.

## Options for real sharing (pick when we get there)

### A. Shareable link (still no backend)
Encode a code (or a small set) as base64 in a URL. Opening the link offers to
import it. Zero infrastructure; great for sending one code to a friend. Limited
by URL length (a handful of codes), and no central library.

### B. Static shared file in the repo
Commit a `codes/*.json` library to the repo; the app fetches and offers to load
them. Read-only, curated, versioned by git. No user accounts. Good for a
"starter pack" of official codes.

### C. Hosted database (true multi-user, read/write)
A real backend (e.g. **Supabase** or Firebase) so anyone can publish and browse
codes. This is the only option that supports a live, growing shared library.

Suggested minimal schema (Supabase/Postgres):

```
codes (
  id uuid primary key default gen_random_uuid(),
  word text not null,
  seq jsonb not null,          -- the 20x66 coordinate array
  dur_ms int,
  author_id uuid references auth.users,
  is_public boolean default false,
  created_at timestamptz default now()
)
```

If we go with C, apply the project security checklist before shipping:

- **No secrets in code:** Supabase URL + anon key via env; service key never in
  the client.
- **Row Level Security ON:** public read only where `is_public = true`; users
  can insert/update/delete only their own rows (`author_id = auth.uid()`).
- **Input sanitization:** cap `word` length and character set; validate that
  `seq` is a well-formed numeric array of the expected shape before insert.
- **Rate limiting:** limit publish/write calls per IP/user.
- **CORS:** allow only our own origin.
- **Security headers:** X-Frame-Options, X-Content-Type-Options, CSP,
  Referrer-Policy.

## Recommendation

Ship v1 on export/import (done). Add **A (shareable link)** as a cheap next step
for person-to-person sharing. Move to **C (Supabase)** only when we actually
want a public, browsable library with contributions.
