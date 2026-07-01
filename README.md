# Khalaifat Catalogues — tablet site (sample)

## What this is
A single-page tablet site (`index.html`). Brands sit on a "shelf" — tap one to
open a drawer of its catalogues, tap a catalogue to read it full-screen, page
by page, rendered from the real PDF (not the browser's native PDF viewer).

## Run it
Do **not** just double-click `index.html` — opening it as a `file://` URL
makes browsers block the PDF fetches (CORS), so every catalogue will fail
to load. Instead, serve the folder over HTTP:

- **Mac/Linux:** run `./start.sh`, then open `http://localhost:8080`
- **Windows:** run `start.bat`, then open `http://localhost:8080`

(Or upload the whole folder to any web host / the Odoo Website custom code
area, which serves it over http(s) automatically.)

Needs an internet connection once, to load:
- Google Fonts (Fraunces + Inter)
- pdf.js from cdnjs.cloudflare.com (used to render the PDF pages on canvas)

## AI chat assistant
A small "Ask AI" button in the footer opens a chat panel that answers
questions about the catalogues (see `ai/README`-style scripts in `ai/`). It
talks to the local RAG server on port 8000 — start it separately with
`ai/3_start_server.sh` (or `.bat`). The site (port 8080) and the AI server
(port 8000) run side by side on the same kiosk PC. If the AI server isn't
running, the chat button still shows but says the assistant is offline.

If this needs to run fully offline (e.g. an in-store kiosk with no Wi-Fi),
download pdf.js + the two fonts and reference them locally instead of the
CDN links in `index.html`.

## Folder structure
```
khalaifat-catalogues/
  index.html
  pdfs/
    beko/
      beko-built-in-2025.pdf
      beko-freestanding-2025.pdf
    <next-brand>/
      <catalogue>.pdf
```

## Adding a new brand
1. Put its PDF(s) in a new folder: `pdfs/<brand-id>/`.
2. Open `index.html`, find the `BRANDS` array near the top of the `<script>`
   block, and add an entry — there's a commented example (Bosch) right there
   showing the exact shape:
   ```js
   {
     id: 'bosch',
     name: 'BOSCH',
     tagline: 'Engineered in Germany',
     catalogues: [
       { title: 'Built-In Range', year: '2025', file: 'pdfs/bosch/bosch-built-in-2025.pdf' }
     ]
   }
   ```
3. Save. No other code changes needed — the shelf, drawer and viewer are all
   generated from this array. The empty "Add brand" tiles just shrink by one
   each time you add a real brand (`EMPTY_SLOTS` controls how many show).

## Colors used
| Token  | Hex      | Use                                  |
|--------|----------|---------------------------------------|
| navy   | #263B59  | page background, plate gradient base |
| steel  | #45738E  | plate gradient, buttons, accents     |
| pale   | #D3E5E8  | hero highlight text, light tints     |
| brass  | #B8935A  | added accent — dividers, hover edges, the "open" arrows |
| paper  | #F7F5F1  | drawer + catalogue card background   |
| ink    | #1B2A3D  | text on light surfaces               |

Brass and paper aren't in your original 3 colors — added for contrast and a
"showroom plaque" feel. Both are easy to delete/swap in the `:root` block at
the top of `index.html` if you'd rather stay strictly on-brand.

## Known limitations (it's a basic sample)
- No thumbnail/cover previews on the catalogue cards — title + year only.
- No pinch-to-zoom inside the page viewer (swipe/arrow-buttons to turn pages).
- Brand logos are typeset text, not real logo files — drop in `<img>` tags
  in the `plate-name` div if you have brand logo assets.
- Large catalogue PDFs (the Freestanding one is ~20MB) take a moment to load
  on first open — pdf.js streams it, so the first page still appears quickly.
