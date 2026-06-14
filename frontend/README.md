# VoxClinical

Research prototype for analyzing speech and text samples for linguistic markers associated with Parkinson's and ALS (Amyotrophic Lateral Sclerosis).

**Not a medical device** — for screening research only. Do not use for clinical diagnosis.

## Local development

```bash
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## Build

```bash
npm run build
```

Output is written to `dist/`.

## Deploy to Netlify

### Option A: Git-based deploy (recommended)

1. Push this repo to GitHub.
2. In [Netlify](https://app.netlify.com), click **Add new site → Import an existing project**.
3. Connect your GitHub repo.
4. Netlify reads settings from `netlify.toml` automatically:
   - **Build command:** `npm run build`
   - **Publish directory:** `dist`
   - **Node version:** 20
5. Click **Deploy site**.

Every push to your default branch triggers a new deploy.

### Option B: Netlify CLI

```bash
npm install -g netlify-cli
netlify login
netlify init
netlify deploy --prod
```

## Stack

- React 19 + TypeScript
- Vite
- Tailwind CSS v4
- Client-side analysis (no backend required for the draft)
- localStorage + IndexedDB for sample persistence
