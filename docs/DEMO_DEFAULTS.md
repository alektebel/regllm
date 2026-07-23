# Default dictionary, rules & cases for the DQC demo

The demo opens preloaded with three files. You can keep the bundled ones,
replace them locally, or host your own anywhere (S3, a CDN, any web server)
and point the app at them — no rebuild needed for the hosted route.

| Default | File | Format |
|---|---|---|
| Dictionary | `diccionario_demo.xlsx` | `.xlsx` — row 1 = headers, one field per row. Only a field-name column is required (`Field`/`Campo`/`Nombre`/…); `Type`, `Description`, `Null`, `Formula`, `Reg ref` are optional. |
| Cases | `casos_demo.xlsx` | `.xlsx` — row 1 = field names (matching the dictionary), one record per row. Optional `DQC_ID` column = ground-truth label(s), `;`-separated. |
| Rules | `reglas_demo.txt` | `.txt` — one rule per line, optionally prefixed `DQC_ID: <text>`. |

The app resolves each default in this order: **URL query param → `assets/demo/sources.json` → the bundled file.**

---

## Option A — replace the bundled files (local, simplest)

Drop your files in, keeping the exact names, then restart:

```bash
cp my_dictionary.xlsx DQC/app/src/assets/demo/diccionario_demo.xlsx
cp my_cases.xlsx      DQC/app/src/assets/demo/casos_demo.xlsx
cp my_rules.txt       DQC/app/src/assets/demo/reglas_demo.txt

./scripts/share_demo.sh          # rebuilds; opens preloaded with your files
```

---

## Option B — host them yourself (e.g. S3) and point the app at them

Upload once, then either pass query params or set `sources.json`. Nothing to
rebuild.

### 1. Upload the files

Set your bucket and region, then copy the three files up:

```bash
BUCKET=my-demo-bucket
REGION=eu-west-1

aws s3 cp my_dictionary.xlsx s3://$BUCKET/dqc-demo/diccionario.xlsx --region $REGION
aws s3 cp my_cases.xlsx      s3://$BUCKET/dqc-demo/casos.xlsx       --region $REGION
aws s3 cp my_rules.txt       s3://$BUCKET/dqc-demo/reglas.txt       --region $REGION
```

> Uploading objects (`s3:PutObject`) and setting bucket CORS (`s3:PutBucketCors`)
> are different permissions from *creating* a bucket — you may have them even if
> bucket/VPC creation is blocked. If you have no bucket, any static host works
> (a web server, a CDN, GitHub raw) — the URLs just point there instead.

### 2. Make them reachable by the browser

The browser fetches these directly, so each object must be readable **and** the
host must send CORS headers for the app's origin.

**Readable** — either make the objects public-read:

```bash
aws s3 cp my_dictionary.xlsx s3://$BUCKET/dqc-demo/diccionario.xlsx --acl public-read --region $REGION
# …repeat for the other two…
```

…**or** generate presigned URLs (no public access needed; each URL expires):

```bash
aws s3 presign s3://$BUCKET/dqc-demo/diccionario.xlsx --expires-in 604800 --region $REGION
aws s3 presign s3://$BUCKET/dqc-demo/casos.xlsx       --expires-in 604800 --region $REGION
aws s3 presign s3://$BUCKET/dqc-demo/reglas.txt       --expires-in 604800 --region $REGION
```

**CORS** — allow the app's origin(s). Save this as `cors.json`:

```json
{
  "CORSRules": [
    {
      "AllowedOrigins": ["http://localhost:4200", "http://YOUR-LAN-IP:4200", "https://YOUR-TUNNEL.trycloudflare.com"],
      "AllowedMethods": ["GET"],
      "AllowedHeaders": ["*"],
      "MaxAgeSeconds": 3000
    }
  ]
}
```

Apply it to the bucket:

```bash
aws s3api put-bucket-cors --bucket $BUCKET --cors-configuration file://cors.json --region $REGION
```

(Add your LAN IP and tunnel domain to `AllowedOrigins` as needed. Without CORS
the browser blocks the fetch and the app silently falls back to the bundled file.)

### 3. Point the app at them

**Query params** (fastest, per-session, no file edits) — open the app like:

```
http://localhost:4200/?dict=https://my-demo-bucket.s3.eu-west-1.amazonaws.com/dqc-demo/diccionario.xlsx&cases=https://my-demo-bucket.s3.eu-west-1.amazonaws.com/dqc-demo/casos.xlsx&rules=https://my-demo-bucket.s3.eu-west-1.amazonaws.com/dqc-demo/reglas.txt
```

**`sources.json`** (persistent default) — edit `DQC/app/src/assets/demo/sources.json`:

```json
{
  "dictionary": "https://my-demo-bucket.s3.eu-west-1.amazonaws.com/dqc-demo/diccionario.xlsx",
  "cases":      "https://my-demo-bucket.s3.eu-west-1.amazonaws.com/dqc-demo/casos.xlsx",
  "rules":      "https://my-demo-bucket.s3.eu-west-1.amazonaws.com/dqc-demo/reglas.txt"
}
```

Restart the app; it now opens preloaded from your hosted files. Leave any value
`null` to keep using the bundled file for that slot.
