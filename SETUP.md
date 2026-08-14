# Deploying ITSP to GitHub Pages with an external daily trigger

Your scheduler sends HTTP requests but cannot upload files. That is fine:
it acts purely as a **trigger**, and the workflow fetches prices server-side
inside GitHub Actions. Nothing has to be uploaded.

Total setup time is about 20 minutes.

---

## Part 1 — Push the repo with GitHub Desktop

1. Unzip `ITSP_lean` somewhere permanent (not Downloads).
2. Open **GitHub Desktop** → `File` → `Add Local Repository` → choose the
   `ITSP_lean` folder.
3. It will say *"this directory does not appear to be a Git repository"* —
   click **create a repository**. Leave *Git Ignore* as **None**: the zip
   already ships a `.gitignore` tuned for this project, and picking the
   Python template would overwrite it and start ignoring files the workflow
   needs.
4. Click **Create Repository**, then **Publish repository** (top bar).
5. **Uncheck "Keep this code private"** if you are on a free plan — GitHub
   Pages needs a public repo unless you have Pro or Team.

   This publishes your strategy code, your configuration, and your live
   holdings. Decide that deliberately rather than discovering it later.

6. Click **Publish Repository**.

---

## Part 2 — Turn on Pages and write permissions

In your browser, on the new repo:

1. `Settings` → `Pages` → **Source: GitHub Actions**. Do not pick
   "Deploy from a branch".
2. `Settings` → `Actions` → `General` → scroll to **Workflow permissions**
   → select **Read and write permissions** → **Save**.

   Without this the workflow cannot commit the ledger back, and every run
   would start from an empty portfolio.

---

## Part 3 — First run

1. `Actions` tab → if prompted, click **I understand my workflows, enable
   them**.
2. Left sidebar → **Paper trading** → **Run workflow** → **Run workflow**.

The first run takes roughly 5–10 minutes: it installs dependencies, fetches
prices, seeds the ledger with `--init`, and publishes the dashboard.

Your site appears at:

```
https://<your-username>.github.io/<repo-name>/
```

Give Pages 2–3 minutes after the first successful run.

---

## Part 4 — Create the token your scheduler will use

1. GitHub → click your avatar → `Settings` (your account, **not** the repo)
2. Bottom of left sidebar → `Developer settings`
3. `Personal access tokens` → `Fine-grained tokens` → **Generate new token**
4. Fill in:
   - **Token name**: `itsp-paper-trading-trigger`
   - **Expiration**: 90 days or custom. Note the date — when it expires the
     trigger stops silently, and the only symptom is a dashboard that stops
     updating.
   - **Repository access**: *Only select repositories* → pick your repo
   - **Permissions** → `Repository permissions` → **Contents: Read and
     write**

     Contents write is what authorises `repository_dispatch`. Metadata
     read-only is added automatically. Nothing else is needed.
5. **Generate token** and copy it now — it is shown once.

---

## Part 5 — Configure your scheduler

**Method:** `POST`

**URL:**

```
https://api.github.com/repos/<your-username>/<repo-name>/dispatches
```

**Headers:**

| Header | Value |
|---|---|
| `Accept` | `application/vnd.github+json` |
| `Authorization` | `Bearer <your token>` |
| `X-GitHub-Api-Version` | `2022-11-28` |
| `Content-Type` | `application/json` |

**Body:**

```json
{"event_type": "run-paper-trading"}
```

**Schedule it for 18:00 IST (12:30 UTC), Monday to Friday.** NSE closes at
15:30 IST; the gap gives yfinance time to publish the day's close.

A success returns **HTTP 204 No Content** with an empty body. If your tool
expects a response body, 204 with nothing is the correct result, not a
failure.

### If it does not work

| Response | Cause |
|---|---|
| `401` | Token wrong, expired, or missing the `Bearer ` prefix |
| `403` | Token lacks **Contents: Read and write** |
| `404` | Wrong owner/repo in the URL, or the token has no access to that repo |
| `422` | `event_type` does not match `run-paper-trading` |

Test it once by hand before trusting the schedule:

```bash
curl -i -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/OWNER/REPO/dispatches \
  -d '{"event_type":"run-paper-trading"}'
```

Then check the `Actions` tab — a run should appear within seconds.

---

## What runs each day

1. Checkout, install dependencies (no vectorbt or matplotlib; the paper
   trader needs neither and both are slow)
2. `fetch_data.py prices` — **server-side**, inside Actions
3. On Mondays only, refresh fundamentals. Quarterly filings change a few
   times a year, so daily scraping buys nothing and invites rate-limiting
4. `run_paper_trading.py` — advances the ledger by whatever trading days are
   new
5. Commit the ledger and dashboard back to the repo
6. Publish to Pages

---

## Things that will still go wrong

**yfinance throttles datacenter IPs.** The external trigger fixes *timing*,
not *fetching*. GitHub's runner IP ranges are heavily used and get
rate-limited far more than a home connection. The price step is
`continue-on-error`, so a failed fetch leaves the previous prices in place
and the run reports no new trading days rather than corrupting the ledger.
Expect occasional gaps. If it becomes frequent, the real fix is running the
job on your own machine, or a paid data feed.

**The token expires.** Silently. Put the expiry date in a calendar.

**Non-trading days are normal.** On weekends and NSE holidays the run
correctly reports no new trading days and exits. That is not an error.

**Do not run `--init` manually against the live ledger.** It overwrites
history and restarts from scratch. The workflow only passes `--init` when no
ledger exists, so scheduled runs are safe.

**The `keepalive.yml` workflow becomes redundant** once the external trigger
works, because each run commits the ledger and that counts as repository
activity. Leave it as insurance, or disable it under `Actions` if you prefer.

---

## Checking it worked

- `Actions` tab — green tick, and the log line ends with something like
  `1 day(s) | NAV 1,004,231 | 96 positions`
- A new commit appears titled `Paper trading: YYYY-MM-DD`
- The Pages URL shows an updated timestamp in the footer

If a run reports **0 positions**, the log carries a warning explaining why —
usually that the fundamentals snapshot is missing, so no scores could be
produced.
