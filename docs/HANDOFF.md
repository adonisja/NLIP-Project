# Taking over Angel Filter

A checklist for the new team, in order. Each step links to the details and
ends with a way to check it worked. Expect an afternoon to get through steps
1–4, and step 6 when you're ready to deploy.

**Nothing from the original team's accounts carries over.** You create your
own repository, database, deployment and API keys. The original team's
Supabase project, Render service and keys stay theirs.

Questions: your mentor, on Discord `akkeemt` or at <tyrellakkeem@gmail.com>
(GitHub [@adonisja](https://github.com/adonisja)).

---

## 1. Your own repository

- [ ] Create a repository under your account or organisation and push this
      code to it (for example, GitHub's **Import repository**, or clone and
      push to a new remote).
- [ ] Add your teammates as collaborators.
- [ ] Protect `main`: **Settings → Rules → Rulesets** (or **Branches**):
      require a pull request, one approval, and the `test` and
      `deploy-install` checks. See [CONTRIBUTING.md](../CONTRIBUTING.md).

**Check:** open a small PR. CI should run both jobs and pass.

## 2. Run it locally

- [ ] Follow the [README quickstart](../README.md#quickstart). It needs no
      paid keys: Ollama is the only provider.
- [ ] `poetry run pytest` passes.

**Check:** `curl http://localhost:8005/health` shows `"mode": "nlip"`, and a
search in the UI at <http://localhost:8005> returns results.

## 3. API keys

Get only what you'll use; [SETUP.md](SETUP.md#providers) lists each one.
Gemini has a free tier and is the easiest second provider.

- [ ] Put keys in `.env` (never commit it; it's in `.gitignore`).
- [ ] **Set spending alerts** on every paid account, especially Google Cloud
      (Places is billed per call) and OpenAI. The app's daily cap limits
      searches, not your monthly bill.

**Check:** each provider you configured appears in `/health`'s `providers`
list.

## 4. Database (Supabase)

Without a database, state lives in memory and resets on every restart,
including the daily spending cap. Follow
[SETUP.md → Shared state](SETUP.md#shared-state-supabase).

- [ ] Create a free Supabase project with the Data API off.
- [ ] Put its **Session pooler** URL in `.env` and set the password with
      `poetry run python scripts/check_database.py --set-password`.
- [ ] Restart the server.
- [ ] Optional: connect Supabase to your repository (**Project Settings →
      Integrations → GitHub**) so new files in `supabase/migrations/` are
      applied when PRs merge to `main`.
- [ ] Add `ANGEL_DATABASE_URL` as a GitHub Actions secret, then run **Actions
      → Keep the database awake → Run workflow** once. Free projects pause
      after about a week without activity; this workflow prevents that.

**Check:** `/health` shows `"state": {"backend": "postgres", "reachable":
true}`, and after a search, Supabase's Table Editor (schema `angel_state`)
shows rows in `kv`. The keep-alive run's log says `heartbeat written`.

## 5. Read before you change things

- [ ] [DESIGN.md](DESIGN.md): how ranking works and why. Several choices look
      odd until you know what they prevent.
- [ ] [KNOWN_ISSUES.md](KNOWN_ISSUES.md): the starter tasks. Task 1, designing
      login and access control, matters before any public deployment.
- [ ] [CONTRIBUTING.md](../CONTRIBUTING.md): workflow, tests, adding a
      provider, changing the database schema.

## 6. Deploy (when you're ready)

Follow [SETUP.md → Deploying your own copy](SETUP.md#deploying-your-own-copy).
**There is no login yet**, so anyone with the URL can search on your keys;
read step 1 there about cost first.

- [ ] Create the Render service from `render.yaml`, in the region nearest
      your Supabase project.
- [ ] Set the secrets (table below) in Render's **Environment** settings.

**Check:** the live `/health` shows `"mode": "nlip"` and
`"state": {"backend": "postgres", "reachable": true}`. Then do the
two-network rate-limit check in SETUP.md's deploy steps.

---

## Where each setting goes

| Setting | `.env` (local) | Render | GitHub Actions secret |
|---|---|---|---|
| Provider keys (`OPENAI_API_KEY`, `GEMINI_API_KEY`, …) | yes | yes | no |
| `OLLAMA_MODEL` | yes | **no** (Render can't run Ollama) | no |
| `ANGEL_DATABASE_URL` | yes | yes | yes (for the keep-alive) |
| `ANGEL_TRUSTED_PROXY_HOPS` | leave unset (0) | `1`, already in `render.yaml` | no |
| Rate limit and daily cap | optional | already in `render.yaml` | no |

If `/health` doesn't show what you expect, [SETUP.md →
Troubleshooting](SETUP.md#troubleshooting) covers every problem hit so far,
and `scripts/check_database.py` diagnoses the database connection.
