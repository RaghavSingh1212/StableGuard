# StableGuard

An RL environment and eval harness for AI agents that review stablecoin payments for fraud.

## Why I built this

Visa settles USDC. So do Stripe, PayPal and a growing list of people who used to be card-only shops.
The interesting part of that is not moving the money, it is deciding which payments to let through.

On card rails you have decades of issuer history, a chargeback window, and the option to claw a payment
back if you got it wrong. On stablecoin rails settlement is fast and final. You get one shot at the
decision, you make it from on-chain signals, and if you are wrong the money is gone.

That is the decision I wanted to model. And I wanted to treat the agent making it as something to be
measured rather than trusted, because "the LLM seemed to do pretty well" is not a thing you can put in
front of a risk team.

So StableGuard is two halves. One generates labelled synthetic stablecoin payments and lets an agent
investigate them by calling tools. The other runs an agent over hundreds of payments and reports what
it actually caught, what it wrongly flagged, how many tool calls it burned and what it cost.

## Try it

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/generate_data.py --n 1000 --fraud-rate 0.15 --seed 42
python scripts/run_eval.py --agent rule_based --n 200      # no API key needed
```

For the Claude agent, copy `.env.example` to `.env` and put your key in it, then:

```bash
python scripts/run_eval.py --agent claude --n 25           # start small, it costs money
```

The dashboard is the fastest way to see what is going on:

```bash
streamlit run dashboard/app.py
```

There is also an API if you want to drive it from somewhere else:

```bash
uvicorn api.main:app --reload     # docs at http://127.0.0.1:8000/docs
```

## How it works

**The data.** `generator.py` builds a small world first: wallets with a home country, an age, a
typical payment size and the merchant categories they actually transact in, plus a merchant directory
and a sanctions list. Then it generates payments inside that world. Everything comes off one seeded
RNG, so `--seed 42` gives you the same dataset every time.

Fraud gets crafted one pattern at a time, which is the only way recall per pattern means anything:

- `velocity`, a burst of payments from one wallet inside an hour
- `geo`, sent from a country the wallet has never sent from
- `sanctions`, a listed wallet on one side of the payment
- `mule`, a wallet opened days ago pushing $5,000 or more straight back out
- `amount_spike`, 10x to 50x what the wallet normally sends
- `merchant_mismatch`, a high risk category the wallet has no history with

Clean payments are kept genuinely clean so the labels are not noisy. The one trap I planted on purpose:
a slice of perfectly legitimate wallets use gambling and crypto exchange categories every single day.
If your model learns "gambling means fraud" it will look great until it starts flagging real customers.

**The tools.** Six read-only lookups: wallet history, sanctions, velocity, amount profile, merchant,
geo. Six calls max per payment, and every call costs reward, so an agent cannot just call everything
and think later. Two of the patterns need two tools combined, which is the part that separates a real
agent from a threshold.

**The environment.** A standard `gymnasium.Env`. One episode is one payment. Reward is +1 for a correct
verdict, -1 for a wrong one, -0.05 per tool call, +0.2 for catching fraud in two calls or fewer, and -1
if you burn the whole budget without deciding. The efficiency bonus only pays for catching fraud, never
for approving, because rewarding a fast approve is rewarding exactly the wrong instinct.

**The agents.** The baseline is a four step checklist: velocity, sanctions, amount ratio, geo. It stops
at the first hit. The Claude agent gets the payment and every tool result so far, and replies with JSON
holding its reasoning and its next action. Same interface, so the evaluator does not care which one it
is running.

## What the numbers say

Both agents reviewed the same 200 payments from `--seed 42`.

The Claude eval is still running. Numbers go here when it finishes.

## What surprised me

The baseline is much harder to beat than I expected. A four line checklist gets F1 0.893 with perfect
precision, which is a good reminder that the boring solution usually deserves a seat at the table.

But the per-pattern breakdown is where it falls apart, and that is the whole argument for this project
existing. The baseline scores 0.00 on `merchant_mismatch`. Not low, zero. It never catches a single one,
because that pattern cannot be expressed as a threshold on any one field. You have to know the category
is high risk and that this specific wallet has no history with it, and those are two different tool
calls that have to be put together.

A single F1 number hides that completely. If I had only reported accuracy I would have shipped something
with a known blind spot and not known it. That is the case for per-pattern recall in any eval harness,
and it is the reason this one reports it by default.

## Honest limitations

The data is synthetic, and I generated it, so the patterns are cleaner than anything real. There is no
adversary adapting to the detector. Each payment is judged on its own with no memory across payments,
which rules out the slow structuring patterns that matter a lot in practice. The cost estimates ignore
the discount on cached input tokens, so they read slightly high.

I would rather state all of that than have someone find it.

## Layout

```
stableguard/     models, generator, tools, env, agents, evaluator, metrics
api/main.py      FastAPI app
dashboard/app.py Streamlit dashboard
scripts/         generate_data.py, run_eval.py
tests/           104 tests
config/          thresholds, budgets, model, synthetic sanctions list
```

## Tests

```bash
pytest -q
```

104 tests. They cover label correctness for every fraud pattern, each tool including its error paths,
the environment's reward and budget rules, both agents, the metrics against a confusion matrix I worked
out by hand, the API endpoints, and a headless render of the dashboard.

The Claude agent is tested against a stub client, so the suite needs no API key and makes no network
calls. I did not want a test suite that costs money to run.

## Next

Things I would do with more time: let an agent see a wallet's recent payments rather than one in
isolation, so structuring becomes detectable. Train a policy against the environment instead of only
evaluating prompted agents. Add an adversarial generator that adapts to whatever the current detector
is good at.
