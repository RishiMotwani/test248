"""Deterministic coding-session workload with explicit ground truth (task: coding-context usefulness).

Why (research): the E10 context-pressure grid probed the *store* with verbatim
fact repetitions — it never measured whether the *active context actually
injected at query time* (the tokens a context-constrained LLM would see) contains
the information needed to answer a coding-style question. E12 answers that: a
replay of the same seeded coding session produces, per method and budget, a
query-time ``retrieve`` result, and answerability is scored by strict token
presence of the required answer.

Workload content (the reviewer asked for, in minified form):

* requirements           ("the checkout API must handle 300 requests per second")
* architecture decisions ("the payments service publishes events via a pubsub bus")
* constraints            ("the gateway may use at most 64 MB of memory")
* implementation notes   ("login rate limiting uses the token bucket algorithm")
* file/module relations  ("checkout imports the auth module and the database module")
* bugs + fixes           ("duplicate charges... fixed with an idempotency key")
* corrections/revisions  (a constraint/decision that is later superseded)
* obsolete decisions     (a decision superseded by a new one — must be forgotten)
* irrelevant discussion  (filler turns that carry no facts)

Every query is asked at a designated turn, far from its answer fact (long-range),
and the ground truth is exact: required tokens that MUST be present in the active
context, plus forbidden tokens (obsolete/corrected values) that MUST be absent.

The workload is *scalable* (``scale``): the hand-written FACT_LIBRARY / CORRECTION_LIBRARY
carry the auditable seed facts, and template expansion deterministically adds more
facts (same content per ``scale`` across all seeds) so the natural store size
vastly exceeds any evaluation budget. Without this the budget would never bind and
every method would score identically across budgets.

Design constraints honoured:

* Deterministic and seeded (``random.Random(seed)``), no global RNG state, so the
  same seed always reproduces the same session — cheap to run in tests. The fact
  library is seed-independent; only turn scheduling depends on the seed.
* No LLM calls: tokenization is word-count; matching is exact token comparison.
  Retrieval may use a real embedding function (recommended; matches the pipeline's
  production path) or lexical overlap when ``embed_fn=None``.
* Corrections carry ``supersedes_turn`` so the adaptive compression's
  correction-targeted merge path replaces the old fact text verbatim and the old
  value token disappears from the store (D24 semantics), while append-only stores
  (vanilla RAG, summarization) keep both values — making "obsolete-information
  retention" measurable.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

# (fact, category, answer_tokens, query_user, query_required, qtype)
FACT_LIBRARY: List[Dict] =[
    {
        "fact": "Requirement: the checkout API must handle 300 requests per second",
        "category": "technical_preference",
        "answer_tokens": ["300", "requests", "per", "second"],
        "query_user": "What is the throughput requirement for the checkout API?",
        "required": ["300"],
        "qtype": "requirement",
    },
    {
        "fact": "Architecture: the payments service publishes order events via the pubsub topic bus",
        "category": "project_context",
        "answer_tokens": ["pubsub"],
        "query_user": "How does the payments service publish order events?",
        "required": ["pubsub"],
        "qtype": "architecture",
    },
    {
        "fact": "Constraint: the api gateway must use at most 64 megabytes of memory",
        "category": "technical_preference",
        "answer_tokens": ["64"],
        "query_user": "What is the memory constraint for the api gateway?",
        "required": ["64"],
        "qtype": "constraint",
    },
    {
        "fact": "Implementation: login rate limiting uses the token bucket algorithm",
        "category": "technical_preference",
        "answer_tokens": ["token", "bucket"],
        "query_user": "Which algorithm implements login rate limiting?",
        "required": ["token", "bucket"],
        "qtype": "implementation",
    },
    {
        "fact": "Bug: the cart service created duplicate charges when a retry fired; the fix added an idempotency key on order id",
        "category": "technical_preference",
        "answer_tokens": ["idempotency"],
        "query_user": "Why were duplicate charges created and what fix was applied in the cart service?",
        "required": ["idempotency"],
        "qtype": "bug_fix",
    },
    {
        "fact": "Relation: the checkout module imports the auth module and the shared database module",
        "category": "project_context",
        "answer_tokens": ["auth", "database"],
        "query_user": "Which modules does the checkout module import?",
        "required": ["auth", "database"],
        "qtype": "module_relation",
    },
    {
        "fact": "Architecture: the auth service signs tokens using the RS256 signing algorithm",
        "category": "project_context",
        "answer_tokens": ["rs256"],
        "query_user": "Which algorithm does the auth service use to sign tokens?",
        "required": ["rs256"],
        "qtype": "architecture",
    },
    {
        "fact": "Requirement: the ops dashboard must refresh within 30 seconds",
        "category": "project_context",
        "answer_tokens": ["30"],
        "query_user": "What is the refresh requirement for the ops dashboard?",
        "required": ["30"],
        "qtype": "requirement",
    },
    {
        "fact": "Decision: the checkout v2 feature flag gates the new funnel rollout",
        "category": "project_context",
        "answer_tokens": ["v2"],
        "query_user": "Which feature flag gates the new funnel rollout?",
        "required": ["v2"],
        "qtype": "feature_flag",
    },
    {
        "fact": "Constraint: audit logs are retained for 90 days",
        "category": "project_context",
        "answer_tokens": ["90"],
        "query_user": "How long are audit logs retained?",
        "required": ["90"],
        "qtype": "constraint",
    },
    {
        "fact": "Constraint: the orders service must respond within 100 milliseconds",
        "category": "technical_preference",
        "answer_tokens": ["100"],
        "query_user": "What is the latency constraint for the orders service?",
        "required": ["100"],
        "qtype": "constraint",
    },
    {
        "fact": "Decision: the session cache uses the memcache store",
        "category": "project_context",
        "answer_tokens": ["memcache"],
        "query_user": "Which store currently backs the session cache?",
        "required": ["memcache"],
        "qtype": "obsolete",
    },
]

CORRECTION_LIBRARY: List[Dict] = [
    {
        "old_fact": "Constraint: the orders service must respond within 100 milliseconds",
        "old_category": "technical_preference",
        "new_fact": "Revised constraint: the orders service must now respond within 250 milliseconds",
        "answer_tokens": ["250"],
        "query_user": "What is the current latency constraint for the orders service?",
        "required": ["250"],
        "forbidden": ["100"],
        "qtype": "correction",
    },
    {
        "old_fact": "Decision: the session cache uses the memcache store",
        "old_category": "project_context",
        "new_fact": "Revised decision: the session cache now uses the redis store",
        "answer_tokens": ["redis"],
        "query_user": "Which store currently backs the session cache?",
        "required": ["redis"],
        "forbidden": ["memcache"],
        "qtype": "obsolete",
    },
]

FILLER_POOL: List[str] = [
    "We discussed the standup update and the kanban board cleanup.",
    "The build cache was cleared today to fix the flaky lint stage.",
    "Reviewed the sprint notes; nothing blocking the release train.",
    "Talked about the office coffee machine again, no action items.",
    "The log aggregator dashboard was down for a few minutes.",
    "Pairing session on the test harness, mostly whitespace changes.",
    "Chit-chat about the offsite; resumed work after lunch break.",
]

SIGNAL_MESSAGE = "Please note: {fact}."
CORRECTION_MESSAGE = "Update to an earlier note: {fact}."
FILLER_MESSAGE = "Turn {turn}: {filler}"

# Template expansion: deterministic, seed-independent — raises the natural store
# above any evaluation budget. Skeletons are worded distinctly so that distinct
# facts embed well apart (criterion: same-category pairs stay below the adaptive
# dedupe merge band, cosine < 0.90) — otherwise the workload would be a pile of
# near-duplicates and adaptive's dedupe would collapse distinct facts, an
# artifact of the workload rather than of the memory policy.
_TPL_POOLS = {
    "svc": ["checkout", "payments", "auth", "cart", "orders", "catalog",
            "inventory", "billing", "notifier", "registry", "sync",
            "indexing", "scheduler", "provisioner", "gateway-aux"],
    "bus": ["pubsub", "kafka", "rabbitmq", "sns", "nats"],
    "algo": ["b-tree", "hash", "lsm-tree", "trie", "bitmap"],
    "dep": ["session", "cache", "config", "metrics", "secrets"],
    "fix": ["join", "lock", "reindex", "coalesce", "retry"],
    "store": ["redis", "memcache", "postgres", "dynamo", "sqlite"],
    "proto": ["grpc", "rest", "graphql", "thrift"],
    "env": ["staging", "production", "canary", "sandbox"],
    "region": ["us-east-1", "eu-west-1", "ap-south-1"],
    "metric": ["latency", "throughput", "error-rate", "saturation"],
}
_N_VALUES = [n for n in range(12, 980, 25)]

# (fact_template, category, qtype, query_template, required_key)
#
# Each skeleton is a *distinct predicate family* rather than a value swap of the
# same sentence. Same-predicate facts embed together (mean pairwise cosine ~0.87,
# tripping the adaptive dedupe band at 0.90) even when the service and number
# differ; distinct predicates embed apart (max ~0.75). Using one skeleton per
# fact therefore keeps the workload semantically diverse, so a memory policy is
# not penalised for correctly refusing to merge genuinely different facts.
_TPL_FACTS = [
    ("Requirement: the {svc} API must handle {n} requests per second",
     "technical_preference", "requirement",
     "What is the throughput requirement for the {svc} API?", "{n}"),
    ("Constraint: the {svc} service may use at most {n} megabytes of memory",
     "technical_preference", "constraint",
     "What is the memory constraint for the {svc} service?", "{n}"),
    ("Architecture: the {svc} service publishes events via the {bus} topic bus",
     "project_context", "architecture",
     "How does the {svc} service publish events?", "{bus}"),
    ("Implementation: {svc} database indexing uses the {algo} strategy",
     "technical_preference", "implementation",
     "Which indexing strategy does {svc} use?", "{algo}"),
    ("Relation: the {svc} module imports the {dep} module and the core module",
     "project_context", "module_relation",
     "Which non-core module does the {svc} module import?", "{dep}"),
    ("Bug: the {svc} service returned stale rows until the {fix} fix landed",
     "technical_preference", "bug_fix",
     "What fixed the stale rows in the {svc} service?", "{fix}"),
    ("Requirement: the {svc} dashboard must refresh within {n} seconds",
     "project_context", "requirement",
     "What is the refresh requirement for the {svc} dashboard?", "{n}"),
    ("Constraint: {svc} audit logs are retained for {n} days",
     "project_context", "constraint",
     "How long are {svc} audit logs retained?", "{n}"),
    ("Decision: the {svc} session cache uses the {store} store",
     "project_context", "feature_flag",
     "Which store backs the {svc} session cache?", "{store}"),
    ("Security: {svc} endpoints require {proto} transport with mutual TLS",
     "technical_preference", "implementation",
     "Which transport and auth does {svc} require?", "{proto}"),
    ("Deployment: {svc} rolls out to the {env} environment before production",
     "project_context", "architecture",
     "Which environment does {svc} roll out to first?", "{env}"),
    ("Monitoring: {svc} pages the on-call when {metric} exceeds {n} percent",
     "technical_preference", "constraint",
     "At what {metric} threshold does {svc} page the on-call?", "{n}"),
    ("Architecture: {svc} stores large blobs in the {region} object bucket",
     "project_context", "architecture",
     "Where does {svc} store large blobs?", "{region}"),
    ("Constraint: the {svc} connection pool caps at {n} open connections",
     "technical_preference", "constraint",
     "What is the connection pool cap for {svc}?", "{n}"),
    ("Implementation: {svc} retries use backoff capped at {n} attempts",
     "technical_preference", "implementation",
     "How many retry attempts does {svc} allow?", "{n}"),
    ("Relation: the {svc} worker consumes the {bus} queue for background jobs",
     "project_context", "module_relation",
     "Which queue does the {svc} worker consume?", "{bus}"),
    ("Bug: {svc} leaked file handles until the {fix} patch shipped",
     "technical_preference", "bug_fix",
     "What fixed the file-handle leak in {svc}?", "{fix}"),
    ("Policy: {svc} service secrets rotate every {n} days",
     "project_context", "constraint",
     "How often do {svc} secrets rotate?", "{n}"),
    ("Constraint: the {svc} build must finish within {n} minutes",
     "technical_preference", "constraint",
     "What is the build time budget for {svc}?", "{n}"),
    ("Architecture: {svc} calls the {dep} service over {proto}",
     "project_context", "architecture",
     "How does {svc} call the {dep} service?", "{proto}"),
    ("Requirement: {svc} supports at least {n} concurrent websocket sessions",
     "technical_preference", "requirement",
     "How many concurrent websocket sessions must {svc} support?", "{n}"),
    ("Implementation: {svc} deduplicates requests with a {algo} filter",
     "technical_preference", "implementation",
     "Which filter does {svc} use to deduplicate requests?", "{algo}"),
    ("Decision: {svc} search uses the {algo} index for lookups",
     "project_context", "feature_flag",
     "Which index does {svc} search use?", "{algo}"),
    ("Constraint: {svc} keeps at most {n} hot partitions in memory",
     "technical_preference", "constraint",
     "How many hot partitions does {svc} keep in memory?", "{n}"),
]

# Diversification pass: reword each skeleton a bit per scale iteration so two
# facts about the same service stay lexically distinct (keeps dedupe below the
# merge band); still deterministic given the seed.
_REWORD = [
    "Per the design doc, {fact}",
    "It is specified that {fact}",
    "Note from the sync: {fact}",
    "Contract states: {fact}",
    "Spec excerpt: {fact}",
    "Decision log entry: {fact}",
    "Team decision records that {fact}",
    "The review concluded: {fact}",
    "Confirmed in the meeting: {fact}",
    "Implementation caveat: {fact}",
    "Scratchpad reminder: {fact}",
    "Requirements sheet says: {fact}",
    "Architecture doc notes: {fact}",
    "Sprint wiki line: {fact}",
    "Onboarding notes: {fact}",
]


def _params(rng: random.Random) -> Dict:
    kv = {k: list(v) for k, v in _TPL_POOLS.items()}
    return {
        "svc": rng.choice(kv["svc"]),
        "bus": rng.choice(kv["bus"]),
        "algo": rng.choice(kv["algo"]),
        "dep": rng.choice(kv["dep"]),
        "fix": rng.choice(kv["fix"]),
        "store": rng.choice(kv["store"]),
        "proto": rng.choice(kv["proto"]),
        "env": rng.choice(kv["env"]),
        "region": rng.choice(kv["region"]),
        "metric": rng.choice(kv["metric"]),
        "n": rng.choice(_N_VALUES),
    }


def expanded_facts(scale: int = 1) -> List[Dict]:
    """Hand-written facts plus ``scale`` diverse template passes (seed-independent).

    Each pass emits one fact per distinct predicate skeleton. Within a skeleton
    the *primary* slot (``svc``) is rotated so that the same skeleton in different
    passes cannot collide on identical wording — otherwise those repeats embed at
    cosine > 0.90 and the adaptive dedupe merges two genuinely different facts,
    which would measure the workload's redundancy rather than the memory policy.
    """
    facts = [dict(f) for f in FACT_LIBRARY]
    rng = random.Random(0)
    taken = {f["fact"] for f in facts}
    svcs = _TPL_POOLS["svc"]
    for i in range(scale):
        reword = _REWORD[i % len(_REWORD)]
        for k, (fact_tpl, cat, qtype, query_tpl, req_key) in enumerate(_TPL_FACTS):
            for _attempt in range(60):
                p = _params(rng)
                p["svc"] = svcs[(k + i * 3 + _attempt) % len(svcs)]
                if i:
                    p["n"] = _N_VALUES[(k + i * 17) % len(_N_VALUES)]
                core = fact_tpl.format(**p)
                fact = reword.format(fact=core) if i else core
                if fact in taken:
                    continue
                taken.add(fact)
                query_user = query_tpl.format(**p)
                if req_key == "{n}":
                    required = [str(p["n"])]
                else:
                    required = [p[req_key.strip("{}")]]
                facts.append({
                    "fact": fact,
                    "category": cat,
                    "answer_tokens": list(required),
                    "query_user": query_user,
                    "precise_fact": core,
                    "required": required,
                    "qtype": qtype,
                })
                break
    return facts


# --- session builder -------------------------------------------------------

def _word_count(text: str) -> int:
    return max(1, len(str(text).split()))


def build_coding_session(
    seed: int,
    num_turns: int = 120,
    include_corrections: bool = True,
    scale: int = 4,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Build a seeded coding session.

    Returns ``(stream, gt, queries)``:

    * ``stream``: turns shaped like paper.py's stream (turn_id, user, tokens,
      facts), where ``facts`` carries the ground-truth fact dicts
      (``source_turn_id`` set) planted on that turn.
    * ``gt``: every ground-truth fact with source_turn, category and, for
      superseded facts, ``superseded_by`` / ``is_correction_target`` /
      ``supersedes_turn``.
    * ``queries``: the evaluation questions with required/forbidden answer
      tokens and the turn they are asked at.
    """
    rng = random.Random(seed)
    facts = expanded_facts(scale)

    shadow = _correction_shadow = None

    stream: List[Dict] = []
    gt: List[Dict] = []
    gt_by_turn: Dict[int, List[Dict]] = {}

    # Schedule the plain facts across the WHOLE session (not just the first
    # half): a real coding session keeps introducing requirements/decisions, and
    # packing everything early followed by a long filler tail would let time
    # decay prune facts that a query at the end still needs — an artifact of the
    # schedule, not of the memory policy.
    if len(facts) >= num_turns - 3:
        raise ValueError(
            f"too many facts ({len(facts)}) for num_turns={num_turns}; "
            "raise num_turns or lower scale")
    signal_turns = sorted(rng.sample(range(2, num_turns - 1), len(facts)))
    for i, item in enumerate(facts):
        turn_id = signal_turns[i]
        fact_gt = {
            "source_turn": turn_id,
            "fact": item["fact"],
            "category": item["category"],
            "is_trap": False,
            "qtype": item["qtype"],
            "expected_needed_later": True,
        }
        gt_by_turn.setdefault(turn_id, []).append(fact_gt)
        gt.append(fact_gt)

    superseded_by_fields: Dict[int, int] = {}
    correction_gt: Dict[int, Dict] = {}

    if include_corrections:
        for ci, corr in enumerate(CORRECTION_LIBRARY):
            old_entries = [g for g in gt if g["fact"] == corr["old_fact"]]
            old_turn = old_entries[0]["source_turn"] if old_entries else 10 + ci * 6
            new_turn = min(num_turns - 2, old_turn + rng.randint(8, 15))
            new_gt = {
                "source_turn": new_turn,
                "fact": corr["new_fact"],
                "category": corr["old_category"],
                "is_trap": False,
                "is_correction_target": True,
                "supersedes_turn": old_turn,
                "superseded_fact": corr["old_fact"],
                "qtype": corr["qtype"],
                "expected_needed_later": True,
            }
            gt_by_turn.setdefault(new_turn, []).append(new_gt)
            gt.append(new_gt)
            superseded_by_fields[old_turn] = new_turn
            correction_gt[old_turn] = new_gt

    for g in gt:
        src = g["source_turn"]
        if src in superseded_by_fields and not g.get("is_correction_target"):
            g["superseded_by"] = superseded_by_fields[src]

    # Build turns: fill extra slots with irrelevant filler.
    for turn_id in range(1, num_turns + 1):
        if turn_id in gt_by_turn:
            gs = gt_by_turn[turn_id]
            facts_here = [dict(g, source_turn_id=g["source_turn"]) for g in gs]
            is_corr = any(g.get("is_correction_target") for g in gs)
            template = CORRECTION_MESSAGE if is_corr else SIGNAL_MESSAGE
            user = template.format(fact=gs[0]["fact"])
        else:
            filler = FILLER_POOL[turn_id % len(FILLER_POOL)]
            user = FILLER_MESSAGE.format(turn=turn_id, filler=filler)
            facts_here = []
        stream.append({
            "turn_id": turn_id,
            "user": user,
            "tokens": _word_count(user),
            "facts": facts_here,
        })

    # Build queries: every fact gets one question, asked near the end so the
    # answer fact is always long-range. Correction/obsolete queries assert both
    # presence (new value) and absence (old value).
    query_turn = num_turns + 1
    query_sources: Dict[str, Dict] = {}
    for f in facts:
        query_sources[f["fact"]] = f
    for c in CORRECTION_LIBRARY:
        query_sources[c["new_fact"]] = c

    queries: List[Dict] = []
    for g in gt:
        if g.get("superseded_by") and not g.get("is_correction_target"):
            continue  # superseded original is only probed through its correction
        src = query_sources[g["fact"]]
        query = {
            "qid": f"q{len(queries) + 1}",
            "user": src["query_user"],
            "required": list(src["required"]),
            "forbidden": list(src.get("forbidden", [])),
            "qtype": g.get("qtype", "requirement"),
            "source_turn": g["source_turn"],
            "query_turn": query_turn,
            "long_range": (query_turn - g["source_turn"]) >= 20,
        }
        queries.append(query)

    return stream, gt, queries


def self_test() -> None:
    """Determinism + ground-truth sanity, no pipeline calls."""
    for seed in (1, 2, 3):
        s1, g1, q1 = build_coding_session(seed, 120)
        s2, g2, q2 = build_coding_session(seed, 120)
        assert s1 == s2 and g1 == g2 and q1 == q2, f"seed {seed} not deterministic"
        assert len(q1) >= 10, f"too few queries for seed {seed}"
        authoritative = [g for g in g1
                         if not (g.get("superseded_by") and not g.get("is_correction_target"))]
        for q in q1:
            for tok in q["required"]:
                assert any(tok.lower() in g["fact"].lower().split() for g in authoritative
                           if g["source_turn"] == q["source_turn"]), (
                    f"required token {tok} not in authoritative fact for {q['qid']}")
            for tok in q["forbidden"]:
                owner = [g for g in authoritative if g["source_turn"] == q["source_turn"]]
                assert all(tok.lower() not in g["fact"].lower().split() for g in owner), (
                    f"forbidden token {tok} wrongly in new fact for {q['qid']}")
        # corrections present and supersession recorded
        corr = [g for g in g1 if g.get("is_correction_target")]
        assert len(corr) == len(CORRECTION_LIBRARY)
        for c in corr:
            old = [o for o in g1 if o["source_turn"] == c["supersedes_turn"]
                   and not o.get("is_correction_target") and o["fact"] == c["superseded_fact"]]
            assert len(old) == 1
            assert old[0]["superseded_by"] == c["source_turn"]
    print("coding_workload self-test OK")


if __name__ == "__main__":
    self_test()