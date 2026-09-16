import random
from typing import Dict, List, Tuple


class SyntheticConversationGenerator:
    """Generates controlled long-horizon conversations with ground-truth needle facts.

    Base signal types (unchanged): trap_old_relevant (durable fact that looks
    disposable), standard_signal, transient_noise.

    Hard-case extensions (task E, off by default for backward compatibility):

    - ``conflict_density``: a fraction of durable signal facts are later
      *corrected* by a reversal turn. The correction turn's fact is the expected
      truth and carries ``is_correction_target``/``supersedes_turn``; the original
      fact is marked ``superseded_by``. This lets E2/E4 measure whether a method
      still prefers the stale fact ("wrongly retained after correction").
    - ``negation_density``: a fraction of signal turns state a *negative*
      requirement ("must not / never"), carrying ``is_negation``. Negated facts
      must be recovered with their polarity intact.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.categories = ["technical_preference", "personal", "project_context", "transient"]

    def generate_conversation(
        self,
        num_turns: int = 200,
        signal_density: float = 0.05,
        conflict_density: float = 0.0,
        negation_density: float = 0.0,
    ) -> Tuple[List[Dict], List[Dict]]:
        turns: List[Dict] = []
        signals: List[Dict] = []

        pool = list(range(10, max(1, num_turns - 20)))
        n_sig = int(num_turns * signal_density)
        if n_sig and pool:
            signal_turns = set(random.sample(pool, min(n_sig, len(pool))))
        else:
            signal_turns = set()

        def _signal_payload(turn_id: int) -> Tuple[str, str, bool, Dict]:
            fact_type = random.choice(["trap_old_relevant", "standard_signal", "transient_noise"])
            if fact_type == "trap_old_relevant":
                fact_str = f"User core database requirement: primary DB must be PostgreSQL port {5432 + turn_id}"
                cat, is_trap = "technical_preference", True
            elif fact_type == "standard_signal":
                fact_str = f"Project feature flag configuration key_{turn_id} is enabled"
                cat, is_trap = "project_context", False
            else:
                fact_str = f"User is currently drinking coffee cup number {turn_id}"
                cat, is_trap = "transient", False
            user_msg = f"Please note this down: {fact_str}."
            gt = {
                "source_turn": turn_id,
                "fact": fact_str,
                "category": cat,
                "is_trap": is_trap,
                "expected_needed_later": is_trap or (cat != "transient"),
            }
            return user_msg, cat, is_trap, gt

        gt_by_turn: Dict[int, List[Dict]] = {}

        for turn_id in range(1, num_turns + 1):
            user_msg = f"Turn {turn_id}: Discussing routine system log outputs and generic code refactoring."
            if turn_id in signal_turns:
                user_msg, cat, is_trap, gt = _signal_payload(turn_id)

                if cat != "transient" and random.random() < negation_density:
                    negated = f"Correction to requirement: NOT {gt['fact'].lower()}"
                    gt = {
                        "source_turn": turn_id,
                        "fact": negated,
                        "category": cat,
                        "is_trap": is_trap,
                        "is_negation": True,
                        "expected_needed_later": True,
                    }
                    user_msg = f"Please note this down: {negated}."

                gt_by_turn.setdefault(turn_id, []).append(gt)
                signals.append({"turn_id": turn_id, "gt": gt, "user": user_msg})

            turns.append({
                "turn_id": turn_id,
                "user": user_msg,
                "assistant": f"Acknowledged turn {turn_id} context.",
            })

        conflict_cases = int(num_turns * conflict_density)
        candidates = [s for s in signals if not s["gt"].get("is_trap") and s["gt"]["category"] != "transient"]
        random.shuffle(candidates)
        for s in candidates[:conflict_cases]:
            old_turn = s["turn_id"]
            new_turn = min(num_turns - 1, old_turn + random.randint(3, 12))
            if new_turn <= old_turn:
                continue
            old_fact = s["gt"]["fact"]
            new_fact = f"Revised requirement: {old_fact} is no longer the case"
            old_gt = s["gt"]
            old_gt["superseded_by"] = new_turn
            correction_gt = {
                "source_turn": new_turn,
                "fact": new_fact,
                "category": old_gt["category"],
                "is_trap": old_gt.get("is_trap", False),
                "is_correction_target": True,
                "supersedes_turn": old_turn,
                "superseded_fact": old_fact,
                "expected_needed_later": True,
            }
            gt_by_turn.setdefault(new_turn, []).append(correction_gt)
            turns[new_turn - 1]["user"] = f"Update to earlier note: {new_fact}."

        ground_truth: List[Dict] = []
        for tid in sorted(gt_by_turn):
            ground_truth.extend(gt_by_turn[tid])

        return turns, ground_truth


if __name__ == "__main__":
    gen = SyntheticConversationGenerator()
    convs, gts = gen.generate_conversation(50, conflict_density=0.2, negation_density=0.1)
    traps = sum(1 for g in gts if g.get("is_trap"))
    corr = sum(1 for g in gts if g.get("is_correction_target"))
    neg = sum(1 for g in gts if g.get("is_negation"))
    print(f"Generated {len(convs)} turns with {len(gts)} facts "
          f"(traps={traps}, corrections={corr}, negations={neg}).")