import math
from typing import Dict, List, Tuple


class CategoryDecayEngine:
    """Usage-fed exponential forgetting (brain.md D4).

    Grounded in the forgetting curves of MemoryBank (AAAI 2024), SF-AMS strategic
    forgetting (arXiv:2607.22562) and Oblivion (EMNLP 2026): decay is keyed on
    time-since-*last-access* (not creation), and every access/re-mention
    reinforces the memory's importance with a capped gain instead of a one-shot
    base_score. M(t) = min(1, base_score * (1 + gain * (access_count - 1)))
    * exp(-lambda_c * (current_turn - last_access_turn)).
    """

    def __init__(self, lambdas: Dict[str, float] = None, pruning_threshold: float = 0.20,
                 access_gain: float = 0.2):
        self.lambdas = lambdas or {
            "transient": 0.15,
            "personal": 0.005,
            "technical_preference": 0.02,
            "project_context": 0.01,
        }
        self.pruning_threshold = pruning_threshold
        self.access_gain = access_gain

    def calculate_decayed_importance(self, memory: Dict, current_turn: int) -> float:
        cat = memory.get("category", "transient")
        lambda_c = self.lambdas.get(cat, self.lambdas.get("transient", 0.15))
        last = memory.get("last_access_turn", memory.get("source_turn_id", current_turn))
        delta_t = max(0, current_turn - last)
        base_importance = memory.get("base_score", 1.0)
        access_count = max(1, int(memory.get("access_count", 1)))
        reinforced = min(1.0, base_importance * (1.0 + self.access_gain * (access_count - 1)))
        return reinforced * math.exp(-lambda_c * delta_t)

    def calculate_retention_priority(self, memory: Dict) -> float:
        """Evidence-based long-term survival score (D32/Phase 11).

        Unlike ``calculate_decayed_importance`` this carries NO time-decay term:
        it answers "how valuable is this memory for the future, given the
        evidence we have" while activation declines with time since last access.
        It is the quantity store-budget eviction should use under
        ``retention.mode = dual_score``, so that old-but-valuable facts survive
        a capacity squeeze while low-evidence transient chatter is evicted.

        Retention priority = reinforced base score, independent of the current
        turn: ``min(1, base_score * (1 + access_gain * (access_count - 1)))``.
        Computed only from existing fields (base_score, access_count), so it
        costs no extra embeddings, no LLM calls, and is deterministic. The
        same quantity restated in the spec is ``min(1.0, base_score * (1.0 +
        access_gain * (access_count - 1)))``.
        """
        base_importance = memory.get("base_score", 1.0)
        access_count = max(1, int(memory.get("access_count", 1)))
        return min(1.0, base_importance * (1.0 + self.access_gain * (access_count - 1)))

    def step_decay_and_prune(self, memories: List[Dict], current_turn: int,
                             retention_mode: str = "hard_threshold") -> Tuple[List[Dict], List[Dict]]:
        """Compute both activation and retention scores for every memory, then
        decide survival according to ``retention_mode`` (D32/Phase 11).

        * ``hard_threshold`` (default, production): the existing behavior —
          prune a memory when its decayed ``current_importance`` drops below
          ``pruning_threshold``.
        * ``soft_decay`` / ``dual_score``: ``current_importance`` still decays
          and is still recorded (so retrieval ranking is unchanged and the
          activation signal stays observable), but decay never deletes a
          memory. The store shrinks only when the store-token budget actually
          evicts (see ``budget.token_budget_evict``).
        """
        active = []
        pruned = []
        if retention_mode not in ("hard_threshold", "soft_decay", "dual_score"):
            raise ValueError(
                f"unknown retention_mode: {retention_mode!r} "
                "(expected 'hard_threshold', 'soft_decay' or 'dual_score')")
        for mem in memories:
            m_t = self.calculate_decayed_importance(mem, current_turn)
            mem["current_importance"] = m_t
            mem["retention_priority"] = self.calculate_retention_priority(mem)
            if retention_mode == "hard_threshold":
                if m_t >= self.pruning_threshold:
                    active.append(mem)
                else:
                    last = mem.get("last_access_turn", mem.get("source_turn_id", current_turn))
                    delta_t = max(0, current_turn - last)
                    cat = mem.get("category", "transient")
                    lambda_c = self.lambdas.get(cat, self.lambdas.get("transient", 0.15))
                    mem["prune_reason"] = (
                        f"M(t)={m_t:.3f} < threshold ({self.pruning_threshold}) — "
                        f"{cat} decayed over {delta_t} turns since last access "
                        f"(lambda={lambda_c})"
                    )
                    pruned.append(mem)
            elif retention_mode in ("soft_decay", "dual_score"):
                # Activation decays but survival is not decided by the decay
                # stage. Decayed-but-retained memories are flagged so later
                # analysis can tell "old + low activation + still retained +
                # later recovered by semantic retrieval" apart from pruned.
                if m_t < self.pruning_threshold:
                    mem["retained_below_threshold"] = True
                active.append(mem)
        return active, pruned