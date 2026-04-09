"""
AblationMode — six controlled conditions for the behavioral probe study.

Each condition isolates one claimed architectural contribution:

  full             — complete architecture (control)
  llm_only         — LLM receives only raw homeostatic values; all computed
                     interoceptive signals (drive urgency, arousal/valence, EFE)
                     removed from prompt
  no_efe           — EFE scores zeroed before reaching LLM; agent must pick
                     from uniform action scores
  no_interoceptive — AllostaticController output dropped (zero urgency batch);
                     drive urgency and dominant_channel absent from prompt;
                     raw homeostatic values shown instead
  no_arousal       — ArousalValenceSystem output zeroed; arousal/valence absent
                     from prompt and from the high-arousal depth trigger
  efe_only         — deterministic argmax on EFE scores; no LLM call

The null hypothesis these conditions are designed to kill:
  "A sufficiently prompted LLM with access to raw observations achieves the
  same behavioral profile, making the interoceptive/predictive layers
  unnecessary."
"""
from enum import Enum


class AblationMode(str, Enum):
    FULL = "full"
    LLM_ONLY = "llm_only"
    NO_EFE = "no_efe"
    NO_INTEROCEPTIVE = "no_interoceptive"
    NO_AROUSAL = "no_arousal"
    EFE_ONLY = "efe_only"
