"""pmbot -- a positive-EV player-prop bot with a spine.

Two independent models have to agree, the price has to beat the de-vigged
market number, the stake comes from fractional Kelly, and every bet lands in
a journal that tracks ROI, win rate and closing line value.

    from pmbot import PropScanner, Settings

    settings = Settings.load()
    for signal in PropScanner(settings).scan("nba"):
        print(signal.headline())
"""

from .config import Settings
from .ev import EvResult, evaluate as evaluate_ev
from .journal import Bet, Journal, summarise
from .kelly import size_bet
from .oddsmath import american_to_decimal, devig, devig_two_way
from .pipeline import PropScanner
from .types import Signal

__version__ = "1.0.0"

__all__ = [
    "Settings",
    "PropScanner",
    "Signal",
    "Journal",
    "Bet",
    "summarise",
    "evaluate_ev",
    "EvResult",
    "size_bet",
    "devig",
    "devig_two_way",
    "american_to_decimal",
    "__version__",
]
