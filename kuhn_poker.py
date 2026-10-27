from typing import NamedTuple, List, Generator, Literal, TypeAlias, Callable, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

class Action(Enum):
    NONE  = 0
    CHECK = 1
    BET   = 2
    FOLD  = 3
    CALL  = 4

NO_ACTIONS : int = len(Action) - 1 # We remove the none as its not a valid move
MAX_STATES : int = 12

class CardType(Enum):
    JACK  = 0
    QUEEN = 1
    KING  = 2

class Card(NamedTuple):
    card_type : CardType

    def value(self) -> int:
        return int(self.card_type.value)

@dataclass(slots=True)
class Deck:
    rng  : np.random.Generator

    deck : List[Card] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.deck = [
            Card(
                card_type = card_type
            )
            for card_type in CardType
        ]

    def _shuffle_deck(self) -> None:
        self.rng.shuffle(self.deck)

    def deal_cards(self) -> Generator[Card, None, None]:
        self._shuffle_deck()

        for i in range(2):
            yield self.deck[i]

PlayerId    : TypeAlias = Literal[0, 1]
History     : TypeAlias = tuple[Action, ...]
Observation : TypeAlias = int
Rewards     : TypeAlias = tuple[float, float]
StepResult  : TypeAlias = tuple[Observation | None, Rewards, bool]

class KuhnPokerEnv:
    def __init__(self, seed: int = 42) -> None:
        self.rng  : np.random.Generator = np.random.default_rng(seed)
        self.deck : Deck = Deck(self.rng)

        self.cards          : tuple[Card, Card] | None = None
        self.current_player : PlayerId  = 0
        self.history        : History   = ()
        self.contributions  : List[int] = [1, 1]
        self.terminated     : bool      = True
        self.rewards        : Rewards   = (0.0, 0.0)

    @property
    def pot(self) -> int:
        return sum(self.contributions)

    @staticmethod
    def _opponent(player: PlayerId) -> PlayerId:
        return 1 if player == 0 else 0

    def reset(self) -> Observation:
        """Start a new hand and return Player 0's observation."""
        first_card, second_card = self.deck.deal_cards()
        self.cards = (first_card, second_card)

        self.current_player = 0
        self.history        = ()
        self.contributions  = [1, 1]
        self.terminated     = False
        self.rewards        = (0.0, 0.0)

        return self.observation()

    def observation(self) -> Observation:
        """Return the acting player's card and public action history."""
        if self.cards is None:
            raise RuntimeError("Call reset() before observing.")

        if self.terminated:
            raise RuntimeError("No decision observation after the hand ends.")

        # Observation=(state_id)
        history_to_id : Dict[History, int] = {
            ()                         : 0,
            (Action.CHECK,)            : 1,
            (Action.BET,)              : 2,
            (Action.CHECK, Action.BET) : 3,
        }

        card_id    : int = self.cards[self.current_player].value()
        history_id : int = history_to_id[self.history]

        return card_id * 4 + history_id

    def legal_actions(self) -> tuple[int, ...]:
        """Return the actions available to the acting player."""
        if self.terminated:
            return ()

        if not self.history or self.history[-1] == Action.CHECK:
            return (Action.CHECK.value - 1, Action.BET.value - 1)

        return (Action.FOLD.value - 1, Action.CALL.value - 1)

    def step(self, action: int | Action) -> StepResult:
        """Apply an action and return the next observation, rewards, and done."""
        if self.terminated:
            raise RuntimeError("Call reset() before starting another hand.")

        action = Action(action)

        if action.value - 1 not in self.legal_actions():
            raise ValueError(f"Illegal action: {action}")

        player : PlayerId = self.current_player

        if action in (Action.BET, Action.CALL):
            self.contributions[player] += 1

        self.history += (action,)

        if action == Action.FOLD:
            self._finish(self._opponent(player))

        elif action == Action.CALL:
            self._showdown()

        elif self.history == (Action.CHECK, Action.CHECK):
            self._showdown()

        else:
            self.current_player = self._opponent(player)

        next_observation = (
            None if self.terminated else self.observation()
        )

        return next_observation, self.rewards, self.terminated

    def _showdown(self) -> None:
        """Reveal the cards and award the pot to the higher card."""
        if self.cards is None:
            raise RuntimeError("Cannot resolve a hand before dealing cards.")

        first_card, second_card = self.cards

        winner : PlayerId = (
            0
            if first_card.card_type.value > second_card.card_type.value
            else 1
        )

        self._finish(winner)

    def _finish(self, winner: PlayerId) -> None:
        """Assign net profits in player order and end the hand."""
        loser = self._opponent(winner)

        winner_profit = float(self.pot - self.contributions[winner])
        loser_profit  = -float(self.contributions[loser])

        self.rewards = (
            (winner_profit, loser_profit)
            if winner == 0
            else (loser_profit, winner_profit)
        )

        self.terminated = True


OpponentPolicy : TypeAlias = Callable[[int, tuple[int, ...]], int,]

opponent_rng : np.random.Generator = np.random.default_rng(123)

def random_opponent_policy(obs : int, legal_actions: tuple[int, ...]) -> int:
    if not legal_actions:
        raise ValueError("Cannot choose an action without legal actions.")

    return int(opponent_rng.choice(legal_actions))
    
class TrainingKuhnPokerEnv(KuhnPokerEnv):
    def __init__(self, opponent_policy : OpponentPolicy, agent_id : PlayerId = 0, seed : int = 42) -> None:
        super().__init__(seed)

        self.opponent_policy = opponent_policy
        self.agent_id        = agent_id

    def switch_agent_id(self) -> None:
        self.agent_id = 1 if self.agent_id == 0 else 0

    def reset(self) -> int:
        super().reset()
        self._play_opponent()

        return self.observation()

    def step(self, action_id : int) -> Tuple[int | None, float, bool]:
        if self.terminated:
            raise RuntimeError("Call reset() before starting another hand.")

        if self.current_player != self.agent_id:
            raise RuntimeError("It is not the agent's turn.")

        if action_id not in self.legal_actions():
            raise ValueError(f"Illegal action ID: {action_id}")

        super().step(Action(action_id + 1))

        self._play_opponent()

        next_state : int | None = None if self.terminated else self.observation()

        reward : float = float(self.rewards[self.agent_id])

        return next_state, reward, self.terminated

    def _play_opponent(self) -> None:
        while (
            not self.terminated
            and self.current_player != self.agent_id
        ):
            state = self.observation()
            legal_actions = self.legal_actions()

            action_id = self.opponent_policy(state, legal_actions)

            if action_id not in legal_actions:
                raise ValueError(
                    f"Opponent selected illegal action ID: {action_id}"
                )

            super().step(Action(action_id + 1))



