"""
Edge Detection Models - Phase 2

Specialized models for specific betting edges validated in Phase 1:
1. WindTotalModel - High wind games go under
2. RefereeTotalModel - Certain refs have systematic over/under tendencies
3. EnsembleModel - Combine multiple signals for high-confidence bets

Each model follows the EdgeModel interface and returns predictions with confidence levels.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
from enum import Enum


class BetType(Enum):
    """Types of bets the models can recommend"""
    OVER = "OVER"
    UNDER = "UNDER"
    NO_BET = "NO_BET"


class BetSide(Enum):
    """Sides for spread bets"""
    HOME = "HOME"
    AWAY = "AWAY"
    NO_BET = "NO_BET"


@dataclass
class EdgePrediction:
    """
    Prediction from an edge detection model

    Attributes:
        model_name: Name of the model making prediction
        bet_type: OVER, UNDER, or NO_BET for totals
        bet_side: HOME, AWAY, or NO_BET for spreads
        confidence: 0-1 confidence score
        reasoning: Human-readable explanation
        expected_value: Expected value in points (optional)
    """
    model_name: str
    bet_type: BetType = BetType.NO_BET
    bet_side: BetSide = BetSide.NO_BET
    confidence: float = 0.0
    reasoning: str = ""
    expected_value: Optional[float] = None

    def has_bet(self) -> bool:
        """Returns True if model recommends a bet"""
        return (self.bet_type != BetType.NO_BET or
                self.bet_side != BetSide.NO_BET)


class EdgeModel:
    """
    Base class for edge detection models

    Each model should:
    1. Identify specific scenarios where Vegas systematically errs
    2. Return predictions only when edge conditions are met
    3. Provide reasoning for debugging/tracking
    """

    def __init__(self, name: str):
        self.name = name

    def predict(self, game: Dict) -> EdgePrediction:
        """
        Predict edge for a single game

        Args:
            game: Dictionary with game data (from schedules.parquet)

        Returns:
            EdgePrediction with bet recommendation
        """
        raise NotImplementedError("Subclasses must implement predict()")

    def validate_game_data(self, game: Dict, required_fields: List[str]) -> bool:
        """Check if game has required fields"""
        for field in required_fields:
            if field not in game or game[field] is None:
                return False
        return True


class WindTotalModel(EdgeModel):
    """
    High Wind Total Model

    Edge: Games with wind ≥15 mph go UNDER at 53.1% rate (n=324)

    Strategy:
    - If wind ≥ 15 mph and game is outdoors → BET UNDER
    - Optional: add confirmation from baseline model
    """

    def __init__(self,
                 wind_threshold: float = 15.0,
                 require_outdoor: bool = True):
        super().__init__("WindTotal")
        self.wind_threshold = wind_threshold
        self.require_outdoor = require_outdoor

    def predict(self, game: Dict) -> EdgePrediction:
        """
        Predict if wind creates under edge

        Required fields: wind, roof (optional), total_line
        """
        # Validate required data
        if not self.validate_game_data(game, ['wind', 'total_line']):
            return EdgePrediction(
                model_name=self.name,
                reasoning="Missing wind or total_line data"
            )

        wind = game['wind']
        roof = game.get('roof', 'outdoors')

        # Check if outdoor requirement met
        if self.require_outdoor:
            is_outdoor = roof in ['outdoors', 'outdoor', 'open']
            if not is_outdoor:
                return EdgePrediction(
                    model_name=self.name,
                    reasoning=f"Indoor game (roof={roof}), no wind edge"
                )

        # Check wind threshold
        if wind >= self.wind_threshold:
            # High wind → UNDER edge
            confidence = self._calculate_confidence(wind)

            return EdgePrediction(
                model_name=self.name,
                bet_type=BetType.UNDER,
                confidence=confidence,
                reasoning=f"High wind ({wind:.1f} mph) → under edge (53.1% historical)",
                expected_value=-1.06  # Average total error from Phase 1
            )
        else:
            return EdgePrediction(
                model_name=self.name,
                reasoning=f"Wind {wind:.1f} mph below {self.wind_threshold} threshold"
            )

    def _calculate_confidence(self, wind: float) -> float:
        """
        Calculate confidence based on wind speed

        Higher wind → higher confidence (up to a point)
        """
        if wind >= 20:
            return 0.85  # Extreme wind
        elif wind >= 17:
            return 0.75  # Very high
        elif wind >= 15:
            return 0.65  # High (threshold)
        else:
            return 0.0


class RefereeTotalModel(EdgeModel):
    """
    Referee Total Model

    Edge: Certain referees have systematic over/under tendencies
    - Jerome Boger: 59% over (194 games)
    - Shawn Hochuli: 57% under (117 games)
    - Ron Winter: 58% over (57 games)
    - Scott Green: 67% over (52 games)
    - Mike Carey: 56% under (57 games)

    Strategy:
    - If ref in high_scoring_refs → BET OVER
    - If ref in low_scoring_refs → BET UNDER
    - Only use active refs (need to filter retired ones)
    """

    # Phase 1 validated refs (2010-2025 data)
    # NOTE: Some may be retired - filter before using
    HIGH_SCORING_REFS = {
        'Jerome Boger': {'rate': 0.59, 'games': 194, 'active': True},
        'Ron Winter': {'rate': 0.58, 'games': 57, 'active': False},  # Retired
        'Scott Green': {'rate': 0.67, 'games': 52, 'active': False},  # Retired
    }

    LOW_SCORING_REFS = {
        'Shawn Hochuli': {'rate': 0.57, 'games': 117, 'active': True},
        'Mike Carey': {'rate': 0.56, 'games': 57, 'active': False},  # Retired
    }

    def __init__(self,
                 min_games: int = 50,
                 only_active: bool = True):
        super().__init__("RefereeTotal")
        self.min_games = min_games
        self.only_active = only_active

    def predict(self, game: Dict) -> EdgePrediction:
        """
        Predict if referee creates over/under edge

        Required fields: referee, total_line
        """
        # Validate required data
        if not self.validate_game_data(game, ['referee', 'total_line']):
            return EdgePrediction(
                model_name=self.name,
                reasoning="Missing referee or total_line data"
            )

        referee = game['referee']

        # Check high-scoring refs
        if referee in self.HIGH_SCORING_REFS:
            ref_data = self.HIGH_SCORING_REFS[referee]

            # Filter inactive refs if required
            if self.only_active and not ref_data['active']:
                return EdgePrediction(
                    model_name=self.name,
                    reasoning=f"{referee} retired, edge may not persist"
                )

            # Check minimum games threshold
            if ref_data['games'] < self.min_games:
                return EdgePrediction(
                    model_name=self.name,
                    reasoning=f"{referee} only has {ref_data['games']} games (< {self.min_games})"
                )

            return EdgePrediction(
                model_name=self.name,
                bet_type=BetType.OVER,
                confidence=ref_data['rate'],
                reasoning=f"{referee}: {ref_data['rate']:.1%} over rate (n={ref_data['games']})",
                expected_value=1.96  # Jerome Boger avg from Phase 1
            )

        # Check low-scoring refs
        elif referee in self.LOW_SCORING_REFS:
            ref_data = self.LOW_SCORING_REFS[referee]

            # Filter inactive refs if required
            if self.only_active and not ref_data['active']:
                return EdgePrediction(
                    model_name=self.name,
                    reasoning=f"{referee} retired, edge may not persist"
                )

            # Check minimum games threshold
            if ref_data['games'] < self.min_games:
                return EdgePrediction(
                    model_name=self.name,
                    reasoning=f"{referee} only has {ref_data['games']} games (< {self.min_games})"
                )

            return EdgePrediction(
                model_name=self.name,
                bet_type=BetType.UNDER,
                confidence=ref_data['rate'],
                reasoning=f"{referee}: {ref_data['rate']:.1%} under rate (n={ref_data['games']})",
                expected_value=-0.06  # Shawn Hochuli avg from Phase 1
            )

        else:
            return EdgePrediction(
                model_name=self.name,
                reasoning=f"{referee} not in edge ref list"
            )


class EnsembleModel:
    """
    Multi-Model Ensemble

    Strategy:
    - Run all edge models on each game
    - Only bet when multiple models agree (conservative)
    - Higher confidence when more models align

    Modes:
    - CONSERVATIVE: Require ≥2 models to agree
    - AGGRESSIVE: Bet on any single model signal
    """

    def __init__(self, models: List[EdgeModel], mode: str = "CONSERVATIVE"):
        self.models = models
        self.mode = mode

    def predict(self, game: Dict) -> Dict:
        """
        Run all models and combine predictions

        Returns:
            {
                'total_bet': BetType,
                'spread_bet': BetSide,
                'confidence': float,
                'models_agreeing': List[str],
                'all_predictions': List[EdgePrediction]
            }
        """
        # Run all models
        predictions = [model.predict(game) for model in self.models]

        # Filter to models with actual bets
        total_bets = [p for p in predictions if p.bet_type != BetType.NO_BET]

        # Check for consensus
        if self.mode == "CONSERVATIVE":
            return self._conservative_predict(total_bets, predictions)
        else:
            return self._aggressive_predict(total_bets, predictions)

    def _conservative_predict(self, total_bets: List[EdgePrediction],
                             all_predictions: List[EdgePrediction]) -> Dict:
        """
        Conservative: Require ≥2 models to agree
        """
        if len(total_bets) < 2:
            return {
                'total_bet': BetType.NO_BET,
                'spread_bet': BetSide.NO_BET,
                'confidence': 0.0,
                'models_agreeing': [],
                'all_predictions': all_predictions,
                'reasoning': f"Only {len(total_bets)} model(s) triggered (need ≥2)"
            }

        # Check if models agree on direction
        over_bets = [p for p in total_bets if p.bet_type == BetType.OVER]
        under_bets = [p for p in total_bets if p.bet_type == BetType.UNDER]

        if len(over_bets) >= 2:
            # Multiple models say OVER
            avg_confidence = sum(p.confidence for p in over_bets) / len(over_bets)
            models = [p.model_name for p in over_bets]

            return {
                'total_bet': BetType.OVER,
                'spread_bet': BetSide.NO_BET,
                'confidence': avg_confidence,
                'models_agreeing': models,
                'all_predictions': all_predictions,
                'reasoning': f"{len(models)} models agree on OVER: {', '.join(models)}"
            }

        elif len(under_bets) >= 2:
            # Multiple models say UNDER
            avg_confidence = sum(p.confidence for p in under_bets) / len(under_bets)
            models = [p.model_name for p in under_bets]

            return {
                'total_bet': BetType.UNDER,
                'spread_bet': BetSide.NO_BET,
                'confidence': avg_confidence,
                'models_agreeing': models,
                'all_predictions': all_predictions,
                'reasoning': f"{len(models)} models agree on UNDER: {', '.join(models)}"
            }

        else:
            # Models disagree
            return {
                'total_bet': BetType.NO_BET,
                'spread_bet': BetSide.NO_BET,
                'confidence': 0.0,
                'models_agreeing': [],
                'all_predictions': all_predictions,
                'reasoning': f"Models disagree: {len(over_bets)} OVER, {len(under_bets)} UNDER"
            }

    def _aggressive_predict(self, total_bets: List[EdgePrediction],
                           all_predictions: List[EdgePrediction]) -> Dict:
        """
        Aggressive: Bet on any single model signal
        """
        if len(total_bets) == 0:
            return {
                'total_bet': BetType.NO_BET,
                'spread_bet': BetSide.NO_BET,
                'confidence': 0.0,
                'models_agreeing': [],
                'all_predictions': all_predictions,
                'reasoning': "No models triggered"
            }

        # Take highest confidence bet
        best_bet = max(total_bets, key=lambda p: p.confidence)

        return {
            'total_bet': best_bet.bet_type,
            'spread_bet': BetSide.NO_BET,
            'confidence': best_bet.confidence,
            'models_agreeing': [best_bet.model_name],
            'all_predictions': all_predictions,
            'reasoning': f"Single model: {best_bet.reasoning}"
        }


# Convenience function to create default ensemble
def create_default_ensemble(mode: str = "CONSERVATIVE") -> EnsembleModel:
    """
    Create ensemble with Phase 1 validated models

    Args:
        mode: "CONSERVATIVE" (≥2 models) or "AGGRESSIVE" (any model)

    Returns:
        EnsembleModel ready to use
    """
    models = [
        WindTotalModel(wind_threshold=15.0, require_outdoor=True),
        RefereeTotalModel(min_games=50, only_active=True)
    ]

    return EnsembleModel(models, mode=mode)
