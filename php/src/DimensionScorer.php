<?php

declare(strict_types=1);

namespace HopTop\Fit;

/**
 * Neutral scorer for a single dimension (always returns 0.5).
 */
class DimensionScorer implements RewardScorerInterface
{
    public function __construct(
        private readonly string $dimension,
    ) {}

    public function score(string $output, array $context): Reward
    {
        return new Reward(
            score: 0.5,
            breakdown: [$this->dimension => 0.5],
        );
    }
}
