<?php

declare(strict_types=1);

namespace Hop\Fit;

/**
 * Single-dimension stub scorer for testing.
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
