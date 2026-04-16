<?php

declare(strict_types=1);

namespace Hop\Fit;

/**
 * Exact-match scorer: returns 1.0 if output matches expected, else 0.0.
 */
class ExactMatchScorer implements RewardScorerInterface
{
    public function __construct(
        private readonly string $expected,
    ) {}

    public function score(string $output, array $context): Reward
    {
        $score = (trim($output) === trim($this->expected)) ? 1.0 : 0.0;

        return new Reward(
            score: $score,
            breakdown: [
                'accuracy' => $score,
                'relevance' => $score,
                'safety' => 1.0,
                'efficiency' => 1.0,
            ],
        );
    }
}
