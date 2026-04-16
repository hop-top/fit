<?php

declare(strict_types=1);

namespace Hop\Fit;

readonly class Reward
{
    /**
     * @param float $score
     * @param array<string, float> $breakdown
     * @param array<string, mixed> $metadata
     */
    public function __construct(
        public float $score,
        public array $breakdown,
        public array $metadata = [],
    ) {}
}
