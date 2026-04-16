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

    /**
     * @return array<string, mixed>
     */
    public function toArray(): array
    {
        return [
            'score' => $this->score,
            'breakdown' => $this->breakdown,
            'metadata' => $this->metadata,
        ];
    }
}
