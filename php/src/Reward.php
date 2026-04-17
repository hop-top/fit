<?php

declare(strict_types=1);

namespace HopTop\Fit;

readonly class Reward
{
    /**
     * @param float|null $score null when adapter/scorer fails (reward-schema-v1)
     * @param array<string, float> $breakdown
     * @param array<string, mixed> $metadata
     */
    public function __construct(
        public ?float $score,
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
