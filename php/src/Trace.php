<?php

declare(strict_types=1);

namespace Hop\Fit;

readonly class Trace
{
    /**
     * @param array<string, mixed> $input
     * @param array<string, mixed> $frontier
     * @param array<string, mixed> $metadata
     */
    public function __construct(
        public string $id,
        public string $sessionId,
        public string $timestamp,
        public array $input,
        public Advice $advice,
        public array $frontier,
        public Reward $reward,
        public array $metadata = [],
    ) {}

    /**
     * Convert all public readonly properties to an associative array.
     *
     * Nested value objects (Advice, Reward) are converted recursively.
     *
     * @return array<string, mixed>
     */
    public function toArray(): array
    {
        return [
            'id' => $this->id,
            'sessionId' => $this->sessionId,
            'timestamp' => $this->timestamp,
            'input' => $this->input,
            'advice' => $this->advice->toArray(),
            'frontier' => $this->frontier,
            'reward' => $this->reward->toArray(),
            'metadata' => $this->metadata,
        ];
    }
}
