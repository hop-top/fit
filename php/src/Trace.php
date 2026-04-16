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
}
