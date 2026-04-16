<?php

declare(strict_types=1);

namespace Hop\Fit;

readonly class Advice
{
    public function __construct(
        public string $domain,
        public string $steeringText,
        public float $confidence,
        public array $constraints = [],
        public array $metadata = [],
        public string $version = '1.0',
    ) {}
}
