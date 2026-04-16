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

    /**
     * @return array<string, mixed>
     */
    public function toArray(): array
    {
        return [
            'domain' => $this->domain,
            'steering_text' => $this->steeringText,
            'confidence' => $this->confidence,
            'constraints' => $this->constraints,
            'metadata' => $this->metadata,
            'version' => $this->version,
        ];
    }
}
