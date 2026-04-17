<?php

declare(strict_types=1);

namespace HopTop\Fit;

interface AdapterInterface
{
    /**
     * @return array{string, array<string, mixed>}
     */
    public function call(string $prompt, Advice $advice): array;
}
