<?php

declare(strict_types=1);

namespace Hop\Fit;

interface AdvisorInterface
{
    public function generateAdvice(array $context): Advice;

    public function modelId(): string;
}
