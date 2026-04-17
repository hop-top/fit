<?php

declare(strict_types=1);

namespace HopTop\Fit;

interface AdvisorInterface
{
    public function generateAdvice(array $context): Advice;

    public function modelId(): string;
}
